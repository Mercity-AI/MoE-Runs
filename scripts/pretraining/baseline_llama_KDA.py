"""Train the 1B LLaMA baseline with Kimi Delta Attention (hybrid KDA + softmax).

Same dense backbone and TorchTitan training loop as ``baseline_llama_torchtitan``,
but the softmax attention modules are replaced by a config-driven hybrid of Kimi
Delta Attention (KDA, a gated-delta *linear* attention) and standard softmax
attention. KDA compute runs through flash-linear-attention's Triton ``chunk_kda``,
which has a full autograd backward. (Moonshot's FlashKDA CUTLASS kernels are
forward/inference-only and are never used in training.) No n-gram embeddings —
this is the plain baseline architecture plus the longcat script's niceties
(QK-norm on softmax layers, richer logging) and its final-checkpoint save fix.
"""

import math
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel

from data import PackedFineWebDataset, build_dataloader
from model import assert_kda_training_kernels, KDA_TRAINING_MODE
from utils import (
    build_model,
    build_torchtitan_ce_loss,
    build_torchtitan_tokenizer,
    ensure_hf_assets,
    load_tokenizer_size,
    require_torchtitan,
    DualLRScheduler,
    barrier,
    build_job_config,
    build_optimizer,
    build_scheduler,
    cleanup_runtime,
    evaluate,
    format_optimizer_log,
    forward_loss,
    get_autocast_context,
    get_rank,
    get_world_size,
    grad_norm_for,
    is_main_process,
    load_checkpoint,
    maybe_init_tracker,
    NativeMuonWithAuxAdam,
    print0,
    progress_bar,
    reduce_sum_scalar,
    resolve_attention_forward_kwargs,
    run_lm_benchmarks,
    save_checkpoint,
    seed_everything,
    setup_runtime,
)


CONFIG = {
    "hidden_size": 1536,
    "num_hidden_layers": 32,
    "num_attention_heads": 12,
    "num_key_value_heads": 6,
    "intermediate_size": 5120,
    "vocab_size": 32000,
    "max_position_embeddings": 8192,
    "rope_theta": 1_000_000,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    # Softmax (full-attention) layers use this backend; KDA layers ignore it.
    "attn_implementation": "flash_attention_4",
    "tokenizer_name": "meta-llama/Llama-2-7b",
    "hf_assets_dir": "./hf_assets_llama_1b_kda_3to1_h12",
    # --- KDA hybrid layout ---------------------------------------------------
    # KDA is the SPARSE attention type here: 8 KDA + 24 GQA over 32 layers, placed
    # by RULE instead of a hand-listed set. `kda_every` = one KDA layer every N
    # layers, at i % kda_every == kda_offset; every other layer is GQA (full).
    #   kda_every=4, kda_offset=0  ->  KDA at 0,4,8,...,28  ->  layout kFFF kFFF ...
    # (one KDA at the start of each block of 4, followed by three GQA).
    # Priority in resolve_kda_layer_types (first set wins):
    #   kda_full_attn_layers > kda_full_attn_range > kda_full_attn_every > kda_every.
    # To instead put the KDA layer AT THE END of each block ("3 GQA then 1 KDA",
    # KDA at 3,7,...,31), set kda_offset=3 -- note that is a different placement than
    # the one benchmarked/verified this session.
    "kda_full_attn_layers": None,
    "kda_full_attn_range": None,
    "kda_full_attn_every": None,
    "kda_every": 4,
    "kda_offset": 0,
    # --- KDA layer hyperparameters (fla KimiDeltaAttention) ------------------
    # head_dim 128 matches Kimi-Linear. 12 heads makes q/k/v dim == hidden_size
    # (1536), same width as the residual stream, +7.4% vs the 1.031B baseline --
    # outside the report's 5% fairness window, so a win here isn't isolated from
    # the extra capacity. 8 heads (1024) was the param-matched control (-0.2%);
    # 10 heads (1280, +3.6%) is the flagship's over-provisioned default.
    "kda_head_dim": 128,
    "kda_num_heads": 12,
    "kda_num_v_heads": None,
    "kda_expand_v": 1.0,
    "kda_use_short_conv": True,
    "kda_conv_size": 4,
    "kda_conv_bias": False,
    "kda_allow_neg_eigval": False,
    # lower_bound/safe_gate only matter for FlashKDA *inference* serving (its
    # kernel requires safe_gate=True). Training uses Triton; leave both unset.
    "kda_lower_bound": None,
    "kda_safe_gate": False,
    # QK-RMSNorm (Qwen3/Gemma-style) on the *softmax* layers only. KDA layers
    # L2-normalize q/k inside their own kernel, so this does not touch them.
    "qk_norm": True,
    "qk_norm_eps": 1e-6,
    "use_liger_kernel": True,
    "liger_kernel_config": {
        "rope": True,
        "swiglu": True,
        "rms_norm": True,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    },
    "optimizer_name": "muon",
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "optimizer_eps": 1e-8,
    "optimizer_implementation": "fused",
    "muon_lr": 0.02,
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_weight_decay": 0.1,
    "aux_adam_lr": 3e-4,
    "aux_adam_beta1": 0.9,
    "aux_adam_beta2": 0.95,
    "aux_adam_eps": 1e-8,
    "aux_adam_weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 150,
    "lr_decay_type": "cosine",
    "lr_decay_ratio": None,
    "min_lr_factor": 0.1,
    "target_train_tokens": None,
    "max_steps": 3053,
    "per_device_batch_size": 10,
    "grad_accum_steps": 24,
    "max_seq_len": 8192,
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_streaming_buffer_size": 1,
    "eval_max_examples": 512,
    "eval_holdout_fraction": 0.005,
    "cross_document_attention": True,
    "eval_every_steps": 250,
    "eval_benchmarks": False,
    "save_every_steps": 250,
    "log_every_steps": 1,
    "seed": 42,
    "resume_from_checkpoint": None,
    "init_from_checkpoint": None,
    "allow_inexact_legacy_data_resume": False,
    "output_dir": "./checkpoints_llama_1b_kda_3to1_h12_6b",
    "sync_checkpoints_to_bucket": False,
    "checkpoint_bucket_folder": "checkpoints_llama_1b_kda_3to1_h12_6b_1308",
    "use_wandb": True,
    "wandb_project": "llama-1b-6b-torchtitan",
    "wandb_run_name": "llama-1b-kda-3to1-h12-1308",
    "wandb_log_console": False,
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,
    "torch_compile_mode": "default",
}


def _kda_gate_decay_stats(kda_module, gate_input: torch.Tensor) -> tuple[float, float, float]:
    """Per-step forget-gate retained fraction exp(g), g = -exp(A_log)*softplus(f_proj(x)+dt_bias)
    (fla's own KimiDeltaAttention formula). Unbounded below (safe_gate=False here, so
    kda_lower_bound is inert) -- mean near 0 means a layer is forgetting almost everything
    every step, mean near 1 means it has stopped forgetting at all; either is the signature
    of a saturated gate, not just a large train/grad_norm.
    """
    with torch.no_grad():
        num_v_heads, head_k_dim = kda_module.num_v_heads, kda_module.head_k_dim
        g = gate_input.float().view(*gate_input.shape[:-1], num_v_heads, head_k_dim)
        dt_bias = kda_module.dt_bias.float().view(num_v_heads, head_k_dim)
        a_log = kda_module.A_log.float().view(num_v_heads, 1)
        decay = (-torch.exp(a_log) * torch.nn.functional.softplus(g + dt_bias)).exp()
    return decay.mean().item(), decay.min().item(), decay.max().item()


def train(config: dict):
    require_torchtitan()
    device = setup_runtime(distributed=True)
    Path(config["output_dir"]).resolve().mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    tracker = maybe_init_tracker(config, config.get("wandb_run_name") or "llama_kda_torchtitan")

    layer_execution = None
    try:
        if is_main_process():
            assets_dir, tokenizer_size = ensure_hf_assets(config, "kda")
        else:
            assets_dir = Path(config["hf_assets_dir"]).resolve()
            tokenizer_size = 0
        barrier()
        if not is_main_process():
            tokenizer_size = load_tokenizer_size(assets_dir)

        tokenizer = build_torchtitan_tokenizer(assets_dir)
        resume_path = config.get("resume_from_checkpoint")
        init_path = config.get("init_from_checkpoint")
        if resume_path and init_path:
            raise ValueError("Choose resume_from_checkpoint or init_from_checkpoint, not both.")
        if resume_path or init_path:
            model, start_step, optimizer_state, scheduler_state = load_checkpoint(
                config,
                "kda",
                checkpoint_path=resume_path or init_path,
                resume_training=bool(resume_path),
            )
        else:
            model = build_model(config, assets_dir, "kda")
            start_step, optimizer_state, scheduler_state = 0, None, None
        model.to(device)
        if config.get("verify_layer_execution", False):
            layer_execution = {
                idx: {"type": kind, "forward": 0, "backward": 0}
                for idx, kind in enumerate(getattr(model.config, "kda_layer_types", []))
            }
            for layer_idx, layer in enumerate(model.model.layers):
                def count_forward(_module, _inputs, _output, idx=layer_idx):
                    layer_execution[idx]["forward"] += 1

                layer.self_attn.register_forward_hook(count_forward)
                probe = next(param for param in layer.self_attn.parameters() if param.requires_grad)

                def count_backward(grad, idx=layer_idx):
                    layer_execution[idx]["backward"] += 1
                    return grad

                probe.register_hook(count_backward)

        kda_gate_probe: dict[int, torch.Tensor] = {}
        kda_gate_modules: dict[int, torch.nn.Module] = {}
        for layer_idx, kind in enumerate(getattr(model.config, "kda_layer_types", [])):
            if kind != "kda":
                continue
            kda_module = model.model.layers[layer_idx].self_attn.kda
            kda_gate_modules[layer_idx] = kda_module

            def _capture_gate_input(_module, _inputs, output, idx=layer_idx):
                kda_gate_probe[idx] = output.detach()

            kda_module.f_proj.register_forward_hook(_capture_gate_input)

        if tokenizer_size != model.config.vocab_size:
            raise ValueError("Tokenizer and model vocabulary sizes differ.")
        if config["torch_compile"]:
            model = torch.compile(model, mode=config["torch_compile_mode"])
        if get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
            )

        raw_data = load_dataset(
            config["dataset_name"],
            name=config["dataset_config"],
            split="train",
            streaming=True,
        )
        if not config.get("cross_document_attention", True):
            raise NotImplementedError("Document-masked attention is not implemented.")
        print0("[Packing] KDA carries one causal stream; attention crosses packed documents.")
        train_data = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"],
            start_batch=start_step * config["grad_accum_steps"],
            partition="train",
            rank=get_rank(),
            world_size=get_world_size(),
            base_dataset=raw_data,
        )
        eval_data = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"] + 9999,
            partition="eval",
            rank=get_rank(),
            world_size=get_world_size(),
            base_dataset=raw_data,
        )
        train_loader = build_dataloader(train_data, config["per_device_batch_size"], config)
        eval_loader = build_dataloader(eval_data, config["per_device_batch_size"], config)

        job = build_job_config(config, get_world_size())
        max_steps = job.training.steps
        if start_step > max_steps:
            raise ValueError(f"Checkpoint step {start_step} exceeds max_steps={max_steps}.")
        optimizer = build_optimizer(model, job, config)
        scheduler = build_scheduler(optimizer, job)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        attention_kwargs = resolve_attention_forward_kwargs(config)
        use_flce = bool(config.get("liger_kernel_config", {}).get("fused_linear_cross_entropy"))
        loss_fn = None if use_flce else build_torchtitan_ce_loss()

        tokens_per_step = (
            config["per_device_batch_size"]
            * config["grad_accum_steps"]
            * (config["max_seq_len"] - 1)
            * get_world_size()
        )
        print0(f"[Train] {max_steps:,} steps | {tokens_per_step * max_steps / 1e9:.3f}B tokens")
        for line in format_optimizer_log(config, job):
            print0(line)

        train_iter = iter(train_loader)
        model.train()
        n_kda = assert_kda_training_kernels(model)
        print0(f"[KDA] Verified {n_kda} KDA layers on the '{KDA_TRAINING_MODE}' training kernel.")
        optimizer.zero_grad(set_to_none=True)
        bar = progress_bar(
            range(start_step + 1, max_steps + 1),
            desc="train",
            total=max_steps,
            initial=start_step,
        )
        for step in bar:
            started = time.time()
            loss_accum = torch.zeros((), device=device, dtype=torch.float32)
            token_count = 0
            for micro_step in range(config["grad_accum_steps"]):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                sync = micro_step == config["grad_accum_steps"] - 1
                context = nullcontext() if sync or not isinstance(model, DistributedDataParallel) else model.no_sync()
                with context, get_autocast_context(device):
                    micro_loss, micro_tokens = forward_loss(
                        model, loss_fn, batch["input_ids"], device, use_flce, attention_kwargs
                    )
                    (micro_loss / config["grad_accum_steps"]).backward()
                # Accumulate the (loss * tokens) product on-device. Calling .item()
                # here would force a CUDA sync every micro-step (grad_accum_steps-1
                # extra syncs per optimizer step), stalling CPU run-ahead and the
                # kernel-launch pipeline. We sync once, after the loop, instead.
                loss_accum += micro_loss.detach().float() * micro_tokens
                token_count += micro_tokens

            muon_grad_norm = adam_grad_norm = None
            if isinstance(optimizer, NativeMuonWithAuxAdam):
                muon_grad_norm = grad_norm_for(p for g in optimizer.muon.param_groups for p in g["params"])
                adam_grad_norm = grad_norm_for(p for g in optimizer.adam.param_groups for p in g["params"])
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"]).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = loss_accum.item()  # single per-step device sync for logging
            total_loss = reduce_sum_scalar(loss_sum, device)
            total_tokens = reduce_sum_scalar(float(token_count), device)
            mean_loss = total_loss / max(total_tokens, 1.0)
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - started
            metrics = {
                "train/loss": mean_loss,
                "train/ppl": math.exp(min(mean_loss, 20.0)),
                "train/lr": lr,
                "train/tokens_per_sec": total_tokens / max(elapsed, 1e-6),
                "train/tokens_seen": step * tokens_per_step,
                "train/grad_norm": grad_norm,
                "step": step,
            }
            if muon_grad_norm is not None:
                metrics["train/grad_norm_muon"] = muon_grad_norm
                metrics["train/grad_norm_adam"] = adam_grad_norm
            if isinstance(scheduler, DualLRScheduler):
                metrics["train/aux_lr"] = scheduler.adam.get_last_lr()[0]
            bar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.2e}")
            if step % config["log_every_steps"] == 0 and tracker is not None:
                tracker.log(metrics, step=step)

            if step % config["eval_every_steps"] == 0:
                eval_metrics = evaluate(
                    model,
                    eval_loader,
                    loss_fn,
                    device,
                    use_flce,
                    math.ceil(config["eval_max_examples"] / get_world_size()),
                    attention_kwargs,
                )
                print0(f"[Eval {step}] loss={eval_metrics['eval_loss']:.4f}")
                if tracker is not None:
                    tracker.log({f"eval/{k.removeprefix('eval_')}": v for k, v in eval_metrics.items()}, step=step)
                if kda_gate_modules:
                    gate_metrics, means, mins, maxs = {}, [], [], []
                    for layer_idx, kda_module in kda_gate_modules.items():
                        probe = kda_gate_probe.get(layer_idx)
                        if probe is None:
                            continue
                        mean_d, min_d, max_d = _kda_gate_decay_stats(kda_module, probe)
                        gate_metrics[f"kda/layer{layer_idx}_gate_decay_mean"] = mean_d
                        gate_metrics[f"kda/layer{layer_idx}_gate_decay_min"] = min_d
                        gate_metrics[f"kda/layer{layer_idx}_gate_decay_max"] = max_d
                        means.append(mean_d)
                        mins.append(min_d)
                        maxs.append(max_d)
                    if means:
                        gate_metrics["kda/gate_decay_mean"] = sum(means) / len(means)
                        gate_metrics["kda/gate_decay_min"] = min(mins)
                        gate_metrics["kda/gate_decay_max"] = max(maxs)
                        print0(
                            f"[Eval {step}] kda gate decay mean={gate_metrics['kda/gate_decay_mean']:.4f} "
                            f"min={gate_metrics['kda/gate_decay_min']:.4f} max={gate_metrics['kda/gate_decay_max']:.4f}"
                        )
                    if tracker is not None:
                        tracker.log(gate_metrics, step=step)
                if config["eval_benchmarks"]:
                    barrier()
                    benchmarks = run_lm_benchmarks(model, assets_dir, device)
                    if tracker is not None and benchmarks:
                        tracker.log(benchmarks, step=step)
                    barrier()
            if step % config["save_every_steps"] == 0:
                save_checkpoint(model, optimizer, scheduler, step, config)

        bar.close()
        # Only save a final checkpoint if the loop didn't already save at max_steps
        # (guards against the baseline's double write when max_steps % save_every == 0).
        if max_steps % config["save_every_steps"] != 0:
            save_checkpoint(model, optimizer, scheduler, max_steps, config)
        if layer_execution is not None:
            failures = []
            expected_backwards = max_steps * config["grad_accum_steps"]
            for layer_idx, counts in layer_execution.items():
                print0(
                    f"[Execution] layer={layer_idx} type={counts['type']} "
                    f"forward={counts['forward']} backward={counts['backward']}"
                )
                if counts["forward"] < expected_backwards or counts["backward"] != expected_backwards:
                    failures.append((layer_idx, counts))
            if failures:
                raise RuntimeError(f"Attention layer execution verification failed: {failures}")
        print0("[Train] Dense LLaMA + KDA pretraining complete.")
        if config.get("sync_checkpoints_to_bucket", False) and is_main_process():
            sync_script = Path(__file__).resolve().with_name("sync_checkpoint_bucket.sh")
            subprocess.run(
                [
                    str(sync_script),
                    str(Path(config["output_dir"]).resolve()),
                    config.get("checkpoint_bucket_folder")
                    or Path(config["output_dir"]).resolve().name,
                ],
                check=True,
            )
    finally:
        if tracker is not None:
            tracker.finish()
        cleanup_runtime()


if __name__ == "__main__":
    train(CONFIG)
