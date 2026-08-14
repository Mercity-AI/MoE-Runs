"""Train the dense 1B LLaMA baseline on packed FineWeb with TorchTitan helpers."""

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel

from data import PackedFineWebDataset, build_dataloader
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
    "attn_implementation": "flash_attention_4",
    # Per-head Q/K RMSNorm after projection and before RoPE (Qwen3/Gemma-style),
    # applied to every dense softmax layer. Matches the KDA/LongCat scripts.
    "qk_norm": True,
    "qk_norm_eps": 1e-6,
    "tokenizer_name": "meta-llama/Llama-2-7b",
    "hf_assets_dir": "./hf_assets_llama_1b_titan",
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
    # Fixed-step mode (target_train_tokens=None), matching the KDA/ngram scripts so
    # all three train the same 3053 optimizer steps. NOTE: setting target_train_tokens
    # would OVERRIDE max_steps (resolve_training_steps) and, on multi-GPU, shrink the
    # step count by world_size instead of scaling tokens -- keep it None.
    # 10 * 24 * 8191 * 3053 = 6.002B tokens at world_size=1.
    "max_steps": 3_053,
    "target_train_tokens": None,
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
    "output_dir": "./checkpoints_llama_baseline_1b_6b_rerun",
    "use_wandb": True,
    "wandb_project": "llama-1b-6b-torchtitan",
    "wandb_run_name": "llama-1b-baseline-rerun-qknorm-6b",
    "wandb_log_console": False,
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,
    "torch_compile_mode": "default",
}


def train(config: dict):
    require_torchtitan()
    device = setup_runtime(distributed=True)
    Path(config["output_dir"]).resolve().mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    tracker = maybe_init_tracker(config, config.get("wandb_run_name") or "llama_baseline_torchtitan")

    try:
        if is_main_process():
            assets_dir, tokenizer_size = ensure_hf_assets(config, "llama")
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
                "llama",
                checkpoint_path=resume_path or init_path,
                resume_training=bool(resume_path),
            )
        else:
            model = build_model(config, assets_dir, "llama")
            start_step, optimizer_state, scheduler_state = 0, None, None
        model.to(device)
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
            raise NotImplementedError("Document-masked FA4 attention is not implemented.")
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
                # Accumulate (loss * tokens) on-device. Calling .item() here would
                # force a CUDA sync every micro-step (grad_accum_steps-1 extra syncs
                # per optimizer step), stalling CPU run-ahead and the kernel-launch
                # pipeline. We sync once, after the loop, instead.
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
        # (guards against a double write when max_steps % save_every == 0).
        if max_steps % config["save_every_steps"] != 0:
            save_checkpoint(model, optimizer, scheduler, max_steps, config)
        print0("[Train] Dense LLaMA pretraining complete.")
    finally:
        if tracker is not None:
            tracker.finish()
        cleanup_runtime()


if __name__ == "__main__":
    train(CONFIG)
