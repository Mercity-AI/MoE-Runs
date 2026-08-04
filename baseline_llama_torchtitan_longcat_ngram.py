"""Train the 1B LLaMA + LongCat input N-gram Embedding ablation."""

import math
import subprocess
import time
from pathlib import Path

import torch
from datasets import load_dataset

from data import PackedFineWebDataset, build_dataloader
from utils import (
    build_model,
    build_torchtitan_ce_loss,
    build_torchtitan_tokenizer,
    ensure_hf_assets,
    require_torchtitan,
    DualLRScheduler,
    build_job_config,
    build_optimizer,
    build_scheduler,
    cleanup_runtime,
    evaluate,
    format_optimizer_log,
    forward_loss,
    get_autocast_context,
    load_checkpoint,
    maybe_init_tracker,
    print0,
    progress_bar,
    resolve_attention_forward_kwargs,
    run_lm_benchmarks,
    save_checkpoint,
    seed_everything,
    setup_runtime,
)


CONFIG = {
    "hidden_size": 1536,
    "num_hidden_layers": 16,
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
    "tokenizer_name": "meta-llama/Llama-2-7b",
    "hf_assets_dir": "./hf_assets_llama_1b_longcat_ngram_titan_2307",
    # LongCat NE: orders 2..4, two hash tables per order, LayerNorm amplification.
    "ngram_max_n": 4,
    "ngram_num_heads": 2,
    "ngram_table_vocab_sizes": [267003, 367007, 305011, 341013, 345017, 308962],
    "ngram_embedding_amplification": "layer_norm",
    # Per-head Q/K RMSNorm after projection and before RoPE (Qwen3/Gemma-style).
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
    "max_steps": 3_053,
    "target_train_tokens": None,
    "per_device_batch_size": 12,
    "grad_accum_steps": 20,
    "max_seq_len": 8192,
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_streaming_buffer_size": 1,
    "eval_max_examples": 512,
    "eval_holdout_fraction": 0.005,
    "cross_document_attention": True,
    "eval_every_steps": 500,
    "eval_benchmarks": False,
    "save_every_steps": 500,
    "log_every_steps": 1,
    "seed": 42,
    "resume_from_checkpoint": None,
    "init_from_checkpoint": None,
    "allow_inexact_legacy_data_resume": False,
    "output_dir": "./checkpoints_llama_1b_longcat_ngram_6b_2307",
    "sync_checkpoints_to_bucket": True,
    "checkpoint_bucket_folder": "checkpoints_llama_1b_longcat_ngram_6b_2307",
    "use_wandb": True,
    "wandb_project": "llama-1b-6b-torchtitan",
    "wandb_run_name": "llama-6b-1b-ngram-2307",
    "wandb_log_console": False,
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,
    "torch_compile_mode": "default",
}


def train(config: dict):
    require_torchtitan()
    device = setup_runtime(distributed=False)
    Path(config["output_dir"]).resolve().mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    tracker = maybe_init_tracker(config, config["wandb_run_name"])

    try:
        assets_dir, tokenizer_size = ensure_hf_assets(config, "longcat_ngram")
        tokenizer = build_torchtitan_tokenizer(assets_dir)
        resume_path = config.get("resume_from_checkpoint")
        init_path = config.get("init_from_checkpoint")
        if resume_path and init_path:
            raise ValueError("Choose resume_from_checkpoint or init_from_checkpoint, not both.")
        if resume_path or init_path:
            model, start_step, optimizer_state, scheduler_state = load_checkpoint(
                config,
                "longcat_ngram",
                checkpoint_path=resume_path or init_path,
                resume_training=bool(resume_path),
            )
        else:
            model = build_model(config, assets_dir, "longcat_ngram")
            start_step, optimizer_state, scheduler_state = 0, None, None
        model.to(device)
        if tokenizer_size != model.config.vocab_size:
            raise ValueError("Tokenizer and model vocabulary sizes differ.")
        if config["torch_compile"]:
            model = torch.compile(model, mode=config["torch_compile_mode"])

        raw_data = load_dataset(
            config["dataset_name"],
            name=config["dataset_config"],
            split="train",
            streaming=True,
        )
        if not config.get("cross_document_attention", True):
            raise NotImplementedError("Document-masked FA4 attention is not implemented.")
        print0("[Packing] EOS resets n-grams; causal attention crosses packed documents.")
        train_data = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"],
            start_batch=start_step * config["grad_accum_steps"],
            partition="train",
            base_dataset=raw_data,
        )
        eval_data = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"] + 9999,
            max_examples=config["eval_max_examples"],
            partition="eval",
            base_dataset=raw_data,
        )
        train_loader = build_dataloader(train_data, config["per_device_batch_size"], config)
        eval_loader = build_dataloader(eval_data, config["per_device_batch_size"], config)

        job = build_job_config(config)
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
            loss_sum, token_count = 0.0, 0
            for _ in range(config["grad_accum_steps"]):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                with get_autocast_context(device):
                    micro_loss, micro_tokens = forward_loss(
                        model, loss_fn, batch["input_ids"], device, use_flce, attention_kwargs
                    )
                    (micro_loss / config["grad_accum_steps"]).backward()
                loss_sum += micro_loss.detach().item() * micro_tokens
                token_count += micro_tokens

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"]).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = loss_sum / max(token_count, 1)
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - started
            metrics = {
                "train/loss": mean_loss,
                "train/ppl": math.exp(min(mean_loss, 20.0)),
                "train/lr": lr,
                "train/tokens_per_sec": token_count / max(elapsed, 1e-6),
                "train/tokens_seen": step * tokens_per_step,
                "train/grad_norm": grad_norm,
                "step": step,
            }
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
                    config["eval_max_examples"],
                    attention_kwargs,
                )
                print0(f"[Eval {step}] loss={eval_metrics['eval_loss']:.4f}")
                if tracker is not None:
                    tracker.log({f"eval/{k.removeprefix('eval_')}": v for k, v in eval_metrics.items()}, step=step)
                if config["eval_benchmarks"]:
                    benchmarks = run_lm_benchmarks(model, assets_dir, device)
                    if tracker is not None and benchmarks:
                        tracker.log(benchmarks, step=step)
            if step % config["save_every_steps"] == 0:
                save_checkpoint(model, optimizer, scheduler, step, config)

        bar.close()
        if max_steps % config["save_every_steps"] != 0:
            save_checkpoint(model, optimizer, scheduler, max_steps, config)
        print0("[Train] LLaMA + LongCat n-gram pretraining complete.")
        if config.get("sync_checkpoints_to_bucket", False):
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
