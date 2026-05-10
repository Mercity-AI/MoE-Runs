"""
Benchmark FA4 vs SDPA throughput on the same model/config to report speedup
and project training time for 500M params on 500M tokens.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TRACKIO_MODE", "disabled")

sys.path.insert(0, ".")
import baseline_llama_torchtitan as m

import torch


def bench(attn_impl: str, warmup_steps: int, measured_steps: int,
          per_device_batch_size: int, grad_accum_steps: int,
          use_liger: bool):
    if attn_impl == "sdpa":
        torch.backends.cuda.enable_cudnn_sdp(False)
    cfg = dict(m.CONFIG)
    cfg["attn_implementation"] = attn_impl
    cfg["use_liger_kernel"] = use_liger
    cfg["use_trackio"] = False
    cfg["per_device_batch_size"] = per_device_batch_size
    cfg["grad_accum_steps"] = grad_accum_steps
    cfg["dataloader_workers"] = 4
    if os.environ.get("BENCH_SEQ_LEN"):
        cfg["max_seq_len"] = int(os.environ["BENCH_SEQ_LEN"])
        cfg["max_position_embeddings"] = max(cfg["max_position_embeddings"], cfg["max_seq_len"])
    cfg["warmup_steps"] = max(1, warmup_steps)
    cfg["max_steps"] = warmup_steps + measured_steps
    cfg["eval_every_steps"] = 10**9
    cfg["save_every_steps"] = 10**9
    cfg["log_every_steps"] = 1
    if os.environ.get("BENCH_NO_COMPILE"):
        cfg["torch_compile"] = False
    cfg["output_dir"] = f"./bench_ckpt_{attn_impl}"
    cfg["hf_assets_dir"] = "./hf_assets_llama_500m_titan"

    # We patch the train() function to capture per-step times.
    timings = []

    orig_train = m.train

    def instrumented_train(config):
        # mostly mirrors original train(), but records timings without checkpoints/eval
        m.require_torchtitan()
        dist_info = m.setup_distributed()
        device = dist_info["device"]
        from pathlib import Path
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        m.seed_everything(config["seed"])
        if m.is_main_process():
            hf_assets_dir, tokenizer_size = m.ensure_hf_assets(config)
        else:
            hf_assets_dir = Path(config["hf_assets_dir"]).resolve()
            tokenizer_size = 0
        tokenizer = m.build_torchtitan_tokenizer(hf_assets_dir)
        model = m.build_torchtitan_model(config, hf_assets_dir)
        model.to(device)
        if config.get("torch_compile"):
            print(f"[Bench] torch.compile mode={config['torch_compile_mode']}")
            model = torch.compile(model, mode=config["torch_compile_mode"])

        train_ds = m.PackedFineWebDataset(
            config, tokenizer, seed=config["seed"],
            rank=0, world_size=1,
        )
        train_loader = m.build_dataloader(
            train_ds, config["per_device_batch_size"], config
        )
        job_config = m.build_torchtitan_job_config(config, 1)
        optimizer = m.build_optimizer(model, job_config)
        scheduler = m.build_scheduler(optimizer, job_config)
        use_flce = bool(cfg.get("liger_kernel_config", {}).get("fused_linear_cross_entropy"))
        loss_fn = None if use_flce else m.build_torchtitan_ce_loss()

        local_tokens_per_update = (
            config["per_device_batch_size"] *
            config["grad_accum_steps"] *
            (config["max_seq_len"] - 1)
        )
        print(f"[Bench {attn_impl}] tokens/optim_step = {local_tokens_per_update:,}")

        train_iter = iter(train_loader)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, config["max_steps"] + 1):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(config["grad_accum_steps"]):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                with m.get_autocast_context(device):
                    loss, tok = m.forward_loss(model, loss_fn, batch["input_ids"], device, use_flce)
                    loss = loss / config["grad_accum_steps"]
                loss.backward()
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            tok_s = local_tokens_per_update / dt
            phase = "warmup" if step <= config["warmup_steps"] else "measure"
            print(f"[{attn_impl}][{phase}] step={step:3d} dt={dt:.3f}s tok/s={tok_s:,.0f} loss={loss.item()*config['grad_accum_steps']:.3f}")
            if phase == "measure":
                timings.append(dt)
        m.cleanup_distributed()
        return local_tokens_per_update

    m.train = instrumented_train
    try:
        tokens_per_step = m.train(cfg)
    finally:
        m.train = orig_train

    if not timings:
        return None
    avg_dt = sum(timings) / len(timings)
    tok_s = tokens_per_step / avg_dt
    return {
        "attn_impl": attn_impl,
        "use_liger": use_liger,
        "tokens_per_step": tokens_per_step,
        "avg_step_time_s": avg_dt,
        "tok_per_s": tok_s,
        "n_measured": len(timings),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", type=str, default="flash_attention_4",
                    choices=["flash_attention_4", "sdpa", "eager"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bs", type=int, default=15)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    res = bench(args.attn, args.warmup, args.steps, args.bs, args.ga, not args.no_liger)
    print("RESULT", res)
