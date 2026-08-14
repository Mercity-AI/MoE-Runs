"""Short, checkpoint-free throughput benchmark for KDA training configs."""

import statistics
import time

import torch

import baseline_llama_KDA as training


TARGET_TOKENS = 6_000_000_000
BENCHMARK_STEPS = 16


def run_benchmark(name: str, overrides: dict) -> None:
    config = {
        **training.CONFIG,
        **overrides,
        "max_steps": BENCHMARK_STEPS,
        "target_train_tokens": None,
        "warmup_steps": 2,
        "eval_every_steps": BENCHMARK_STEPS + 1,
        "save_every_steps": BENCHMARK_STEPS,
        "use_wandb": False,
        "sync_checkpoints_to_bucket": False,
        "verify_layer_execution": False,
    }
    full_steps = round(
        TARGET_TOKENS
        / (
            config["per_device_batch_size"]
            * config["grad_accum_steps"]
            * (config["max_seq_len"] - 1)
        )
    )
    full_tokens = (
        full_steps
        * config["per_device_batch_size"]
        * config["grad_accum_steps"]
        * (config["max_seq_len"] - 1)
    )

    # Keep benchmark runs isolated and avoid writing a ~GB checkpoint at step 16.
    training.save_checkpoint = lambda *args, **kwargs: None

    step_times = []
    step_peaks = []

    class TimedProgress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            for step in self.iterable:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                yield step
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                step_times.append(elapsed)
                step_peaks.append(peak)
                print(f"[BenchStep] name={name} step={len(step_times)} seconds={elapsed:.3f} peak_gib={peak:.3f}")

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

    original_progress_bar = training.progress_bar
    training.progress_bar = lambda iterable, **kwargs: TimedProgress(iterable)
    try:
        training.train(config)
    finally:
        training.progress_bar = original_progress_bar

    measured = step_times[3:] if len(step_times) > 3 else step_times
    median_step = statistics.median(measured)
    mean_step = statistics.mean(measured)
    tokens_per_step = (
        config["per_device_batch_size"]
        * config["grad_accum_steps"]
        * (config["max_seq_len"] - 1)
    )
    eta_hours = median_step * full_steps / 3600
    print(
        f"[BenchResult] name={name} full_steps={full_steps} full_tokens={full_tokens} "
        f"median_step_s={median_step:.3f} mean_step_s={mean_step:.3f} "
        f"tokens_per_s={tokens_per_step / median_step:.1f} eta_hours={eta_hours:.2f} "
        f"peak_gib={max(step_peaks):.3f}"
    )
