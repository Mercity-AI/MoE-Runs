"""
Baseline Pretraining - LLaMA Architecture via TorchTitan HF Backend
===================================================================
First-pass port of baseline_llama.py that preserves the original
model shape and FineWeb packing semantics while replacing the
HuggingFace Trainer stack with:

  - TorchTitan's dense HuggingFace model backend
  - TorchTitan's HuggingFace tokenizer wrapper
  - TorchTitan's cross-entropy loss helper
  - A manual training loop that keeps the baseline data path intact

This is intentionally a small, explicit bridge script. It uses the
TorchTitan HF backend for the model, but does not yet adopt TorchTitan's
full config-registry / launcher stack.

Optional:
  - Liger Kernel can patch the underlying HF LLaMA modules before the
    TorchTitan model is instantiated.

Run:
    python baseline_llama_torchtitan.py

Distributed:
    torchrun --nproc_per_node=8 baseline_llama_torchtitan.py
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # --- Model (LLaMA architecture, ~1B-ish params with this shape) ---
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_2",
    # --- Tokenizer / HF assets ---
    "tokenizer_name": "meta-llama/Llama-3.2-1B",
    "hf_assets_dir": "./hf_assets_llama_baseline_titan",
    # --- Optional kernels ---
    "use_liger_kernel": False,
    "liger_kernel_config": {
        "rope": True,
        "swiglu": True,
        "rms_norm": True,
        "cross_entropy": False,
        "fused_linear_cross_entropy": False,
    },
    # --- Training ---
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 125,
    "max_steps": 2500,
    "per_device_batch_size": 12,
    "grad_accum_steps": 8,
    "max_seq_len": 2048,
    # --- Data ---
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 256,
    # --- Eval & logging ---
    "eval_every_steps": 500,
    "save_every_steps": 500,
    "log_every_steps": 1,
    # --- Misc ---
    "seed": 42,
    "output_dir": "./checkpoints_llama_baseline_torchtitan",
    "use_wandb": True,
    "wandb_project": "lec-llama-baseline-pretrain-final",
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,
    "torch_compile_mode": "default",
}

# =============================================================================
# IMPORTS
# =============================================================================

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, LlamaConfig

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    LIGER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import guard only
    apply_liger_kernel_to_llama = None
    LIGER_IMPORT_ERROR = exc

try:
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.config.job_config import JobConfig
    from torchtitan.experiments.transformers_modeling_backend.job_config import (
        HFTransformers,
    )
    from torchtitan.experiments.transformers_modeling_backend.model.args import (
        HFTransformerModelArgs,
        TitanDenseModelArgs,
    )
    from torchtitan.experiments.transformers_modeling_backend.model.model import (
        HFTransformerModel,
    )

    TORCHTITAN_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import guard only
    CrossEntropyLoss = None
    HuggingFaceTokenizer = None
    JobConfig = None
    HFTransformers = None
    HFTransformerModelArgs = None
    TitanDenseModelArgs = None
    HFTransformerModel = None
    TORCHTITAN_IMPORT_ERROR = exc


def require_torchtitan():
    if TORCHTITAN_IMPORT_ERROR is not None:
        raise ImportError(
            "TorchTitan is required for baseline_llama_torchtitan.py.\n"
            "Install it first, for example:\n"
            "  pip install torchtitan\n"
            "or follow the official source/nightly instructions in the TorchTitan README."
        ) from TORCHTITAN_IMPORT_ERROR


def maybe_enable_liger_kernel(config: dict):
    if not config.get("use_liger_kernel", False):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Liger Kernel requires CUDA for this training script.")
    if apply_liger_kernel_to_llama is None:
        raise ImportError(
            "Liger Kernel is enabled but not installed.\n"
            "Install it first, for example:\n"
            "  pip install liger-kernel"
        ) from LIGER_IMPORT_ERROR

    liger_cfg = dict(config.get("liger_kernel_config", {}))
    if liger_cfg.get("cross_entropy") or liger_cfg.get("fused_linear_cross_entropy"):
        raise ValueError(
            "This script computes loss outside the HF model with TorchTitan's "
            "CrossEntropyLoss, so Liger's cross-entropy kernels are not wired in. "
            "Keep 'cross_entropy' and 'fused_linear_cross_entropy' set to False, "
            "or rewrite the loss path to use Liger's fused loss explicitly."
        )
    apply_liger_kernel_to_llama(**liger_cfg)
    print0(f"[Liger] Enabled with config: {liger_cfg}")


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print0(msg: str):
    if is_main_process():
        print(msg, flush=True)


def barrier():
    if is_dist_available_and_initialized():
        dist.barrier()


def reduce_sum_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if is_dist_available_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def setup_distributed() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return {
        "device": device,
        "local_rank": local_rank,
        "rank": get_rank(),
        "world_size": get_world_size(),
    }


def cleanup_distributed():
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_autocast_context(device: torch.device):
    enabled = device.type == "cuda"
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if use_bf16() else torch.float16,
        enabled=enabled,
    )


def maybe_init_wandb(config: dict):
    if not config["use_wandb"] or not is_main_process():
        return None

    wandb.init(
        project=config["wandb_project"],
        name="llama_baseline_torchtitan",
        config=config,
    )
    return wandb


# =============================================================================
# ASSETS / MODEL / TOKENIZER
# =============================================================================


def build_llama_config(config: dict, tokenizer) -> LlamaConfig:
    return LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"],
        initializer_range=config["initializer_range"],
        tie_word_embeddings=config["tie_word_embeddings"],
        attention_bias=config["attention_bias"],
        hidden_act=config["hidden_act"],
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def ensure_hf_assets(config: dict) -> tuple[Path, int]:
    assets_dir = Path(config["hf_assets_dir"]).resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.model_max_length = config["max_seq_len"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(assets_dir)

    llama_cfg = build_llama_config(config, tokenizer)
    llama_cfg.save_pretrained(assets_dir)

    config_path = assets_dir / "baseline_llama_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print0(
        f"[Assets] HF assets ready at {assets_dir} | "
        f"vocab_size={len(tokenizer)} | eos={tokenizer.eos_token_id}"
    )
    return assets_dir, len(tokenizer)


def load_tokenizer_size_from_assets(hf_assets_dir: Path) -> int:
    tokenizer = AutoTokenizer.from_pretrained(hf_assets_dir)
    return len(tokenizer)


def build_torchtitan_tokenizer(hf_assets_dir: Path):
    require_torchtitan()
    tokenizer = HuggingFaceTokenizer(
        HuggingFaceTokenizer.Config(),
        tokenizer_path=str(hf_assets_dir),
    )
    print0(
        f"[Tokenizer] vocab_size={tokenizer.get_vocab_size()} | eos={tokenizer.eos_id}"
    )
    return tokenizer


def build_torchtitan_model(config: dict, hf_assets_dir: Path):
    require_torchtitan()
    maybe_enable_liger_kernel(config)

    job_config = JobConfig()
    job_config.model.hf_assets_path = str(hf_assets_dir)
    job_config.training.seq_len = config["max_seq_len"]
    job_config.training.dtype = "bfloat16" if use_bf16() else "float32"
    job_config.training.seed = config["seed"]
    job_config.hf_transformers = HFTransformers(model=str(hf_assets_dir))

    titan_args = TitanDenseModelArgs()
    model_args = HFTransformerModelArgs(titan_dense_args=titan_args).update_from_config(
        job_config
    )
    model = HFTransformerModel(model_args)

    # The baseline explicitly requested FlashAttention 2. The TorchTitan HF
    # backend constructs the HF model directly from config, so we set the
    # requested attention backend on the wrapped config before training.
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config._attn_implementation = config["attn_implementation"]
        model.model.config.attn_implementation = config["attn_implementation"]

    n_params = sum(p.numel() for p in model.parameters())
    print0(
        f"[Model] TorchTitan HF backend LLaMA: {n_params:,} params ({n_params / 1e9:.2f}B)"
    )
    return model


# =============================================================================
# DATASET - streaming + packing
# =============================================================================


class PackedFineWebDataset(IterableDataset):
    """
    Streaming FineWeb with document packing.

    Documents are concatenated end-to-end with EOS between them and sliced
    into exactly max_seq_len tokens. The trainer computes next-token loss
    externally by shifting logits/labels, matching HF causal-LM semantics.
    """

    def __init__(
        self,
        config: dict,
        tokenizer,
        seed: int = 42,
        max_examples: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_seq_len = config["max_seq_len"]
        self.eos_id = tokenizer.eos_id
        self.seed = seed
        self.epoch = 0
        self.buffer_size = config["streaming_buffer_size"]
        self.max_examples = max_examples
        self.rank = rank
        self.world_size = world_size

        self.base_dataset = load_dataset(
            config["dataset_name"],
            name=config["dataset_config"],
            split="train",
            streaming=True,
        )
        print0(
            f"[Data] {config['dataset_name']} / {config['dataset_config']} streaming"
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _split_for_process_and_worker(self, dataset):
        from datasets.distributed import split_dataset_by_node

        dataset = split_dataset_by_node(dataset, self.rank, self.world_size)

        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            dataset = split_dataset_by_node(dataset, worker.id, worker.num_workers)

        return dataset

    def __iter__(self):
        dataset = self.base_dataset.shuffle(
            seed=self.seed + self.epoch,
            buffer_size=self.buffer_size,
        )
        dataset = self._split_for_process_and_worker(dataset)
        dataset_iter = iter(dataset)

        buffer = []
        examples_yielded = 0

        while True:
            if self.max_examples is not None and examples_yielded >= self.max_examples:
                return

            while len(buffer) < self.max_seq_len:
                try:
                    doc = next(dataset_iter)
                except StopIteration:
                    return

                text = doc.get("text", "")
                if not text or not text.strip():
                    continue

                tokens = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                if tokens:
                    buffer.extend(tokens)
                    buffer.append(self.eos_id)

            chunk = buffer[: self.max_seq_len]
            buffer = buffer[self.max_seq_len :]

            yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}
            examples_yielded += 1


def build_dataloader(
    dataset: IterableDataset, batch_size: int, config: dict
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "pin_memory": torch.cuda.is_available(),
    }
    if config["dataloader_workers"] > 0:
        kwargs["num_workers"] = config["dataloader_workers"]
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config["dataloader_prefetch_factor"]
    return DataLoader(dataset, **kwargs)


# =============================================================================
# OPTIM / SCHED / CKPT
# =============================================================================


def build_optimizer(model: torch.nn.Module, config: dict) -> AdamW:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
    )


def build_scheduler(optimizer: AdamW, config: dict) -> LambdaLR:
    warmup_steps = config["warmup_steps"]
    max_steps = config["max_steps"]

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    config: dict,
):
    if not is_main_process():
        return

    ckpt_dir = Path(config["output_dir"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"

    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        path,
    )
    print0(f"[Checkpoint] Saved {path}")


# =============================================================================
# TRAIN / EVAL
# =============================================================================


def forward_loss(
    model, loss_fn, input_ids: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, int]:
    input_ids = input_ids.to(device, non_blocking=True)
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    local_tokens = shift_labels.numel()
    local_tokens_tensor = torch.tensor(local_tokens, device=device, dtype=torch.float32)
    loss = loss_fn(shift_logits, shift_labels, local_tokens_tensor)
    return loss, local_tokens


@torch.no_grad()
def evaluate(model, eval_loader: DataLoader, loss_fn, device: torch.device) -> dict:
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0.0

    for batch in eval_loader:
        loss, token_count = forward_loss(model, loss_fn, batch["input_ids"], device)
        total_loss_sum += loss.item() * token_count
        total_tokens += token_count

    total_loss_sum = reduce_sum_scalar(total_loss_sum, device)
    total_tokens = reduce_sum_scalar(total_tokens, device)

    mean_loss = total_loss_sum / max(total_tokens, 1.0)
    ppl = math.exp(min(mean_loss, 20.0))
    model.train()
    return {"eval_loss": mean_loss, "eval_ppl": ppl}


def train(config: dict):
    require_torchtitan()

    dist_info = setup_distributed()
    device = dist_info["device"]
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])
    wandb_run = maybe_init_wandb(config)

    try:
        if is_main_process():
            hf_assets_dir, tokenizer_size = ensure_hf_assets(config)
        else:
            hf_assets_dir = Path(config["hf_assets_dir"]).resolve()
            tokenizer_size = 0
        barrier()
        if not is_main_process():
            tokenizer_size = load_tokenizer_size_from_assets(hf_assets_dir)

        tokenizer = build_torchtitan_tokenizer(hf_assets_dir)
        model = build_torchtitan_model(config, hf_assets_dir)
        model.to(device)

        if tokenizer_size != getattr(unwrap_model(model), "model").config.vocab_size:
            raise ValueError(
                f"Tokenizer/model vocab mismatch: tokenizer={tokenizer_size}, "
                f"model={unwrap_model(model).model.config.vocab_size}. "
                "Regenerate assets/config before training."
            )

        if config["torch_compile"]:
            print0(f"[Compile] torch.compile enabled ({config['torch_compile_mode']})")
            model = torch.compile(model, mode=config["torch_compile_mode"])

        if get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
            )

        train_ds = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"],
            rank=get_rank(),
            world_size=get_world_size(),
        )
        eval_examples_per_rank = math.ceil(
            config["eval_max_examples"] / get_world_size()
        )
        eval_ds = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"] + 9999,
            max_examples=eval_examples_per_rank,
            rank=get_rank(),
            world_size=get_world_size(),
        )

        train_loader = build_dataloader(
            train_ds, config["per_device_batch_size"], config
        )
        eval_loader = build_dataloader(eval_ds, config["per_device_batch_size"], config)

        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        loss_fn = CrossEntropyLoss(CrossEntropyLoss.Config())

        local_tokens_per_update = (
            config["per_device_batch_size"]
            * config["grad_accum_steps"]
            * (config["max_seq_len"] - 1)
        )
        global_tokens_per_update = local_tokens_per_update * get_world_size()
        print0(f"[Train] Local tokens/step: {local_tokens_per_update:,}")
        print0(f"[Train] Global tokens/step: {global_tokens_per_update:,}")
        print0(
            f"[Train] Total: {global_tokens_per_update * config['max_steps'] / 1e9:.1f}B "
            f"tokens over {config['max_steps']:,} optimizer steps"
        )

        train_iter = iter(train_loader)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, config["max_steps"] + 1):
            step_start = time.time()

            accum_loss_sum = 0.0
            accum_tokens = 0

            for _ in range(config["grad_accum_steps"]):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                with get_autocast_context(device):
                    normalized_loss, token_count = forward_loss(
                        model,
                        loss_fn,
                        batch["input_ids"],
                        device,
                    )
                    loss = normalized_loss / config["grad_accum_steps"]

                loss.backward()
                accum_loss_sum += normalized_loss.detach().item() * token_count
                accum_tokens += token_count

            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_loss_sum = reduce_sum_scalar(accum_loss_sum, device)
            global_tokens = reduce_sum_scalar(float(accum_tokens), device)
            mean_loss = global_loss_sum / max(global_tokens, 1.0)
            ppl = math.exp(min(mean_loss, 20.0))
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - step_start
            tok_s = global_tokens / max(elapsed, 1e-6)

            if step % config["log_every_steps"] == 0:
                print0(
                    f"[Step {step:05d}] loss={mean_loss:.4f} | ppl={ppl:.2f} | "
                    f"lr={lr:.3e} | tok/s={tok_s:,.0f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": mean_loss,
                            "train/ppl": ppl,
                            "train/lr": lr,
                            "train/tokens_per_sec": tok_s,
                            "step": step,
                        },
                        step=step,
                    )

            if step % config["eval_every_steps"] == 0:
                metrics = evaluate(model, eval_loader, loss_fn, device)
                print0(
                    f"[Eval {step:05d}] loss={metrics['eval_loss']:.4f} | "
                    f"ppl={metrics['eval_ppl']:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/loss": metrics["eval_loss"],
                            "eval/ppl": metrics["eval_ppl"],
                            "step": step,
                        },
                        step=step,
                    )

            if step % config["save_every_steps"] == 0:
                save_checkpoint(model, optimizer, scheduler, step, config)

        save_checkpoint(model, optimizer, scheduler, config["max_steps"], config)
        print0("[Train] LLaMA TorchTitan baseline pretraining complete.")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


if __name__ == "__main__":
    train(CONFIG)
