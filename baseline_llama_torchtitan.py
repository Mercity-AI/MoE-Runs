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

Fix note (transformers >= 4.x / TorchTitan skew):
  TorchTitan's _patch_hf_llama_like registers _initialize_weights_patched
  with signature (module, is_remote_code) — 2 args. Newer transformers
  changed smart_apply to call fn(module, fn, is_remote_code) — 3 args.
  This causes a TypeError at model construction time. The fix monkey-patches
  PreTrainedModel.initialize_weights to a no-op around construction, then
  manually drives _init_weights per-module via model.apply() afterward.
  This is safe for pretraining from scratch because all weights are random
  regardless; initialize_weights just sets the initial distribution.

Run:
    python baseline_llama_torchtitan.py

Distributed:
    torchrun --nproc_per_node=8 baseline_llama_torchtitan.py
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # --- Model: ~500M LLaMA-style ---
    "hidden_size": 1536,
    "num_hidden_layers": 16,
    "num_attention_heads": 12,
    "num_key_value_heads": 6,
    "intermediate_size": 4608,
    "vocab_size": 32000,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_4",
    # --- Tokenizer / HF assets ---
    "tokenizer_name": "meta-llama/Llama-2-7b",
    "hf_assets_dir": "./hf_assets_llama_500m_titan",
    # --- Optional kernels ---
    "use_liger_kernel": True,
    "liger_kernel_config": {
        "rope": True,
        "swiglu": True,
        "rms_norm": True,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    },
    # --- Training ---
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
    "warmup_steps": 100,
    "lr_decay_type": "cosine",
    "lr_decay_ratio": None,
    "min_lr_factor": 0.1,
    "max_steps": 3_053,
    "per_device_batch_size": 20,
    "grad_accum_steps": 8,
    "max_seq_len": 8192,
    # --- Data ---
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 512,
    # --- Eval & logging ---
    "eval_every_steps": 500,
    "eval_benchmarks": False,
    "save_every_steps": 500,
    "log_every_steps": 1,
    # --- Misc ---
    "seed": 42,
    "resume_from_checkpoint": None,  # e.g. "./checkpoints_llama_500m_4b/step_001500_hf"
    "output_dir": "./checkpoints_llama_500m_4b",
    "use_wandb": True,
    "wandb_project": "llama-500m-4b-torchtitan",
    "dataloader_workers": 8,
    "dataloader_prefetch_factor": 2,
    # NOTE: torch.compile + Liger fused_linear_cross_entropy produces unstable
    # loss on this stack (transformers 5.8 / torch 2.11 / liger-kernel 0.8.0).
    # Loss bounces between sane values and 17/60+ across steps — wrong grads.
    # Compile alone (FLCE off) works; FLCE alone works. Don't combine until
    # the upstream interaction is fixed. See flce_compile_results.md.
    "torch_compile": False,
    "torch_compile_mode": "default",
}

# =============================================================================
# IMPORTS
# =============================================================================

import json
import math
import os
import re
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
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

if hasattr(torch.optim, "Muon"):
    MUON_IMPORT_ERROR = None
else:
    MUON_IMPORT_ERROR = ImportError(
        "torch.optim.Muon not available — requires PyTorch >= 2.11"
    )

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    LIGER_IMPORT_ERROR = None
except ImportError as exc:
    apply_liger_kernel_to_llama = None
    LIGER_IMPORT_ERROR = exc

try:
    from torchtitan.components import loss as torchtitan_loss
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

    CrossEntropyLoss = getattr(torchtitan_loss, "CrossEntropyLoss", None)
    cross_entropy_loss = getattr(torchtitan_loss, "cross_entropy_loss", None)
    TORCHTITAN_IMPORT_ERROR = None
except ImportError as exc:
    CrossEntropyLoss = None
    cross_entropy_loss = None
    HuggingFaceTokenizer = None
    JobConfig = None
    HFTransformers = None
    HFTransformerModelArgs = None
    TitanDenseModelArgs = None
    HFTransformerModel = None
    TORCHTITAN_IMPORT_ERROR = exc


# =============================================================================
# UTILITIES
# =============================================================================


def require_torchtitan():
    if TORCHTITAN_IMPORT_ERROR is not None:
        raise ImportError(
            "TorchTitan import failed for baseline_llama_torchtitan.py.\n"
            f"Original import error: {TORCHTITAN_IMPORT_ERROR}"
        ) from TORCHTITAN_IMPORT_ERROR
    if CrossEntropyLoss is None and cross_entropy_loss is None:
        raise ImportError(
            "TorchTitan loss API not found. Expected either "
            "'CrossEntropyLoss' or 'cross_entropy_loss' in "
            "torchtitan.components.loss."
        )


def require_muon():
    if MUON_IMPORT_ERROR is not None:
        raise ImportError(
            "Muon requires PyTorch >= 2.11 (torch.optim.Muon).\n"
            f"Original error: {MUON_IMPORT_ERROR}"
        ) from MUON_IMPORT_ERROR


def resolve_optimizer_name(config: dict) -> str:
    name = str(config.get("optimizer_name", "adamw")).strip().lower()
    if name in {"adamw", "adam"}:
        return "adamw"
    if name == "muon":
        return "muon"
    raise NotImplementedError(f"Unsupported optimizer: {config.get('optimizer_name')}")


def is_no_decay_param(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()


def is_muon_hidden_param(name: str, param: torch.nn.Parameter) -> bool:
    return name.startswith("model.layers.") and param.ndim >= 2


def format_optimizer_log(config: dict, job_config: JobConfig) -> list[str]:
    optimizer_name = resolve_optimizer_name(config)
    if optimizer_name == "muon":
        return [
            "[Optim] Muon(hidden) + AuxAdam(non-hidden)",
            (
                "[Optim] "
                f"muon_lr={config.get('muon_lr', 0.02):.3e} | "
                f"muon_momentum={config.get('muon_momentum', 0.95):.2f} | "
                f"muon_wd={config.get('muon_weight_decay', config['weight_decay']):.3e} | "
                f"aux_lr={config.get('aux_adam_lr', config['learning_rate']):.3e} | "
                f"aux_betas=({config.get('aux_adam_beta1', config['beta1']):.2f}, "
                f"{config.get('aux_adam_beta2', config['beta2']):.2f}) | "
                f"aux_eps={config.get('aux_adam_eps', config['optimizer_eps']):.1e}"
            ),
        ]
    return [
        f"[Optim] {job_config.optimizer.name} | "
        f"impl={job_config.optimizer.implementation}"
    ]


def maybe_enable_liger_kernel(config: dict):
    if not config.get("use_liger_kernel", False):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Liger Kernel requires CUDA for this training script.")
    if apply_liger_kernel_to_llama is None:
        raise ImportError(
            "Liger Kernel is enabled but not installed.\n  pip install liger-kernel"
        ) from LIGER_IMPORT_ERROR

    liger_cfg = dict(config.get("liger_kernel_config", {}))
    if liger_cfg.get("cross_entropy") and liger_cfg.get("fused_linear_cross_entropy"):
        raise ValueError(
            "Liger 'cross_entropy' and 'fused_linear_cross_entropy' are mutually "
            "exclusive. Pick one (FLCE is preferred — fuses LM head + softmax + CE)."
        )
    apply_liger_kernel_to_llama(**liger_cfg)
    print0(f"[Liger] Enabled with config: {liger_cfg}")


def resolve_attention_forward_kwargs(config: dict) -> dict:
    kwargs = dict(config.get("attention_forward_kwargs", {}))
    uses_gqa = config["num_attention_heads"] != config["num_key_value_heads"]

    if (
        config["attn_implementation"] == "flash_attention_4"
        and uses_gqa
        and "pack_gqa" not in kwargs
    ):
        # Keep FA4 on the non-packed GQA path; the packed path is the one that
        # previously crashed in flash_attn/cute/pack_gqa.py on this stack.
        kwargs["pack_gqa"] = False

    return kwargs


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


def progress_bar(*args, **kwargs):
    kwargs.setdefault("disable", not is_main_process())
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    return tqdm(*args, **kwargs)


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


def maybe_init_tracker(config: dict):
    if not config["use_wandb"] or not is_main_process():
        return None

    wandb.init(
        project=config["wandb_project"],
        name="llama_baseline_torchtitan",
        config=config,
    )
    return wandb


def run_lm_benchmarks(
    model: torch.nn.Module,
    hf_assets_dir: Path,
    device: torch.device,
    tasks: tuple = ("hellaswag", "winogrande"),
    num_fewshot: int = 0,
) -> dict:
    if not is_main_process():
        return {}
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print0("[Bench] lm_eval not installed — skipping. pip install lm-eval")
        return {}

    base_model = unwrap_model(model)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_assets_dir)

    lm = HFLM(
        pretrained=base_model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=str(device),
        batch_size=4,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        log_samples=False,
    )
    base_model.train()

    metrics = {}
    for task, task_metrics in results["results"].items():
        for metric, value in task_metrics.items():
            if isinstance(value, float):
                metrics[f"bench/{task}/{metric}"] = value
    return metrics


# =============================================================================
# SMART-APPLY COMPAT PATCH
# =============================================================================


def _patch_initialize_weights_compat():
    """
    Context manager that temporarily replaces PreTrainedModel.initialize_weights
    with a no-op to work around a signature mismatch between TorchTitan and
    newer versions of transformers.

    Root cause:
        TorchTitan._patch_hf_llama_like() registers:
            _initialize_weights_patched(module, is_remote_code)   # 2 args

        Newer transformers.modeling_utils.smart_apply calls:
            fn(module, fn, is_remote_code)                        # 3 args

        This causes: TypeError: takes 2 positional arguments but 3 were given

    Fix:
        Suppress initialize_weights entirely during model __init__, then
        manually call model.apply(model._init_weights) after construction
        to ensure all parameters get the correct initial distribution.
        This is safe for pretraining from scratch — weights are random either
        way and the checkpoint overwrites them on any resumed run.

    Usage:
        with _patch_initialize_weights_compat():
            model = SomeHFModel(config)
        model.apply(model._init_weights)
    """
    from contextlib import contextmanager

    from transformers.modeling_utils import PreTrainedModel

    @contextmanager
    def _ctx():
        original = PreTrainedModel.initialize_weights

        def _noop(self):
            pass

        PreTrainedModel.initialize_weights = _noop
        try:
            yield
        finally:
            PreTrainedModel.initialize_weights = original

    return _ctx()


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

    if hasattr(HuggingFaceTokenizer, "Config"):
        tokenizer = HuggingFaceTokenizer(
            HuggingFaceTokenizer.Config(),
            tokenizer_path=str(hf_assets_dir),
        )
    else:
        tokenizer = HuggingFaceTokenizer(str(hf_assets_dir))

    vocab_size = (
        tokenizer.get_vocab_size()
        if hasattr(tokenizer, "get_vocab_size")
        else getattr(tokenizer, "n_words", None)
    )
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_id = tokenizer.eos_token_id

    print0(f"[Tokenizer] vocab_size={vocab_size} | eos={eos_id}")
    return tokenizer


def build_torchtitan_ce_loss():
    """
    Returns a unified loss callable regardless of which TorchTitan loss API
    is available (CrossEntropyLoss class vs bare cross_entropy_loss function).

    Signature of returned fn:
        loss_fn(pred, labels, global_valid_tokens) -> scalar tensor
    """
    require_torchtitan()

    if CrossEntropyLoss is not None:
        loss_obj = CrossEntropyLoss(CrossEntropyLoss.Config())

        def loss_fn(pred, labels, global_valid_tokens=None):
            return loss_obj(pred, labels, global_valid_tokens)

        return loss_fn

    if cross_entropy_loss is not None:

        def loss_fn(pred, labels, global_valid_tokens=None):
            loss = cross_entropy_loss(pred, labels)
            if global_valid_tokens is not None:
                loss = loss / global_valid_tokens
            return loss

        return loss_fn

    raise ImportError(
        "No compatible TorchTitan cross-entropy loss implementation was found."
    )


def build_direct_hf_model(config: dict, hf_assets_dir: Path) -> LlamaForCausalLM:
    """
    Fallback model constructor that bypasses TorchTitan's HFTransformerModel
    wrapper and builds a plain LlamaForCausalLM directly from the saved config.

    Uses _patch_initialize_weights_compat() to suppress the TorchTitan /
    transformers smart_apply arity mismatch, then manually re-runs
    _init_weights via model.apply() to restore correct weight initialization.
    """
    llama_cfg = LlamaConfig.from_pretrained(hf_assets_dir)
    # Ensure RoPE covers the full runtime seq len regardless of what was saved.
    llama_cfg.max_position_embeddings = max(
        llama_cfg.max_position_embeddings, config["max_seq_len"]
    )

    with _patch_initialize_weights_compat():
        model = LlamaForCausalLM._from_config(
            llama_cfg,
            attn_implementation=config["attn_implementation"],
            dtype=torch.bfloat16,
        )

    # Manually re-run per-module weight init now that construction is done.
    # _init_weights is the standard HF per-module initializer; calling it via
    # apply() is equivalent to what initialize_weights does internally, minus
    # the broken smart_apply dispatcher.
    model.apply(model._init_weights)

    n_params = sum(p.numel() for p in model.parameters())
    print0(f"[Model] Direct HF LLaMA: {n_params:,} params ({n_params / 1e9:.2f}B)")
    return model


def build_torchtitan_model(config: dict, hf_assets_dir: Path):
    """
    Constructs the model using TorchTitan's loss/tokenizer utilities but
    bypasses HFTransformerModel entirely.

    Why not use HFTransformerModel:
      1. HFTransformerModelArgs.update_from_config ignores the custom LlamaConfig
         saved in hf_assets_dir and loads a different (much larger) model,
         giving 6B+ params instead of the intended ~500M.
      2. HFTransformerModel.forward indexes args[0] positionally and routes
         through self.model.model(...) in a way that conflicts with the standard
         HF causal-LM forward signature we rely on for logit extraction.
      3. The _patch_hf_llama_like weight-init patch has a signature mismatch
         with newer transformers (see _patch_initialize_weights_compat).

    We still use TorchTitan for its tokenizer and loss utilities; the model
    itself is a plain LlamaForCausalLM constructed from the saved config,
    which is exactly what HFTransformerModel wraps internally anyway.
    """
    require_torchtitan()
    maybe_enable_liger_kernel(config)
    return build_direct_hf_model(config, hf_assets_dir)


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
        base_dataset=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = config["max_seq_len"]
        self.eos_id = tokenizer.eos_id
        self.seed = seed
        self.epoch = 0
        self.buffer_size = config["streaming_buffer_size"]
        self.max_examples = max_examples
        self.rank = rank
        self.world_size = world_size

        if base_dataset is not None:
            self.base_dataset = base_dataset
        else:
            from datasets import load_dataset

            self.base_dataset = load_dataset(
                config["dataset_name"],
                name=config["dataset_config"],
                split="train",
            )
            print0(
                f"[Data] {config['dataset_name']} / {config['dataset_config']} downloaded"
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


def build_torchtitan_job_config(config: dict, world_size: int) -> JobConfig:
    require_torchtitan()
    optimizer_name = resolve_optimizer_name(config)

    job_config = JobConfig()
    job_config.job.dump_folder = str(Path(config["output_dir"]).resolve())
    job_config.optimizer.name = "Muon" if optimizer_name == "muon" else "AdamW"
    job_config.optimizer.lr = (
        config.get("muon_lr", 0.02)
        if optimizer_name == "muon"
        else config["learning_rate"]
    )
    job_config.optimizer.beta1 = config["beta1"]
    job_config.optimizer.beta2 = config["beta2"]
    job_config.optimizer.eps = config["optimizer_eps"]
    job_config.optimizer.weight_decay = (
        config.get("muon_weight_decay", config["weight_decay"])
        if optimizer_name == "muon"
        else config["weight_decay"]
    )
    if optimizer_name == "muon":
        job_config.optimizer.implementation = "muon"
    else:
        job_config.optimizer.implementation = config.get(
            "optimizer_implementation"
        ) or ("fused" if torch.cuda.is_available() else "for-loop")
    job_config.lr_scheduler.warmup_steps = config["warmup_steps"]
    job_config.lr_scheduler.decay_ratio = config.get("lr_decay_ratio")
    job_config.lr_scheduler.decay_type = config.get("lr_decay_type", "cosine")
    job_config.lr_scheduler.min_lr_factor = config.get("min_lr_factor", 0.0)
    job_config.training.local_batch_size = config["per_device_batch_size"]
    job_config.training.global_batch_size = (
        config["per_device_batch_size"] * config["grad_accum_steps"] * world_size
    )
    job_config.training.seq_len = config["max_seq_len"]
    job_config.training.steps = config["max_steps"]
    job_config.training.max_norm = config["grad_clip"]
    return job_config


class _NativeMuonWithAuxAdam:
    """Pairs torch.optim.Muon (hidden 2-D params) with AdamW (aux params).

    Exposes a unified param_groups list so that a single LambdaLR scheduler
    can drive the LR multiplier for both optimizers simultaneously.
    The list holds references to the same dicts used internally by each
    optimizer, so LambdaLR's in-place `group['lr']` writes propagate.
    """

    def __init__(self, muon: torch.optim.Muon, adam: AdamW):
        self._muon = muon
        self._adam = adam
        self.param_groups = muon.param_groups + adam.param_groups

    def step(self, closure=None):
        self._muon.step(closure)
        self._adam.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self._muon.state_dict(), "adam": self._adam.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self._muon.load_state_dict(state_dict["muon"])
        self._adam.load_state_dict(state_dict["adam"])


class _DualLRScheduler:
    """Drives the same LR lambda over two separate LambdaLR schedulers."""

    def __init__(self, sched_muon: LambdaLR, sched_adam: LambdaLR):
        self._muon = sched_muon
        self._adam = sched_adam

    def step(self):
        self._muon.step()
        self._adam.step()

    def get_last_lr(self) -> list:
        return self._muon.get_last_lr()

    def state_dict(self) -> dict:
        return {"muon": self._muon.state_dict(), "adam": self._adam.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self._muon.load_state_dict(state_dict["muon"])
        self._adam.load_state_dict(state_dict["adam"])


def build_optimizer(
    model: torch.nn.Module, job_config: JobConfig, config: dict
) -> torch.optim.Optimizer:
    """
    Builds either:
      - AdamW with standard decay / no-decay groups
      - Muon for hidden block matrix weights plus AdamW-style auxiliary groups
    """
    optimizer_name = resolve_optimizer_name(config)
    base_model = unwrap_model(model)

    if optimizer_name == "muon":
        require_muon()
        muon_params, aux_decay_params, aux_no_decay_params = [], [], []
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if is_muon_hidden_param(name, param):
                muon_params.append(param)
            elif is_no_decay_param(name, param):
                aux_no_decay_params.append(param)
            else:
                aux_decay_params.append(param)
        if not muon_params:
            raise RuntimeError("Muon partitioning found no hidden matrix parameters.")
        muon_opt = torch.optim.Muon(
            muon_params,
            lr=config.get("muon_lr", 0.02),
            momentum=config.get("muon_momentum", 0.95),
            nesterov=config.get("muon_nesterov", True),
            weight_decay=config.get("muon_weight_decay", config["weight_decay"]),
            ns_steps=config.get("muon_ns_steps", 5),
        )
        aux_lr = config.get("aux_adam_lr", config["learning_rate"])
        adam_opt = AdamW(
            [
                {
                    "params": aux_decay_params,
                    "weight_decay": config.get(
                        "aux_adam_weight_decay", config["weight_decay"]
                    ),
                },
                {"params": aux_no_decay_params, "weight_decay": 0.0},
            ],
            lr=aux_lr,
            betas=(
                config.get("aux_adam_beta1", config["beta1"]),
                config.get("aux_adam_beta2", config["beta2"]),
            ),
            eps=config.get("aux_adam_eps", config["optimizer_eps"]),
        )
        return _NativeMuonWithAuxAdam(muon_opt, adam_opt)

    decay_params = []
    no_decay_params = []

    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay_param(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if job_config.optimizer.name != "AdamW":
        raise NotImplementedError(f"Unsupported optimizer: {job_config.optimizer.name}")

    optim_kwargs = {
        "lr": job_config.optimizer.lr,
        "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
        "eps": job_config.optimizer.eps,
    }
    implementation = job_config.optimizer.implementation
    if implementation == "fused":
        if not torch.cuda.is_available():
            raise ValueError("'fused' optimizer requires CUDA.")
        optim_kwargs["fused"] = True
    elif implementation == "foreach":
        optim_kwargs["foreach"] = True
    elif implementation != "for-loop":
        raise NotImplementedError(
            f"Unsupported optimizer implementation: {implementation}"
        )

    return AdamW(
        [
            {"params": decay_params, "weight_decay": job_config.optimizer.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **optim_kwargs,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, job_config: JobConfig
) -> LambdaLR:
    """
    LR schedule:
      [0, warmup_steps)       -> linear warmup from 0 to peak LR
      [warmup_steps, decay_start) -> constant at peak LR
      [decay_start, max_steps]    -> cosine / linear / sqrt decay to min_lr_factor

    decay_start = warmup_steps unless lr_decay_ratio is set, in which case
    decay_start = max_steps - int(max_steps * lr_decay_ratio).
    """
    warmup_steps = job_config.lr_scheduler.warmup_steps
    max_steps = job_config.training.steps
    decay_ratio = job_config.lr_scheduler.decay_ratio
    min_lr_factor = job_config.lr_scheduler.min_lr_factor
    decay_type = job_config.lr_scheduler.decay_type
    decay_start = warmup_steps
    if decay_ratio is not None:
        decay_steps = max(1, int(max_steps * decay_ratio))
        decay_start = max(warmup_steps, max_steps - decay_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if step < decay_start:
            return 1.0

        progress = float(step - decay_start) / float(max(1, max_steps - decay_start))
        progress = min(max(progress, 0.0), 1.0)
        if decay_type == "linear":
            decay = 1.0 - progress
        elif decay_type == "sqrt":
            decay = 1.0 - math.sqrt(progress)
        elif decay_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise NotImplementedError(f"Unsupported lr decay type: {decay_type}")
        return min_lr_factor + (1.0 - min_lr_factor) * decay

    if isinstance(optimizer, _NativeMuonWithAuxAdam):
        return _DualLRScheduler(
            LambdaLR(optimizer._muon, lr_lambda=lr_lambda),
            LambdaLR(optimizer._adam, lr_lambda=lr_lambda),
        )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    config: dict,
):
    if not is_main_process():
        return

    ckpt_dir = Path(config["output_dir"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hf_dir = ckpt_dir / f"step_{step:06d}_hf"
    unwrap_model(model).save_pretrained(hf_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(Path(config["hf_assets_dir"]).resolve())
    tokenizer.save_pretrained(hf_dir)

    state_path = ckpt_dir / f"step_{step:06d}_train_state.pt"
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        state_path,
    )
    print0(f"[Checkpoint] HF model → {hf_dir}")
    print0(f"[Checkpoint] Train state → {state_path}")


def load_checkpoint(config: dict, hf_assets_dir: Path):
    """
    Load model weights and training state from a resume checkpoint.

    Returns (model, start_step, opt_state_dict, sched_state_dict).
    opt_state_dict / sched_state_dict are None if no train_state.pt is found
    (e.g. checkpoint was saved by an older version of this script).
    """
    ckpt_path = Path(config["resume_from_checkpoint"]).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    maybe_enable_liger_kernel(config)

    llama_cfg = LlamaConfig.from_pretrained(ckpt_path)
    llama_cfg.max_position_embeddings = max(
        llama_cfg.max_position_embeddings, config["max_seq_len"]
    )

    with _patch_initialize_weights_compat():
        model = LlamaForCausalLM.from_pretrained(
            ckpt_path,
            config=llama_cfg,
            attn_implementation=config["attn_implementation"],
            torch_dtype=torch.bfloat16,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print0(f"[Resume] Loaded {n_params:,} params from {ckpt_path}")

    match = re.search(r"step_(\d+)", ckpt_path.name)
    start_step = int(match.group(1)) if match else 0
    print0(f"[Resume] Resuming from step {start_step}")

    state_path = ckpt_path.parent / f"step_{start_step:06d}_train_state.pt"
    opt_state = sched_state = None
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        opt_state = saved.get("optimizer")
        sched_state = saved.get("scheduler")
        print0(f"[Resume] Optimizer/scheduler state loaded from {state_path}")
    else:
        print0(f"[Resume] No train state at {state_path} — optimizer starts fresh")

    return model, start_step, opt_state, sched_state


# =============================================================================
# TRAIN / EVAL
# =============================================================================


def forward_loss(
    model,
    loss_fn,
    input_ids: torch.Tensor,
    device: torch.device,
    use_flce: bool,
    attention_forward_kwargs: Optional[dict] = None,
) -> tuple[torch.Tensor, int]:
    """
    Single forward pass + loss computation.

    Two paths:
      - use_flce=True  : pass labels=input_ids into the HF model. Liger's
        fused_linear_cross_entropy patches LlamaForCausalLM so the LM head +
        softmax + CE are fused — the (B, S, V) logits tensor never materializes.
        Loss is mean-reduced internally; HF shifts labels by 1 inside forward.
      - use_flce=False : original path — get logits, shift, hand to TorchTitan's
        cross_entropy_loss for the externally-computed CE.

    Returns (loss_scalar, num_tokens). num_tokens counts the (S-1) prediction
    positions per row so the caller can convert mean→sum across grad accum / DDP.
    """
    input_ids = input_ids.to(device, non_blocking=True)
    model_kwargs = dict(attention_forward_kwargs or {})

    if use_flce:
        outputs = model(input_ids, labels=input_ids, **model_kwargs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        local_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        return loss, local_tokens

    outputs = model(input_ids, **model_kwargs)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # NOTE: global_valid_tokens should be the global (all-reduce'd) token count
    # for correct loss scaling in DDP. Using local count is fine for single-GPU.
    local_tokens = shift_labels.numel()
    local_tokens_tensor = torch.tensor(local_tokens, device=device, dtype=torch.float32)
    loss = loss_fn(shift_logits, shift_labels, local_tokens_tensor)
    return loss, local_tokens


@torch.no_grad()
def evaluate(
    model,
    eval_loader: DataLoader,
    loss_fn,
    device: torch.device,
    use_flce: bool,
    attention_forward_kwargs: Optional[dict] = None,
) -> dict:
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0.0

    eval_pbar = progress_bar(eval_loader, desc="eval", position=1)
    try:
        for batch in eval_pbar:
            with get_autocast_context(device):
                loss, token_count = forward_loss(
                    model,
                    loss_fn,
                    batch["input_ids"],
                    device,
                    use_flce,
                    attention_forward_kwargs,
                )
            total_loss_sum += loss.item() * token_count
            total_tokens += token_count
    finally:
        eval_pbar.close()

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
    tracker_run = maybe_init_tracker(config)

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
        resume_ckpt = config.get("resume_from_checkpoint")
        if resume_ckpt:
            model, start_step, opt_state, sched_state = load_checkpoint(
                config, hf_assets_dir
            )
        else:
            model = build_torchtitan_model(config, hf_assets_dir)
            start_step = 0
            opt_state = sched_state = None
        model.to(device)

        # Resolve model config regardless of whether we ended up with the
        # TorchTitan wrapper or the plain HF model.
        unwrapped = unwrap_model(model)
        model_config = getattr(unwrapped, "config", None)
        if model_config is None and hasattr(unwrapped, "model"):
            model_config = getattr(unwrapped.model, "config", None)
        if model_config is None:
            raise AttributeError(
                "Could not locate model config on the constructed model."
            )

        if tokenizer_size != model_config.vocab_size:
            raise ValueError(
                f"Tokenizer/model vocab mismatch: tokenizer={tokenizer_size}, "
                f"model={model_config.vocab_size}. "
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

        from datasets import load_dataset

        print0(
            f"[Data] Loading {config['dataset_name']} / {config['dataset_config']}..."
        )
        raw_dataset = load_dataset(
            config["dataset_name"],
            name=config["dataset_config"],
            split="train",
        )
        print0(
            f"[Data] {config['dataset_name']} / {config['dataset_config']} downloaded"
        )

        train_ds = PackedFineWebDataset(
            config,
            tokenizer,
            seed=config["seed"],
            rank=get_rank(),
            world_size=get_world_size(),
            base_dataset=raw_dataset,
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
            base_dataset=raw_dataset,
        )

        train_loader = build_dataloader(
            train_ds, config["per_device_batch_size"], config
        )
        eval_loader = build_dataloader(eval_ds, config["per_device_batch_size"], config)

        job_config = build_torchtitan_job_config(config, get_world_size())
        optimizer = build_optimizer(model, job_config, config)
        scheduler = build_scheduler(optimizer, job_config)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            print0("[Resume] Optimizer state restored.")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
            print0("[Resume] Scheduler state restored.")
        attention_forward_kwargs = resolve_attention_forward_kwargs(config)
        if attention_forward_kwargs:
            print0(f"[Attention] Forward kwargs: {attention_forward_kwargs}")
        use_flce = bool(
            config.get("liger_kernel_config", {}).get("fused_linear_cross_entropy")
        )
        loss_fn = None if use_flce else build_torchtitan_ce_loss()
        if use_flce:
            print0(
                "[Loss] Liger fused_linear_cross_entropy ON (logits not materialized)"
            )
        else:
            print0(
                "[Loss] TorchTitan cross_entropy_loss (external, materializes logits)"
            )

        local_tokens_per_update = (
            config["per_device_batch_size"]
            * config["grad_accum_steps"]
            * (config["max_seq_len"] - 1)
        )
        global_tokens_per_update = local_tokens_per_update * get_world_size()
        print0(f"[Train] Local tokens/step:  {local_tokens_per_update:,}")
        print0(f"[Train] Global tokens/step: {global_tokens_per_update:,}")
        for optim_line in format_optimizer_log(config, job_config):
            print0(optim_line)
        print0(
            f"[LR] warmup={job_config.lr_scheduler.warmup_steps} | "
            f"decay={job_config.lr_scheduler.decay_type} | "
            f"decay_ratio={job_config.lr_scheduler.decay_ratio} | "
            f"min_lr_factor={job_config.lr_scheduler.min_lr_factor}"
        )
        print0(
            f"[Train] Total: "
            f"{global_tokens_per_update * config['max_steps'] / 1e9:.1f}B tokens "
            f"over {config['max_steps']:,} optimizer steps"
        )
        if start_step > 0:
            print0(
                f"[Resume] Starting at step {start_step} — dataset restarts from "
                f"beginning (~{start_step * global_tokens_per_update / 1e9:.2f}B "
                f"tokens of potential overlap with original run)"
            )

        train_iter = iter(train_loader)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_pbar = progress_bar(
            range(start_step + 1, config["max_steps"] + 1),
            desc="train",
            total=config["max_steps"],
            initial=start_step,
            position=0,
        )
        for step in train_pbar:
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
                        use_flce,
                        attention_forward_kwargs,
                    )
                    loss = normalized_loss / config["grad_accum_steps"]

                loss.backward()
                accum_loss_sum += normalized_loss.detach().item() * token_count
                accum_tokens += token_count

            grad_norm = None
            if config["grad_clip"] is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                ).item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_loss_sum = reduce_sum_scalar(accum_loss_sum, device)
            global_tokens = reduce_sum_scalar(float(accum_tokens), device)
            mean_loss = global_loss_sum / max(global_tokens, 1.0)
            ppl = math.exp(min(mean_loss, 20.0))
            lr = scheduler.get_last_lr()[0]
            aux_lr = (
                scheduler._adam.get_last_lr()[0]
                if isinstance(scheduler, _DualLRScheduler)
                else None
            )
            elapsed = time.time() - step_start
            tok_s = global_tokens / max(elapsed, 1e-6)
            tokens_seen = step * global_tokens_per_update
            train_pbar.set_postfix(
                loss=f"{mean_loss:.4f}",
                ppl=f"{ppl:.2f}",
                lr=f"{lr:.2e}",
                toks=f"{tok_s:,.0f}",
            )

            if step % config["log_every_steps"] == 0:
                print0(
                    f"[Step {step:05d}] loss={mean_loss:.4f} | ppl={ppl:.2f} | "
                    f"lr={lr:.3e} | tok/s={tok_s:,.0f}"
                    + (f" | gnorm={grad_norm:.3f}" if grad_norm is not None else "")
                )
                if tracker_run is not None:
                    log_dict = {
                        "train/loss": mean_loss,
                        "train/ppl": ppl,
                        "train/lr": lr,
                        "train/tokens_per_sec": tok_s,
                        "train/tokens_seen": tokens_seen,
                        "step": step,
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm
                    if aux_lr is not None:
                        log_dict["train/aux_lr"] = aux_lr
                    if device.type == "cuda":
                        log_dict["sys/gpu_mem_alloc_gb"] = (
                            torch.cuda.memory_allocated(device) / 1e9
                        )
                        log_dict["sys/gpu_mem_reserved_gb"] = (
                            torch.cuda.memory_reserved(device) / 1e9
                        )
                    tracker_run.log(log_dict, step=step)

            if step % config["eval_every_steps"] == 0:
                metrics = evaluate(
                    model,
                    eval_loader,
                    loss_fn,
                    device,
                    use_flce,
                    attention_forward_kwargs,
                )
                print0(
                    f"[Eval  {step:05d}] loss={metrics['eval_loss']:.4f} | "
                    f"ppl={metrics['eval_ppl']:.2f}"
                )
                if tracker_run is not None:
                    tracker_run.log(
                        {
                            "eval/loss": metrics["eval_loss"],
                            "eval/ppl": metrics["eval_ppl"],
                            "step": step,
                        },
                        step=step,
                    )

                if config.get("eval_benchmarks"):
                    bench = run_lm_benchmarks(model, hf_assets_dir, device)
                    if bench:
                        for task in ("hellaswag", "winogrande"):
                            acc = bench.get(f"bench/{task}/acc_norm,none") or bench.get(
                                f"bench/{task}/acc,none"
                            )
                            if acc is not None:
                                print0(f"[Bench {step:05d}] {task}: {acc:.4f}")
                        if tracker_run is not None:
                            tracker_run.log({**bench, "step": step}, step=step)

            if step % config["save_every_steps"] == 0:
                save_checkpoint(model, optimizer, scheduler, step, config)

        train_pbar.close()
        save_checkpoint(model, optimizer, scheduler, config["max_steps"], config)
        print0("[Train] LLaMA TorchTitan baseline pretraining complete.")

    finally:
        if tracker_run is not None:
            tracker_run.finish()
        cleanup_distributed()


if __name__ == "__main__":
    train(CONFIG)
