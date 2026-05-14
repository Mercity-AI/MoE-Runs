"""
LEC Pretraining - LLaMA + PLE + Router via TorchTitan Components
================================================================
First-pass port of initial_lec_llama.py that preserves:

  - the HF LLaMA backbone
  - the current hook-based LEC injection
  - FineWeb streaming + exact packing semantics

while replacing HuggingFace Trainer with:

  - TorchTitan's tokenizer wrapper
  - TorchTitan's cross-entropy loss helper
  - an explicit distributed/manual training loop

This is intentionally a bridge script. It keeps direct control over the
custom LEC model instead of forcing it through TorchTitan's dense HF backend.

Optional:
  - Liger Kernel can patch the underlying HF LLaMA modules before the model
    is instantiated.

Run:
    python initial_lec_llama_torchtitan.py

Distributed:
    torchrun --nproc_per_node=8 initial_lec_llama_torchtitan.py
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # --- Model ---
    "hidden_size": 512,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "hidden_act": "silu",
    "attn_implementation": "flash_attention_2",
    "tokenizer_name": "meta-llama/Llama-2-7b",
    "hf_assets_dir": "./hf_assets_llama_lec_titan",
    # --- LEC ---
    "num_experts": 4,
    "ple_dim": 512,
    "lb_ema_alpha": 0.1,
    "lb_bias_lr": 1e-3,
    "entropy_collapse_threshold": 0.5,
    "lec_metric_log_every_steps": 10,
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
    "optimizer_name": "muon",
    "learning_rate": 3e-4,
    "min_lr_ratio": 0.1,
    "min_lr_factor": 0.1,
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
    "warmup_steps": 25,
    "lr_decay_type": "cosine",
    "lr_decay_ratio": None,
    "max_steps": 500,
    "per_device_batch_size": 6,
    "grad_accum_steps": 4,
    "max_seq_len": 256,
    # --- Data ---
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",
    "streaming_buffer_size": 10_000,
    "eval_max_examples": 256,
    # --- Eval & logging ---
    "eval_every_steps": 100,
    "save_every_steps": 250,
    "log_every_steps": 1,
    # --- Misc ---
    "seed": 42,
    "resume_from_checkpoint": None,
    "output_dir": "./checkpoints_llama_lec_torchtitan",
    "use_wandb": False,
    "wandb_project": "llama-lec-pretrain",
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
import torch.nn as nn
import torch.nn.functional as F
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
        "torch.optim.Muon not available - requires PyTorch >= 2.11"
    )

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    LIGER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import guard only
    apply_liger_kernel_to_llama = None
    LIGER_IMPORT_ERROR = exc

try:
    from torchtitan.components import loss as torchtitan_loss
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.config.job_config import JobConfig

    CrossEntropyLoss = getattr(torchtitan_loss, "CrossEntropyLoss", None)
    cross_entropy_loss = getattr(torchtitan_loss, "cross_entropy_loss", None)
    TORCHTITAN_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import guard only
    CrossEntropyLoss = None
    cross_entropy_loss = None
    HuggingFaceTokenizer = None
    JobConfig = None
    TORCHTITAN_IMPORT_ERROR = exc


def require_torchtitan():
    if TORCHTITAN_IMPORT_ERROR is not None:
        raise ImportError(
            "TorchTitan import failed for initial_lec_llama_torchtitan.py.\n"
            "This usually means TorchTitan is missing or incompatible with your "
            "installed PyTorch build.\n"
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
            "Liger Kernel is enabled but not installed.\n"
            "Install it first, for example:\n"
            "  pip install liger-kernel"
        ) from LIGER_IMPORT_ERROR

    liger_cfg = dict(config.get("liger_kernel_config", {}))
    if liger_cfg.get("cross_entropy") and liger_cfg.get("fused_linear_cross_entropy"):
        raise ValueError(
            "Liger 'cross_entropy' and 'fused_linear_cross_entropy' are mutually "
            "exclusive."
        )
    if liger_cfg.get("cross_entropy"):
        raise ValueError(
            "Liger 'cross_entropy' is not wired into this LEC script. "
            "Use 'fused_linear_cross_entropy' instead."
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


def reduce_mean_metrics(
    metric_sums: dict, metric_count: int, device: torch.device
) -> dict:
    if metric_count <= 0 or not metric_sums:
        return {}

    total_count = reduce_sum_scalar(float(metric_count), device)
    if total_count <= 0:
        return {}

    reduced = {}
    for key, value in metric_sums.items():
        reduced[key] = reduce_sum_scalar(float(value), device) / total_count
    return reduced


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
        name="llama_lec_torchtitan",
        config=config,
    )
    return wandb


# =============================================================================
# PLE TABLE
# =============================================================================


class PLETable(nn.Module):
    """
    Per-layer expert embedding table: [K, L, d_ple].

    Each expert k has one d_ple-dimensional vector per layer l. A shared
    projection maps d_ple -> hidden_size.
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        ple_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.hidden_size = hidden_size

        self.embeddings = nn.Parameter(torch.zeros(num_experts, num_layers, ple_dim))
        self.proj = nn.Linear(ple_dim, hidden_size, bias=False)
        self._init_orthogonal()

    def _init_orthogonal(self):
        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                if self.num_experts <= self.ple_dim:
                    q, _ = torch.linalg.qr(
                        torch.randn(
                            self.ple_dim,
                            self.ple_dim,
                            device=self.embeddings.device,
                            dtype=self.embeddings.dtype,
                        )
                    )
                    self.embeddings[:, layer_idx, :] = q[:, : self.num_experts].T * 0.02
                else:
                    nn.init.normal_(self.embeddings[:, layer_idx, :], std=0.02)

    def get_all_projected(self, layer_idx: int) -> torch.Tensor:
        return self.proj(self.embeddings[:, layer_idx, :])

    @torch.no_grad()
    def pairwise_cosine(self) -> float:
        total = 0.0
        for layer_idx in range(self.num_layers):
            vecs = self.proj(self.embeddings[:, layer_idx, :]).detach().float()
            norms = F.normalize(vecs, dim=-1)
            sim = norms @ norms.T
            mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=sim.device)
            total += sim[mask].mean().item()
        return total / self.num_layers


# =============================================================================
# ROUTER
# =============================================================================


class LECRouter(nn.Module):
    """One decoder layer's linear router with EMA load-balancing bias."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        lb_ema_alpha: float = 0.1,
        lb_bias_lr: float = 1e-3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.lb_ema_alpha = lb_ema_alpha
        self.lb_bias_lr = lb_bias_lr

        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)

        self.register_buffer("lb_bias", torch.zeros(num_experts))
        self.register_buffer("expert_ema_load", torch.ones(num_experts) / num_experts)
        self._load_accum: Optional[torch.Tensor] = None
        self._accum_count = 0

    def forward(self, hidden_states: torch.Tensor):
        logits = self.linear(hidden_states)
        biased_logits = logits + self.lb_bias.to(logits.dtype)
        probs = F.softmax(biased_logits.float(), dim=-1).to(hidden_states.dtype)

        expert_ids = probs.argmax(dim=-1)
        hard = F.one_hot(expert_ids, num_classes=self.num_experts).to(
            hidden_states.dtype
        )
        gate = hard + probs - probs.detach()
        return expert_ids, gate, logits, biased_logits, probs

    def accumulate_load(self, expert_ids: torch.Tensor):
        with torch.no_grad():
            flat = expert_ids.detach().reshape(-1)
            counts = torch.bincount(flat, minlength=self.num_experts).float()
            load = counts / counts.sum().clamp_min(1.0)
            load = load.to(self.lb_bias.device)

            if self._load_accum is None:
                self._load_accum = load
            else:
                self._load_accum += load
            self._accum_count += 1

    @torch.no_grad()
    def flush_load_balance(self):
        if self._load_accum is None or self._accum_count == 0:
            return
        avg_load = (self._load_accum / self._accum_count).to(self.lb_bias.device)
        self.expert_ema_load = (
            1 - self.lb_ema_alpha
        ) * self.expert_ema_load + self.lb_ema_alpha * avg_load
        self.lb_bias -= self.lb_bias_lr * (
            self.expert_ema_load - 1.0 / self.num_experts
        )
        self._load_accum = None
        self._accum_count = 0

    @torch.no_grad()
    def soft_entropy_norm(self, biased_logits: torch.Tensor) -> float:
        probs = F.softmax(biased_logits.detach().float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        return entropy / math.log(self.num_experts)

    @staticmethod
    @torch.no_grad()
    def hard_load_entropy_norm(expert_ids: torch.Tensor, num_experts: int) -> float:
        flat = expert_ids.detach().reshape(-1)
        counts = torch.bincount(flat, minlength=num_experts).float()
        load = counts / counts.sum().clamp_min(1.0)
        entropy = -(load * torch.log(load + 1e-10)).sum().item()
        return entropy / math.log(num_experts)

    @staticmethod
    @torch.no_grad()
    def batch_load(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        flat = expert_ids.detach().reshape(-1)
        counts = torch.bincount(flat, minlength=num_experts).float()
        return counts / counts.sum().clamp_min(1.0)


# =============================================================================
# HOOKED LLAMA + LEC
# =============================================================================


class LlamaLEC(LlamaForCausalLM):
    """
    LLaMA + LEC using forward pre-hooks on the official HF decoder layers.

    This preserves HF LLaMA internals while injecting the learned PLE shift
    immediately before each decoder layer forward.
    """

    def __init__(self, config: LlamaConfig, lec_config: dict):
        super().__init__(config)
        self.lec_config = lec_config
        self.num_experts = lec_config["num_experts"]
        self.num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        self.routers = nn.ModuleList(
            [
                LECRouter(
                    hidden_size=hidden_size,
                    num_experts=self.num_experts,
                    lb_ema_alpha=lec_config["lb_ema_alpha"],
                    lb_bias_lr=lec_config["lb_bias_lr"],
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ple = PLETable(
            self.num_experts,
            self.num_layers,
            lec_config["ple_dim"],
            hidden_size,
        )

        self.log_lec_metrics = False
        self._metric_layer_records = []
        self._hook_handles = []
        self._install_lec_hooks()

        base_params = sum(
            p.numel()
            for n, p in self.named_parameters()
            if "routers" not in n and "ple" not in n
        )
        lec_params = sum(p.numel() for p in self.routers.parameters()) + sum(
            p.numel() for p in self.ple.parameters()
        )
        print0(
            f"[LEC] Base: {base_params:,} | LEC overhead: {lec_params:,} "
            f"({100 * lec_params / (base_params + lec_params):.3f}%)"
        )
        print0(
            f"[LEC] Using {self.num_layers} per-layer routers via decoder-layer pre-hooks."
        )

    def _install_lec_hooks(self):
        for layer_idx, layer in enumerate(self.model.layers):
            handle = layer.register_forward_pre_hook(
                self._make_layer_pre_hook(layer_idx),
                with_kwargs=True,
            )
            self._hook_handles.append(handle)

    def _make_layer_pre_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if len(args) > 0:
                hidden_states = args[0]
                rest_args = args[1:]
            else:
                hidden_states = kwargs["hidden_states"]
                rest_args = ()

            router = self.routers[layer_idx]
            expert_ids, gate, _logits, biased_logits, _probs = router(hidden_states)

            ple_all = self.ple.get_all_projected(layer_idx).to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            ple_shift = torch.matmul(gate, ple_all)
            hidden_states = hidden_states + ple_shift

            if self.training:
                router.accumulate_load(expert_ids)

            if self.log_lec_metrics:
                with torch.no_grad():
                    load = LECRouter.batch_load(expert_ids.detach(), self.num_experts)
                    self._metric_layer_records.append(
                        {
                            "soft_entropy": router.soft_entropy_norm(
                                biased_logits.detach()
                            ),
                            "hard_entropy": LECRouter.hard_load_entropy_norm(
                                expert_ids.detach(),
                                self.num_experts,
                            ),
                            "load": load.detach(),
                        }
                    )

            if len(args) > 0:
                return (hidden_states, *rest_args), kwargs

            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden_states
            return args, new_kwargs

        return hook

    def _reset_lec_metric_records(self):
        self._metric_layer_records = []

    @torch.no_grad()
    def _finalize_lec_metrics(self) -> dict:
        if not self._metric_layer_records:
            return {}

        soft = [record["soft_entropy"] for record in self._metric_layer_records]
        hard = [record["hard_entropy"] for record in self._metric_layer_records]
        loads = torch.stack(
            [record["load"] for record in self._metric_layer_records], dim=0
        )
        avg_load = loads.mean(dim=0)

        metrics = {
            "router_soft_entropy": float(sum(soft) / len(soft)),
            "router_hard_entropy": float(sum(hard) / len(hard)),
            "router_entropy": float(sum(hard) / len(hard)),
            "router_hard_entropy_min": float(min(hard)),
            "ple_pairwise_cosine": self.ple.pairwise_cosine(),
            "expert_load_balance_std": avg_load.std(unbiased=False).item(),
            "expert_load_balance_min": avg_load.min().item(),
            "expert_load_balance_max": avg_load.max().item(),
            "expert_load_max_any_layer": loads.max().item(),
            "expert_utilization": (avg_load > 0).float().mean().item(),
        }

        avg_ema_load = torch.stack(
            [router.expert_ema_load.detach() for router in self.routers],
            dim=0,
        ).mean(dim=0)
        for idx, load in enumerate(avg_ema_load.tolist()):
            metrics[f"expert_ema_load_{idx}"] = float(load)
            metrics[f"expert_load_{idx}"] = float(load)

        collapse_threshold = self.lec_config.get("entropy_collapse_threshold", 0.5)
        metrics["router_collapsed_layers"] = float(
            sum(1 for value in hard if value < collapse_threshold)
        )
        return metrics

    def forward(self, *args, **kwargs):
        self._reset_lec_metric_records()
        outputs = super().forward(*args, **kwargs)
        outputs.lec_metrics = (
            self._finalize_lec_metrics() if self.log_lec_metrics else {}
        )
        return outputs


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

    config_path = assets_dir / "initial_lec_llama_config.json"
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

    print0(
        f"[Tokenizer] vocab_size={vocab_size} | eos={eos_id}"
    )
    return tokenizer


def build_torchtitan_ce_loss():
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


def build_model(config: dict, hf_assets_dir: Path) -> LlamaLEC:
    maybe_enable_liger_kernel(config)
    llama_cfg = LlamaConfig.from_pretrained(hf_assets_dir)
    llama_cfg.max_position_embeddings = max(
        llama_cfg.max_position_embeddings, config["max_seq_len"]
    )
    lec_config = {
        key: config[key]
        for key in [
            "num_experts",
            "ple_dim",
            "lb_ema_alpha",
            "lb_bias_lr",
            "entropy_collapse_threshold",
        ]
    }

    model_kwargs = {
        "lec_config": lec_config,
        "attn_implementation": config["attn_implementation"],
    }
    if use_bf16():
        model_kwargs["dtype"] = torch.bfloat16

    model = LlamaLEC._from_config(
        llama_cfg,
        **model_kwargs,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print0(f"[Model] LLaMA + LEC hooks: {n_params:,} params ({n_params / 1e9:.2f}B)")
    print0(
        f"[Attention] impl={config['attn_implementation']} | "
        f"class={model.model.layers[0].self_attn.__class__.__name__}"
    )
    return model


# =============================================================================
# DATASET
# =============================================================================


class PackedFineWebDataset(IterableDataset):
    """Streaming FineWeb with document packing to exact max_seq_len chunks."""

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
    job_config.lr_scheduler.min_lr_factor = config.get(
        "min_lr_factor",
        config.get("min_lr_ratio", 0.0),
    )
    job_config.training.local_batch_size = config["per_device_batch_size"]
    job_config.training.global_batch_size = (
        config["per_device_batch_size"] * config["grad_accum_steps"] * world_size
    )
    job_config.training.seq_len = config["max_seq_len"]
    job_config.training.steps = config["max_steps"]
    job_config.training.max_norm = config["grad_clip"]
    return job_config


class _NativeMuonWithAuxAdam:
    def __init__(self, muon, adam: AdamW):
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
    optimizer_name = resolve_optimizer_name(config)
    base_model = unwrap_model(model)

    if optimizer_name == "muon":
        require_muon()
        muon_params = []
        aux_decay_params = []
        aux_no_decay_params = []
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
            lr=config.get("aux_adam_lr", config["learning_rate"]),
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
        raise NotImplementedError(
            f"Unsupported TorchTitan optimizer: {job_config.optimizer.name}"
        )

    optim_kwargs = {
        "lr": job_config.optimizer.lr,
        "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
        "eps": job_config.optimizer.eps,
    }
    implementation = job_config.optimizer.implementation
    if implementation == "fused":
        if not torch.cuda.is_available():
            raise ValueError(
                "TorchTitan optimizer implementation 'fused' requires CUDA. "
                "Use 'for-loop' or 'foreach' on CPU."
            )
        optim_kwargs["fused"] = True
    elif implementation == "foreach":
        optim_kwargs["foreach"] = True
    elif implementation != "for-loop":
        raise NotImplementedError(
            f"Unsupported TorchTitan optimizer implementation: {implementation}"
        )

    return AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": job_config.optimizer.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **optim_kwargs,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, job_config: JobConfig
) -> LambdaLR:
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
            raise NotImplementedError(
                f"Unsupported TorchTitan lr decay type: {decay_type}"
            )
        return min_lr_factor + (1.0 - min_lr_factor) * decay

    if isinstance(optimizer, _NativeMuonWithAuxAdam):
        return _DualLRScheduler(
            LambdaLR(optimizer._muon, lr_lambda=lr_lambda),
            LambdaLR(optimizer._adam, lr_lambda=lr_lambda),
        )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def flush_router_load_balance(model: torch.nn.Module):
    unwrapped = unwrap_model(model)
    if hasattr(unwrapped, "routers"):
        for router in unwrapped.routers:
            router.flush_load_balance()


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


def load_checkpoint(config: dict, hf_assets_dir: Path):
    ckpt_path = Path(config["resume_from_checkpoint"]).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(config, hf_assets_dir)
    model.load_state_dict(saved["model"])
    start_step = int(saved.get("step", 0))
    print0(f"[Resume] Loaded checkpoint {ckpt_path}")
    print0(f"[Resume] Resuming from step {start_step}")
    return model, start_step, saved.get("optimizer"), saved.get("scheduler")


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
    collect_lec_metrics: bool = False,
) -> tuple[torch.Tensor, int, dict]:
    input_ids = input_ids.to(device, non_blocking=True)
    unwrapped = unwrap_model(model)
    model_kwargs = dict(attention_forward_kwargs or {})

    if hasattr(unwrapped, "log_lec_metrics"):
        unwrapped.log_lec_metrics = collect_lec_metrics

    try:
        if use_flce:
            outputs = model(input_ids=input_ids, labels=input_ids, **model_kwargs)
        else:
            outputs = model(input_ids=input_ids, **model_kwargs)
    finally:
        if hasattr(unwrapped, "log_lec_metrics"):
            unwrapped.log_lec_metrics = False

    local_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
    if use_flce:
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return loss, local_tokens, getattr(outputs, "lec_metrics", {})

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    local_tokens_tensor = torch.tensor(local_tokens, device=device, dtype=torch.float32)
    loss = loss_fn(shift_logits, shift_labels, local_tokens_tensor)
    return loss, local_tokens, getattr(outputs, "lec_metrics", {})


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
                loss, token_count, _ = forward_loss(
                    model,
                    loss_fn,
                    batch["input_ids"],
                    device,
                    use_flce,
                    attention_forward_kwargs,
                    collect_lec_metrics=False,
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
        resume_ckpt = config.get("resume_from_checkpoint")
        if resume_ckpt:
            model, start_step, opt_state, sched_state = load_checkpoint(
                config, hf_assets_dir
            )
        else:
            model = build_model(config, hf_assets_dir)
            start_step = 0
            opt_state = None
            sched_state = None
        model.to(device)

        if tokenizer_size != model.config.vocab_size:
            raise ValueError(
                f"Tokenizer/model vocab mismatch: tokenizer={tokenizer_size}, "
                f"model={model.config.vocab_size}. Regenerate assets/config before training."
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

        local_tokens_per_update = (
            config["per_device_batch_size"]
            * config["grad_accum_steps"]
            * (config["max_seq_len"] - 1)
        )
        global_tokens_per_update = local_tokens_per_update * get_world_size()
        print0(f"[Train] Local tokens/step: {local_tokens_per_update:,}")
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
            f"[Train] Total: {global_tokens_per_update * config['max_steps'] / 1e9:.1f}B "
            f"tokens over {config['max_steps']:,} optimizer steps"
        )
        if use_flce:
            print0("[Loss] Liger fused_linear_cross_entropy ON")
        else:
            print0("[Loss] TorchTitan cross_entropy_loss")

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
            log_lec_this_step = (
                config["lec_metric_log_every_steps"] > 0
                and step % config["lec_metric_log_every_steps"] == 0
            )

            accum_loss_sum = 0.0
            accum_tokens = 0
            metric_sums = {}
            metric_count = 0

            for _ in range(config["grad_accum_steps"]):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                with get_autocast_context(device):
                    normalized_loss, token_count, lec_metrics = forward_loss(
                        model,
                        loss_fn,
                        batch["input_ids"],
                        device,
                        use_flce,
                        attention_forward_kwargs,
                        collect_lec_metrics=log_lec_this_step,
                    )
                    loss = normalized_loss / config["grad_accum_steps"]

                loss.backward()
                accum_loss_sum += normalized_loss.detach().item() * token_count
                accum_tokens += token_count

                if log_lec_this_step and lec_metrics:
                    for key, value in lec_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                    metric_count += 1

            grad_norm = None
            if config["grad_clip"] is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                ).item()

            flush_router_load_balance(model)
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
            reduced_lec_metrics = reduce_mean_metrics(metric_sums, metric_count, device)
            postfix = {
                "loss": f"{mean_loss:.4f}",
                "ppl": f"{ppl:.2f}",
                "lr": f"{lr:.2e}",
                "toks": f"{tok_s:,.0f}",
            }
            if "router_entropy" in reduced_lec_metrics:
                postfix["router"] = f"{reduced_lec_metrics['router_entropy']:.3f}"
            train_pbar.set_postfix(postfix)

            if step % config["log_every_steps"] == 0:
                log_line = (
                    f"[Step {step:05d}] loss={mean_loss:.4f} | ppl={ppl:.2f} | "
                    f"lr={lr:.3e} | tok/s={tok_s:,.0f}"
                    + (f" | gnorm={grad_norm:.3f}" if grad_norm is not None else "")
                )
                if reduced_lec_metrics:
                    if "router_entropy" in reduced_lec_metrics:
                        log_line += f" | router_entropy={reduced_lec_metrics['router_entropy']:.3f}"
                    if "expert_load_balance_std" in reduced_lec_metrics:
                        log_line += (
                            " | load_std="
                            f"{reduced_lec_metrics['expert_load_balance_std']:.3f}"
                        )
                print0(log_line)

                if wandb_run is not None:
                    payload = {
                        "train/loss": mean_loss,
                        "train/ppl": ppl,
                        "train/lr": lr,
                        "train/tokens_per_sec": tok_s,
                        "train/tokens_seen": tokens_seen,
                        "step": step,
                    }
                    if grad_norm is not None:
                        payload["train/grad_norm"] = grad_norm
                    if aux_lr is not None:
                        payload["train/aux_lr"] = aux_lr
                    if device.type == "cuda":
                        payload["sys/gpu_mem_alloc_gb"] = (
                            torch.cuda.memory_allocated(device) / 1e9
                        )
                        payload["sys/gpu_mem_reserved_gb"] = (
                            torch.cuda.memory_reserved(device) / 1e9
                        )
                    for key, value in reduced_lec_metrics.items():
                        payload[f"lec/{key}"] = value
                    wandb_run.log(payload, step=step)

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

        train_pbar.close()
        save_checkpoint(model, optimizer, scheduler, config["max_steps"], config)
        print0("[Train] LLaMA LEC hook-based TorchTitan pretraining complete.")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


if __name__ == "__main__":
    train(CONFIG)
