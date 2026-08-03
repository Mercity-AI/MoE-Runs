"""Shared training, logging, optimizer, checkpoint, and runtime utilities."""

import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel

from model import LlamaLongCatNgram, LlamaLongCatNgramConfig

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

    CrossEntropyLoss = getattr(torchtitan_loss, "CrossEntropyLoss", None)
    cross_entropy_loss = getattr(torchtitan_loss, "cross_entropy_loss", None)
    JOB_CONFIG_IMPORT_ERROR = None
except ImportError as exc:
    CrossEntropyLoss = None
    cross_entropy_loss = None
    HuggingFaceTokenizer = None
    JobConfig = None
    JOB_CONFIG_IMPORT_ERROR = exc


def require_torchtitan():
    if JOB_CONFIG_IMPORT_ERROR is not None:
        raise ImportError("TorchTitan is required by these entrypoints.") from JOB_CONFIG_IMPORT_ERROR
    if CrossEntropyLoss is None and cross_entropy_loss is None:
        raise ImportError("No supported TorchTitan cross-entropy API was found.")


def maybe_enable_liger_kernel(config: dict):
    if not config.get("use_liger_kernel", False):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Liger Kernel requires CUDA.")
    if apply_liger_kernel_to_llama is None:
        raise ImportError("Liger Kernel is enabled but not installed.") from LIGER_IMPORT_ERROR
    liger_config = dict(config.get("liger_kernel_config", {}))
    if liger_config.get("cross_entropy") and liger_config.get("fused_linear_cross_entropy"):
        raise ValueError("Liger cross_entropy and fused_linear_cross_entropy conflict.")
    apply_liger_kernel_to_llama(**liger_config)
    print0(f"[Liger] Enabled with config: {liger_config}")


@contextmanager
def _patch_initialize_weights_compat():
    original = PreTrainedModel.initialize_weights
    PreTrainedModel.initialize_weights = lambda self: None
    try:
        yield
    finally:
        PreTrainedModel.initialize_weights = original


def _architecture_types(architecture: str):
    if architecture == "longcat_ngram":
        return LlamaLongCatNgramConfig, LlamaLongCatNgram
    if architecture == "llama":
        return LlamaConfig, LlamaForCausalLM
    raise ValueError(f"Unknown architecture: {architecture!r}")


def build_model_config(config: dict, tokenizer, architecture: str):
    common = dict(
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
    if architecture == "longcat_ngram":
        return LlamaLongCatNgramConfig(
            architectures=["LlamaLongCatNgram"],
            ngram_max_n=config["ngram_max_n"],
            ngram_num_heads=config["ngram_num_heads"],
            ngram_table_vocab_sizes=config["ngram_table_vocab_sizes"],
            ngram_embedding_amplification=config["ngram_embedding_amplification"],
            qk_norm=config.get("qk_norm", False),
            qk_norm_eps=config.get("qk_norm_eps"),
            **common,
        )
    if architecture == "llama":
        return LlamaConfig(architectures=["LlamaForCausalLM"], **common)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def ensure_hf_assets(config: dict, architecture: str) -> tuple[Path, int]:
    assets_dir = Path(config["hf_assets_dir"]).resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.model_max_length = config["max_seq_len"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(assets_dir)
    build_model_config(config, tokenizer, architecture).save_pretrained(assets_dir)
    (assets_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print0(f"[Assets] Ready at {assets_dir} | vocab={len(tokenizer)}")
    return assets_dir, len(tokenizer)


def load_tokenizer_size(hf_assets_dir: Path) -> int:
    return len(AutoTokenizer.from_pretrained(hf_assets_dir))


def build_torchtitan_tokenizer(hf_assets_dir: Path):
    require_torchtitan()
    if hasattr(HuggingFaceTokenizer, "Config"):
        tokenizer = HuggingFaceTokenizer(
            HuggingFaceTokenizer.Config(), tokenizer_path=str(hf_assets_dir)
        )
    else:
        tokenizer = HuggingFaceTokenizer(str(hf_assets_dir))
    return tokenizer


def build_torchtitan_ce_loss():
    require_torchtitan()
    if CrossEntropyLoss is not None:
        loss_object = CrossEntropyLoss(CrossEntropyLoss.Config())
        return lambda pred, labels, global_valid_tokens=None: loss_object(
            pred, labels, global_valid_tokens
        )

    def loss_fn(pred, labels, global_valid_tokens=None):
        loss = cross_entropy_loss(pred, labels)
        return loss / global_valid_tokens if global_valid_tokens is not None else loss

    return loss_fn


def _log_model_stats(model, architecture: str):
    total = sum(parameter.numel() for parameter in model.parameters())
    if architecture == "longcat_ngram":
        ngram = sum(parameter.numel() for parameter in model.ngram_embedder.parameters())
        print0(
            f"[LongCat-Ngram] Base: {total-ngram:,} | N-gram: {ngram:,} "
            f"({100 * ngram / total:.2f}%)"
        )
    print0(f"[Model] {architecture}: {total:,} params ({total / 1e9:.2f}B)")


def build_model(config: dict, hf_assets_dir: Path, architecture: str):
    require_torchtitan()
    maybe_enable_liger_kernel(config)
    config_class, model_class = _architecture_types(architecture)
    model_config = config_class.from_pretrained(hf_assets_dir)
    model_config.max_position_embeddings = max(
        model_config.max_position_embeddings, config["max_seq_len"]
    )
    with _patch_initialize_weights_compat():
        model = model_class._from_config(
            model_config,
            attn_implementation=config["attn_implementation"],
            dtype=torch.bfloat16,
        )
    model.apply(model._init_weights)
    _log_model_stats(model, architecture)
    return model


def load_pretrained_model(config: dict, checkpoint: Path, architecture: str):
    maybe_enable_liger_kernel(config)
    config_class, model_class = _architecture_types(architecture)
    model_config = config_class.from_pretrained(checkpoint)
    model_config.max_position_embeddings = max(
        model_config.max_position_embeddings, config["max_seq_len"]
    )
    with _patch_initialize_weights_compat():
        model = model_class.from_pretrained(
            checkpoint,
            config=model_config,
            attn_implementation=config["attn_implementation"],
            torch_dtype=torch.bfloat16,
        )
    _log_model_stats(model, architecture)
    return model


def resolve_optimizer_name(config: dict) -> str:
    name = str(config.get("optimizer_name", "adamw")).strip().lower()
    if name in {"adam", "adamw"}:
        return "adamw"
    if name == "muon":
        return "muon"
    raise NotImplementedError(f"Unsupported optimizer: {name}")


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


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print0(message: str):
    if is_main_process():
        print(message, flush=True)


def barrier():
    if is_dist_initialized():
        dist.barrier()


def setup_runtime(distributed: bool) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    if distributed and world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def cleanup_runtime():
    if is_dist_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    seed += get_rank()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_autocast_context(device: torch.device):
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


def progress_bar(*args, **kwargs):
    kwargs.setdefault("disable", not is_main_process())
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    return tqdm(*args, **kwargs)


def reduce_sum_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if is_dist_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def maybe_init_tracker(config: dict, run_name: str):
    """Initialize W&B; the complete runtime config is recorded once."""
    if not config["use_wandb"] or not is_main_process():
        return None
    return wandb.init(project=config["wandb_project"], name=run_name, config=config)


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
        print0("[Bench] lm_eval not installed; skipping.")
        return {}

    base_model = unwrap_model(model)
    base_model.eval()
    lm = HFLM(
        pretrained=base_model,
        tokenizer=AutoTokenizer.from_pretrained(hf_assets_dir),
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
    return {
        f"bench/{task}/{metric}": value
        for task, task_metrics in results["results"].items()
        for metric, value in task_metrics.items()
        if isinstance(value, float)
    }


def resolve_training_steps(config: dict, world_size: int = 1) -> int:
    target_tokens = config.get("target_train_tokens")
    if target_tokens is None:
        return config["max_steps"]
    tokens_per_update = (
        config["per_device_batch_size"]
        * config["grad_accum_steps"]
        * (config["max_seq_len"] - 1)
        * world_size
    )
    return max(1, round(target_tokens / tokens_per_update))


def build_job_config(config: dict, world_size: int = 1):
    if JOB_CONFIG_IMPORT_ERROR is not None:
        raise ImportError("TorchTitan JobConfig is unavailable.") from JOB_CONFIG_IMPORT_ERROR
    optimizer_name = resolve_optimizer_name(config)
    job = JobConfig()
    job.job.dump_folder = str(Path(config["output_dir"]).resolve())
    job.optimizer.name = "Muon" if optimizer_name == "muon" else "AdamW"
    job.optimizer.lr = config.get("muon_lr", 0.02) if optimizer_name == "muon" else config["learning_rate"]
    job.optimizer.beta1 = config["beta1"]
    job.optimizer.beta2 = config["beta2"]
    job.optimizer.eps = config["optimizer_eps"]
    job.optimizer.weight_decay = config.get("muon_weight_decay", config["weight_decay"]) if optimizer_name == "muon" else config["weight_decay"]
    job.optimizer.implementation = "muon" if optimizer_name == "muon" else config.get("optimizer_implementation") or ("fused" if torch.cuda.is_available() else "for-loop")
    job.lr_scheduler.warmup_steps = config["warmup_steps"]
    job.lr_scheduler.decay_ratio = config.get("lr_decay_ratio")
    job.lr_scheduler.decay_type = config.get("lr_decay_type", "cosine")
    job.lr_scheduler.min_lr_factor = config.get("min_lr_factor", 0.0)
    job.training.local_batch_size = config["per_device_batch_size"]
    job.training.global_batch_size = config["per_device_batch_size"] * config["grad_accum_steps"] * world_size
    job.training.seq_len = config["max_seq_len"]
    job.training.steps = resolve_training_steps(config, world_size)
    job.training.max_norm = config["grad_clip"]
    return job


class NativeMuonWithAuxAdam:
    """Pair Muon matrix updates with AdamW for embeddings/norms/biases."""

    def __init__(self, muon, adam: AdamW):
        self.muon = muon
        self.adam = adam
        self.param_groups = muon.param_groups + adam.param_groups

    def step(self):
        self.muon.step()
        self.adam.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])


class DualLRScheduler:
    def __init__(self, muon: LambdaLR, adam: LambdaLR):
        self.muon = muon
        self.adam = adam

    def step(self):
        self.muon.step()
        self.adam.step()

    def get_last_lr(self):
        return self.muon.get_last_lr()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()


def build_optimizer(model: torch.nn.Module, job, config: dict):
    base_model = unwrap_model(model)
    if resolve_optimizer_name(config) == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise ImportError("Muon requires PyTorch >= 2.11.")
        muon_params, aux_decay, aux_no_decay = [], [], []
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.layers.") and param.ndim >= 2:
                muon_params.append(param)
            elif _is_no_decay(name, param):
                aux_no_decay.append(param)
            else:
                aux_decay.append(param)
        if not muon_params:
            raise RuntimeError("Muon partitioning found no hidden matrix parameters.")
        muon = torch.optim.Muon(
            muon_params,
            lr=config.get("muon_lr", 0.02),
            momentum=config.get("muon_momentum", 0.95),
            nesterov=config.get("muon_nesterov", True),
            weight_decay=config.get("muon_weight_decay", config["weight_decay"]),
            ns_steps=config.get("muon_ns_steps", 5),
        )
        implementation = config.get("optimizer_implementation", "for-loop")
        adam_kwargs = {"fused": True} if implementation == "fused" else {"foreach": True} if implementation == "foreach" else {}
        if implementation not in {"fused", "foreach", "for-loop"}:
            raise NotImplementedError(f"Unsupported optimizer implementation: {implementation}")
        adam = AdamW(
            [
                {"params": aux_decay, "weight_decay": config.get("aux_adam_weight_decay", config["weight_decay"])},
                {"params": aux_no_decay, "weight_decay": 0.0},
            ],
            lr=config.get("aux_adam_lr", config["learning_rate"]),
            betas=(config.get("aux_adam_beta1", config["beta1"]), config.get("aux_adam_beta2", config["beta2"])),
            eps=config.get("aux_adam_eps", config["optimizer_eps"]),
            **adam_kwargs,
        )
        return NativeMuonWithAuxAdam(muon, adam)

    decay, no_decay = [], []
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            (no_decay if _is_no_decay(name, param) else decay).append(param)
    implementation = job.optimizer.implementation
    optim_kwargs = {"fused": True} if implementation == "fused" else {"foreach": True} if implementation == "foreach" else {}
    if implementation not in {"fused", "foreach", "for-loop"}:
        raise NotImplementedError(f"Unsupported optimizer implementation: {implementation}")
    return AdamW(
        [
            {"params": decay, "weight_decay": job.optimizer.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=job.optimizer.lr,
        betas=(job.optimizer.beta1, job.optimizer.beta2),
        eps=job.optimizer.eps,
        **optim_kwargs,
    )


def build_scheduler(optimizer, job):
    warmup = job.lr_scheduler.warmup_steps
    max_steps = job.training.steps
    ratio = job.lr_scheduler.decay_ratio
    min_factor = job.lr_scheduler.min_lr_factor
    decay_type = job.lr_scheduler.decay_type
    decay_start = max(warmup, max_steps - max(1, int(max_steps * ratio))) if ratio is not None else warmup

    def lr_lambda(step: int):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        if step < decay_start:
            return 1.0
        progress = min(max((step - decay_start) / max(1, max_steps - decay_start), 0.0), 1.0)
        if decay_type == "linear":
            decay = 1.0 - progress
        elif decay_type == "sqrt":
            decay = 1.0 - math.sqrt(progress)
        elif decay_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise NotImplementedError(f"Unsupported LR decay: {decay_type}")
        return min_factor + (1.0 - min_factor) * decay

    if isinstance(optimizer, NativeMuonWithAuxAdam):
        return DualLRScheduler(
            LambdaLR(optimizer.muon, lr_lambda=lr_lambda),
            LambdaLR(optimizer.adam, lr_lambda=lr_lambda),
        )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def format_optimizer_log(config: dict, job) -> list[str]:
    if resolve_optimizer_name(config) == "muon":
        return [
            "[Optim] Muon(hidden) + AuxAdam(non-hidden)",
            f"[Optim] muon_lr={config.get('muon_lr', 0.02):.3e} | "
            f"aux_lr={config.get('aux_adam_lr', config['learning_rate']):.3e} | "
            f"aux_impl={config.get('optimizer_implementation', 'for-loop')}",
        ]
    return [f"[Optim] {job.optimizer.name} | impl={job.optimizer.implementation}"]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, optimizer, scheduler, step: int, config: dict):
    barrier()
    if not is_main_process():
        barrier()
        return
    root = Path(config["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    hf_dir = root / f"step_{step:06d}_hf"
    unwrap_model(model).save_pretrained(hf_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(Path(config["hf_assets_dir"]).resolve()).save_pretrained(hf_dir)
    (hf_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_path = root / f"step_{step:06d}_train_state.pt"
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "world_size": get_world_size(),
            "training_config": config,
        },
        state_path,
    )
    print0(f"[Checkpoint] HF model -> {hf_dir}")
    print0(f"[Checkpoint] Train state -> {state_path}")
    barrier()


def load_checkpoint(
    config: dict,
    architecture: str,
    checkpoint_path: Optional[str] = None,
    resume_training: bool = True,
):
    checkpoint = Path(checkpoint_path or config["resume_from_checkpoint"]).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = load_pretrained_model(config, checkpoint, architecture)
    match = re.search(r"step_(\d+)", checkpoint.name)
    checkpoint_step = int(match.group(1)) if match else 0
    if not resume_training:
        print0("[Init] Loaded weights only; training state starts fresh.")
        return model, 0, None, None

    state_path = checkpoint.parent / f"step_{checkpoint_step:06d}_train_state.pt"
    optimizer_state = scheduler_state = None
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_world_size = saved.get("world_size")
        if saved_world_size is not None and saved_world_size != get_world_size():
            raise ValueError("Exact resume requires the checkpoint's original world size.")
        saved_config = saved.get("training_config")
        critical = (
            "dataset_name", "dataset_config", "seed", "max_seq_len",
            "per_device_batch_size", "grad_accum_steps", "dataloader_workers",
            "streaming_buffer_size", "eval_holdout_fraction",
        )
        if saved_config is not None:
            mismatches = [key for key in critical if saved_config.get(key) != config.get(key)]
            if mismatches:
                raise ValueError(f"Exact resume data settings changed: {mismatches}")
        elif checkpoint_step and not config.get("allow_inexact_legacy_data_resume", False):
            raise ValueError("Legacy checkpoint cannot guarantee exact data resume.")
        optimizer_state = saved.get("optimizer")
        scheduler_state = saved.get("scheduler")
    else:
        print0(f"[Resume] No train state at {state_path}; optimizer starts fresh.")
    print0(f"[Resume] Loaded step {checkpoint_step} from {checkpoint}")
    return model, checkpoint_step, optimizer_state, scheduler_state


def forward_loss(
    model,
    loss_fn,
    input_ids: torch.Tensor,
    device: torch.device,
    use_flce: bool,
    attention_forward_kwargs: Optional[dict] = None,
) -> tuple[torch.Tensor, int]:
    input_ids = input_ids.to(device, non_blocking=True)
    kwargs = dict(attention_forward_kwargs or {})
    if use_flce:
        outputs = model(input_ids, labels=input_ids, **kwargs)
        return outputs.loss if hasattr(outputs, "loss") else outputs[0], input_ids.shape[0] * (input_ids.shape[1] - 1)
    outputs = model(input_ids, **kwargs)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
    labels = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()
    token_count = labels.numel()
    loss = loss_fn(logits, labels, torch.tensor(token_count, device=device, dtype=torch.float32))
    return loss, token_count


@torch.no_grad()
def evaluate(
    model,
    eval_loader: DataLoader,
    loss_fn,
    device: torch.device,
    use_flce: bool,
    max_examples: Optional[int] = None,
    attention_forward_kwargs: Optional[dict] = None,
) -> dict:
    model.eval()
    loss_sum = tokens = 0.0
    examples = 0
    bar = progress_bar(eval_loader, desc="eval", position=1)
    try:
        for batch in bar:
            if max_examples is not None:
                remaining = max_examples - examples
                if remaining <= 0:
                    break
                batch["input_ids"] = batch["input_ids"][:remaining]
            with get_autocast_context(device):
                loss, token_count = forward_loss(
                    model, loss_fn, batch["input_ids"], device, use_flce,
                    attention_forward_kwargs,
                )
            loss_sum += loss.item() * token_count
            tokens += token_count
            examples += batch["input_ids"].shape[0]
    finally:
        bar.close()
    loss_sum = reduce_sum_scalar(loss_sum, device)
    tokens = reduce_sum_scalar(tokens, device)
    mean_loss = loss_sum / max(tokens, 1.0)
    model.train()
    return {"eval_loss": mean_loss, "eval_ppl": math.exp(min(mean_loss, 20.0))}


__all__ = [name for name in globals() if not name.startswith("_")]
