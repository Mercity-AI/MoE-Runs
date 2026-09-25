# Dense 1B pretraining ablations: KDA and LongCat n-gram embeddings

Code for pretraining dense ~1B-parameter Llama-style checkpoints from scratch on 6B FineWeb tokens, testing two ideas from large sparse MoE models against a shared baseline:

- **KDA** (Kimi Delta Attention): 8 of 32 attention layers replaced with KDA.
- **LongCat n-gram embeddings**: hashed n-gram embedding tables that take 25% or ~48% of the parameter budget, paid for with transformer depth.

The same repo holds the downstream NER fine-tuning (LoRA on Few-NERD) and its evaluation and analysis scripts.

## Repository layout

```
scripts/
  pretraining/   training scripts, model definitions, shared data and training helpers
  ner/           NER fine-tuning, evaluation, vLLM port
    analysis/    NER analysis scripts
configs/
  ner/           one NER fine-tuning config per run
docs/            design notes: FA4 setup, n-gram sizing, KDA config, packing, ablation plan
```

| Path | What it is |
| --- | --- |
| `scripts/pretraining/baseline_llama_torchtitan.py` | Dense 1B baseline (QK-norm on), TorchTitan training loop |
| `scripts/pretraining/baseline_llama_KDA.py` | Baseline with hybrid KDA + softmax attention |
| `scripts/pretraining/baseline_llama_torchtitan_longcat_ngram.py` | Baseline with LongCat n-gram embedding tables |
| `scripts/pretraining/model.py` | Model definitions: `LlamaQKNorm`, `LlamaKDA`, `LlamaLongCatNgram` |
| `scripts/pretraining/utils.py`, `data.py` | Optimizer (Muon + AdamW), checkpointing, logging; packed streaming FineWeb |
| `scripts/pretraining/kda_benchmark_common.py` | Shared helpers for KDA benchmarking |
| `scripts/ner/ner_data.py` | NER prompt template, Few-NERD loading, JSON target building and parsing |
| `scripts/ner/ner_sft.py` | LoRA NER fine-tuning, driven by one YAML config |
| `scripts/ner/ner_eval.py` | NER eval with Hugging Face `generate` (reference, slow) |
| `scripts/ner/ner_eval_vllm.py` | NER eval with vLLM (fast) |
| `scripts/ner/vllm_env.py`, `vllm_ngram_model.py` | vLLM environment patch and vLLM port of the custom architectures |
| `scripts/ner/analysis/` | Parity check, significance tests, failure modes, win/loss, validation loss |
| `configs/ner/` | NER fine-tuning configs |
| `eval.py` | Zero-shot benchmark evaluation |
| `setup.sh`, `requirements.txt` | Environment setup |

## Setup

### 1. Python environment

`setup.sh` installs a fresh `torch` 2.11.0, TorchTitan, `transformers`, `datasets`, Liger kernels, `flash-linear-attention` (needed for KDA), and a CUDA-matched build of FlashAttention 4. It targets CUDA 13 by default (Blackwell: B200, RTX 5090); set `CUDA_VARIANT=cu12` on a CUDA 12 machine.

```bash
CUDA_VARIANT=cu13 bash setup.sh
```

The FlashAttention 4 pip package is `flash-attn-4`, but it is imported as `from flash_attn import cute`.

Extra packages, depending on what you run:

```bash
pip install peft scikit-learn   # NER fine-tuning and analysis
pip install bitsandbytes        # 4-bit quantization eval
pip install vllm==0.29.0        # fast NER eval (see the note under "NER evaluation")
```

Log in to Weights & Biases (`wandb login`), or set `WANDB_MODE=offline`.

Sanity check:

```bash
python -c "
import torch, transformers
print(torch.__version__, torch.cuda.is_available(), transformers.__version__)
from flash_attn import cute
import fla, torchtitan
print('all imports OK')
"
```

### 2. Base checkpoints

The trained checkpoints live in the Hugging Face bucket `Mercity/MoE-bucket`. The `hf buckets` commands need a recent `huggingface_hub` (2.0 works).

```bash
hf auth login
mkdir -p checkpoints/baseline checkpoints/kda checkpoints/longcat_ngram_50 checkpoints/longcat_ngram_25

hf buckets sync "hf://buckets/Mercity/MoE-bucket/baseline_llama_rerun_qknorm_1408/step_003053_hf"          "checkpoints/baseline/step_003053_hf"
hf buckets sync "hf://buckets/Mercity/MoE-bucket/checkpoints_kda_run_1308_12h_1b_6b/step_003053_hf"        "checkpoints/kda/step_003053_hf"
hf buckets sync "hf://buckets/Mercity/MoE-bucket/checkpoints_llama_1b_longcat_ngram_6b_2307/step_003053_hf" "checkpoints/longcat_ngram_50/step_003053_hf"
hf buckets sync "hf://buckets/Mercity/MoE-bucket/checkpoints_llama_ngram_25pct_1108/step_003053_hf"        "checkpoints/longcat_ngram_25/step_003053_hf"
```

Only the final `step_003053_hf` folder is needed (about 8 GB for all four). Each one ships its own `model.py`, so load it with `trust_remote_code=True`:

| Local path | Model | Architecture class |
| --- | --- | --- |
| `checkpoints/baseline/step_003053_hf` | QK-norm baseline, 32 layers | `LlamaQKNorm` |
| `checkpoints/kda/step_003053_hf` | KDA, 24 GQA + 8 KDA layers | `LlamaKDA` |
| `checkpoints/longcat_ngram_50/step_003053_hf` | N-gram 50%, 16 layers | `LlamaLongCatNgram` |
| `checkpoints/longcat_ngram_25/step_003053_hf` | N-gram 25%, 23 layers | `LlamaLongCatNgram` |

## Pretraining

Every pretraining script is configured by the `CONFIG` dict at the top of the file (model shape, n-gram or KDA settings, optimizer, batch size, token budget, `output_dir`). Edit it, then run the script from the repo root on a single GPU (`output_dir` is relative to where you launch):

```bash
python scripts/pretraining/baseline_llama_torchtitan.py                 # dense baseline
python scripts/pretraining/baseline_llama_KDA.py                        # KDA hybrid
python scripts/pretraining/baseline_llama_torchtitan_longcat_ngram.py   # LongCat n-gram
```

The runs in this project used a single B200 (180 GB), streaming FineWeb `sample-10BT`, 8,192-token packed sequences, and 6B tokens (3,053 steps). They took 10.5 to 18 hours each. Checkpoints are written to `output_dir`, including a Hugging Face-format `step_XXXXXX_hf` folder.

`torch.compile` is off: it clashed with the Liger kernels, and compiling the fused cross-entropy kernel silently corrupted gradients.

## Benchmark evaluation

```bash
python eval.py --checkpoint checkpoints/kda/step_003053_hf
```

`eval.py` in this repo scores HellaSwag and WinoGrande by log-likelihood. The full nine-task suite used in the reports (via `lm-eval`) lives in the newer `eval.py` in the bucket's `2209-moe-runs/scripts/`.

## NER fine-tuning

NER is framed as generative SFT: the model reads a task prompt plus a sentence and writes the entities as JSON (`{"person": ["..."], "location": ["..."]}`). Training uses the Hugging Face `Trainer` with LoRA on every transformer layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); the n-gram tables stay frozen.

```bash
python scripts/ner/ner_sft.py configs/ner/qknorm_baseline.yaml
```

Before running, edit the config:

- `checkpoint`: path to one of the base checkpoints above. The configs currently point at absolute `/workspace/...` paths from the training machine.
- `output_dir`: where the run is saved.

| Config | Base model | Recipe |
| --- | --- | --- |
| `configs/ner/qknorm_baseline.yaml` | Baseline | batch 30 × grad accum 4, LR 3e-5, 2 epochs |
| `configs/ner/ngram_longcat_25.yaml` | N-gram 25% | same as baseline |
| `configs/ner/ngram_longcat.yaml` | N-gram 50% | batch 20, LR 5e-5: change to batch 30 / LR 3e-5 for a like-for-like comparison |
| `configs/ner/smoke_test.yaml` | any | short run to check the pipeline |

Notes:

- Inputs are `BOS + prompt + target + EOS`, with the prompt masked out of the loss. The tokenizer does **not** add BOS itself; `ner_data.py` adds it, and any inference path must too.
- `ner_sft.py` prints three worked examples at start-up. Read them before letting a run go.
- A checkpoint is saved every 250 steps, plus the final adapter at the run root. Each holds the LoRA adapter, tokenizer and `ner_config.yaml`.
- Use `attn_implementation: sdpa`.
- One run takes about 1 h 45 min (n-gram 25%) to 2 h 20 min (baseline) on an RTX 5090.

## NER evaluation

Two eval paths score the same way:

```bash
# Hugging Face generate: the reference, slow
python scripts/ner/ner_eval.py --run ner_runs/qknorm_baseline_json --eval-limit 100000 --max-new-tokens 1024 \
  --output results/baseline_hf.json

# vLLM: full validation set (18,824 sentences) in minutes
python scripts/ner/ner_eval_vllm.py --run ner_runs/qknorm_baseline_json --eval-limit 100000 --max-new-tokens 1024 \
  --output results/baseline_vllm.json
```

- `--eval-limit 100000` evaluates the whole validation set.
- Pass `--base <checkpoint>` if the run's saved `ner_config.yaml` points at a base-checkpoint path that doesn't exist on this machine.
- Lower `--gpu-memory-utilization` (default 0.9) if another job shares the GPU; 0.2 is enough for a 1B model.

**vLLM needs its own port of these architectures** (`scripts/ner/vllm_ngram_model.py`), and `vllm_env.py` must be imported before `vllm`. Installing `vllm==0.29.0` upgrades `torch` and `transformers` past the versions in `setup.sh`. Before trusting vLLM numbers for a new checkpoint, or after changing `vllm_ngram_model.py`, run the parity check:

```bash
python scripts/ner/analysis/vllm_hf_parity.py --run ner_runs/ngram_longcat_25_json
```

A healthy port shows ≥99% teacher-forced top-1 agreement with Hugging Face and most greedy generations identical. The script exits with status 1 if agreement is below the threshold.

The n-gram port depends on a few rules that the eval script enforces:

1. N-gram models decode one sequence at a time (`max_num_seqs=1`).
2. Prefix caching is off.
3. Prompts are passed as token IDs with BOS, not as strings.
4. The n-gram hash must match the checkpoint: n-gram 25% uses the per-table multipliers in `config.json` (`ngram_hash_multipliers`), n-gram 50% uses the vocabulary size.

## NER analysis

All analysis scripts take the eval JSONs written by `ner_eval.py` / `ner_eval_vllm.py`:

```bash
R=results
python scripts/ner/analysis/significance.py   $R/baseline_vllm.json $R/ngram25_vllm.json       # paired bootstrap, randomization, McNemar
python scripts/ner/analysis/failure_modes.py  baseline=$R/baseline_vllm.json ngram25=$R/ngram25_vllm.json
python scripts/ner/analysis/win_loss.py       $R/baseline_vllm.json $R/ngram25_vllm.json       # who wins, by entity feature
python scripts/ner/analysis/compare_evals.py  $R/baseline_hf.json $R/baseline_vllm.json        # sentence-by-sentence diff
python scripts/ner/analysis/val_loss.py ner_runs/ngram_longcat_25_json checkpoints/longcat_ngram_25/step_003053_hf
```

`win_loss.py` caches Few-NERD training-set entity counts in `scripts/ner/analysis/train_entity_counts.pkl` on first run.
