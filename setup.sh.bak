#!/usr/bin/env bash
set -euo pipefail

# Install procedure from FA4.md.
# Default path targets CUDA 13 drivers. Set CUDA_VARIANT=cu12 to use the
# alternative wheel index and flash-attn-4 extra.

CUDA_VARIANT="${CUDA_VARIANT:-cu13}"

python -m pip uninstall -y torch torchvision torchaudio

if [[ "${CUDA_VARIANT}" == "cu12" ]]; then
  python -m pip install torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
else
  python -m pip install torch==2.11.0 torchvision torchaudio
fi

python -m pip install torchtitan==0.2.2 transformers==5.8.0 datasets==4.8.5 \
  wandb liger-kernel==0.8.0 accelerate \
  trackio tiktoken sentencepiece protobuf tqdm ninja

python -m pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-core \
  nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13 \
  flash-attn-4 \
  quack-kernels

if [[ "${CUDA_VARIANT}" == "cu12" ]]; then
  python -m pip install \
    "flash-attn-4==4.0.0b19" \
    "quack-kernels==0.5.0" \
    "nvidia-cutlass-dsl==4.5.2"
else
  python -m pip install \
    "flash-attn-4[cu13]==4.0.0b19" \
    "quack-kernels==0.5.0" \
    "nvidia-cutlass-dsl[cu13]==4.5.2"
fi

python -m pip check
