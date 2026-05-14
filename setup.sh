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

python -m pip install torchtitan transformers datasets wandb liger-kernel accelerate \
  trackio tiktoken sentencepiece protobuf tqdm ninja

if [[ "${CUDA_VARIANT}" == "cu12" ]]; then
  python -m pip install --pre flash-attn-4
else
  python -m pip install --pre "flash-attn-4[cu13]"
fi
