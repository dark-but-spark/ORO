#!/bin/bash
set -euo pipefail

# Recreate the environment with PyTorch official pip CUDA wheels.
# Use this when conda PyTorch fails with:
#   libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent

module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh

conda deactivate || true
conda env remove -n multiresunet -y || true

conda create -n multiresunet python=3.10 -y
conda activate multiresunet

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  numpy \
  scipy \
  scikit-learn \
  opencv-python-headless \
  tqdm \
  matplotlib \
  tensorboard \
  pandas \
  pillow \
  psutil \
  tensorflow-cpu \
  keras

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
import cv2
import tensorboard
import sklearn
print("core packages ok")
PY

