#!/bin/bash
set -euo pipefail

# Run this once on the cluster after uploading MultiResUNet.
# Default install location: /work/phy-tongrj/miniconda3/envs/multiresunet

module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh

conda create -n multiresunet python=3.10 -y
conda activate multiresunet

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install \
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
