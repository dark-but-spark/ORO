#!/bin/bash
set -euo pipefail

# Fix:
#   ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
#
# Cause:
#   incompatible MKL / intel-openmp / libittnotify runtime versions.

module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiresunet

conda install -y "mkl<2024.1" "intel-openmp<2024.1" libittnotify

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

