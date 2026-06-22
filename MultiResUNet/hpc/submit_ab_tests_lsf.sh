#!/bin/bash
set -euo pipefail

# Submit the AB test suite to LSF.
# Usage on cluster:
#   cd /work/phy-tongrj/MultiResUNet
#   bash hpc/submit_ab_tests_lsf.sh
#
# Queue is intentionally not specified. LSF will use the cluster default queue.

PROJECT="${PROJECT:-/work/phy-tongrj/MultiResUNet}"
CONDA_ENV="${CONDA_ENV:-multiresunet}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"

cd "$PROJECT"
mkdir -p lsf_logs models runs/logs

COMMON="--validation-split 0.1 --scale --scale-factor 0.5 --input-channels 3 --output-channels 4 --epochs 50 --batch-size 16 --learning-rate 2e-5 --gradient-clip 0.5 --weight-decay 2e-4 --num-workers 4 --prefetch-factor 2 --repeat-factor 1 --verbose --save-model --tensorboard --early-stopping-patience 8 --device cuda --seed 42"

make_job () {
  local name="$1"
  local extra="$2"
  local job_file="hpc/lsf_${name}.lsf"

  cat > "$job_file" <<JOB
#BSUB -J ${name}
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o lsf_logs/${name}.%J.out
#BSUB -e lsf_logs/${name}.%J.err

cd ${PROJECT}

module load ${CUDA_MODULE}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python train.py ${COMMON} --save-dir models/${name} ${extra}
JOB

  echo "Submitting ${name}"
  bsub < "$job_file"
}

make_job A_plain_baseline "--no-train-augmentation --lr-scheduler cosine"
make_job B_mild_aug "--train-augmentation --augmentation-strength mild --lr-scheduler cosine"
make_job B_strong_aug "--train-augmentation --augmentation-strength strong --lr-scheduler cosine"
make_job B_focal_loss "--no-train-augmentation --use-focal-loss --focal-alpha 0.25 --focal-gamma 1.0 --lr-scheduler cosine"
make_job B_combined_loss "--no-train-augmentation --use-combined-loss --bce-weight 0.7 --dice-weight 0.3 --lr-scheduler cosine"
make_job B_lr_step "--no-train-augmentation --lr-scheduler step --lr-step-size 20 --lr-gamma 0.5"
