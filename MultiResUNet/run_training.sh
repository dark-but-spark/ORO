#!/bin/bash

# MultiResUNet Training Script with Logging and Backup
# This script trains MultiResUNet and saves all outputs with timestamps

# Activate the Python environment
source /share/home/zjm/anaconda3/bin/activate zjm


set -e  # Exit on error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${PROJECT_DIR}/runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default training parameters (OPTIMIZED for 3000-4000 samples @ 640x640)
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-4}              # Reduced for large datasets
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATA_LIMIT=${DATA_LIMIT:-""}  # Empty means use all data (None in Python)
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.1}
INPUT_CHANNELS=${INPUT_CHANNELS:-3}
OUTPUT_CHANNELS=${OUTPUT_CHANNELS:-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}      # Prevent gradient explosion
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-8}            # Optimized for 32-core CPU
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}    # Increased prefetch for large datasets
TENSORBOARD=${TENSORBOARD:-true}         # Enable TensorBoard by default

# Memory protection: Auto-adjust based on data size
MEMORY_SAFE_MODE=${MEMORY_SAFE_MODE:-true}  # Enable automatic memory management

# Scale optimization for large datasets
SCALE_ENABLED=${SCALE_ENABLED:-false}
SCALE_FACTOR=${SCALE_FACTOR:-0.5}  # 50% reduction (640x640 -> 320x320)

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --learning-rate) LEARNING_RATE="$2"; shift ;;
        --data-limit) DATA_LIMIT="$2"; shift ;;
        --validation-split) VALIDATION_SPLIT="$2"; shift ;;
        --input-channels) INPUT_CHANNELS="$2"; shift ;;
        --output-channels) OUTPUT_CHANNELS="$2"; shift ;;
        --gradient-clip) GRADIENT_CLIP="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift ;;
        --prefetch-factor) PREFETCH_FACTOR="$2"; shift ;;
        --tensorboard) TENSORBOARD="true"; shift ;;
        --no-tensorboard) TENSORBOARD="false"; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs NUM           Number of training epochs (default: 50)"
            echo "  --batch-size NUM       Batch size for training (default: 2)"
            echo "  --learning-rate FLOAT  Learning rate (default: 1e-4)"
            echo "  --data-limit NUM       Number of training samples (default: 100)"
            echo "  --validation-split FLOAT Validation split ratio (default: 0.2)"
            echo "  --input-channels NUM   Number of input channels (default: 3)"
            echo "  --output-channels NUM  Number of output channels (default: 4)"
            echo "  --gradient-clip FLOAT  Gradient clipping threshold (default: 1.0)"
            echo "  --device DEVICE        Training device: cuda or cpu (default: cuda)"
            echo "  --num-workers NUM      Number of data loading workers (default: 6 for 32-core CPU)"
            echo "  --prefetch-factor NUM  Batches prefetched per worker (default: 3)"
            echo "  --tensorboard          Enable TensorBoard logging (default: true)"
            echo "  --no-tensorboard       Disable TensorBoard logging"
            echo "  --help                 Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --epochs 100 --batch-size 4 --data-limit 200 --tensorboard"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Create directories
mkdir -p "${RUNS_DIR}"
mkdir -p "${RUNS_DIR}/models"
mkdir -p "${RUNS_DIR}/logs"
mkdir -p "${RUNS_DIR}/histories"

# Create log file with timestamp
LOG_FILE="${RUNS_DIR}/logs/training_${TIMESTAMP}.log"
echo "========================================" | tee "$LOG_FILE"
echo "MultiResUNet Training Log" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "Project Directory: ${PROJECT_DIR}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Print configuration
echo "Training Configuration:" | tee -a "$LOG_FILE"
echo "  Epochs: ${EPOCHS}" | tee -a "$LOG_FILE"
echo "  Batch Size: ${BATCH_SIZE}" | tee -a "$LOG_FILE"
echo "  Learning Rate: ${LEARNING_RATE}" | tee -a "$LOG_FILE"
echo "  Data Limit: ${DATA_LIMIT:-None}" | tee -a "$LOG_FILE"
echo "  Validation Split: ${VALIDATION_SPLIT}" | tee -a "$LOG_FILE"
echo "  Input Channels: ${INPUT_CHANNELS}" | tee -a "$LOG_FILE"
echo "  Output Channels: ${OUTPUT_CHANNELS}" | tee -a "$LOG_FILE"
echo "  Gradient Clip: ${GRADIENT_CLIP}" | tee -a "$LOG_FILE"
echo "  Device: ${DEVICE}" | tee -a "$LOG_FILE"
echo "  TensorBoard: ${TENSORBOARD}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Navigate to project directory
cd "${PROJECT_DIR}"

# Run training and log output
echo "Starting training..." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Build training command with conditional data-limit handling
TRAIN_CMD="python -u train.py"
TRAIN_CMD+=" --epochs ${EPOCHS}"
TRAIN_CMD+=" --batch-size ${BATCH_SIZE}"
TRAIN_CMD+=" --learning-rate ${LEARNING_RATE}"

# Only add --data-limit if it's specified (non-empty)
if [ -n "${DATA_LIMIT}" ]; then
    TRAIN_CMD+=" --data-limit ${DATA_LIMIT}"
fi

TRAIN_CMD+=" --validation-split ${VALIDATION_SPLIT}"
TRAIN_CMD+=" --input-channels ${INPUT_CHANNELS}"
TRAIN_CMD+=" --output-channels ${OUTPUT_CHANNELS}"
TRAIN_CMD+=" --gradient-clip ${GRADIENT_CLIP}"
TRAIN_CMD+=" --device ${DEVICE}"
TRAIN_CMD+=" --num-workers ${NUM_WORKERS}"
TRAIN_CMD+=" --prefetch-factor ${PREFETCH_FACTOR}"
TRAIN_CMD+=" --verbose"
TRAIN_CMD+=" --save-model"
TRAIN_CMD+=" --save-dir ${RUNS_DIR}/models"
TRAIN_CMD+=" --debug"

# Add TensorBoard arguments if enabled
if [ "${TENSORBOARD}" = "true" ]; then
    TRAIN_CMD+=" --tensorboard"
    TRAIN_CMD+=" --log-dir ${RUNS_DIR}/tensorboard"
    echo "TensorBoard logging enabled" | tee -a "$LOG_FILE"
    echo "  Log directory: ${RUNS_DIR}/tensorboard" | tee -a "$LOG_FILE"
fi

# Execute the constructed command and log output
eval $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"

TRAIN_STATUS=$?

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "Training completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Training failed with status: ${TRAIN_STATUS}" | tee -a "$LOG_FILE"
fi
echo "========================================" | tee -a "$LOG_FILE"

# Copy training history if it exists
if [ -f "training_history.npy" ]; then
    cp training_history.npy "${RUNS_DIR}/histories/history_${TIMESTAMP}.npy"
    echo "Training history saved to: ${RUNS_DIR}/histories/history_${TIMESTAMP}.npy" | tee -a "$LOG_FILE"
fi

# Copy model files if they exist
if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
    cp -r models/* "${RUNS_DIR}/models/" 2>/dev/null || true
    echo "Model files copied to: ${RUNS_DIR}/models/" | tee -a "$LOG_FILE"
fi

# Create backup
BACKUP_FILE="${RUNS_DIR}/backup_${TIMESTAMP}.tar.gz"
echo "" | tee -a "$LOG_FILE"
echo "Creating backup: ${BACKUP_FILE}" | tee -a "$LOG_FILE"

# Create a manifest file with run information
MANIFEST_FILE="${RUNS_DIR}/manifest_${TIMESTAMP}.txt"
cat > "$MANIFEST_FILE" << EOF
MultiResUNet Training Run Manifest
===================================
Timestamp: ${TIMESTAMP}
Log File: ${LOG_FILE}

Configuration:
  Epochs: ${EPOCHS}
  Batch Size: ${BATCH_SIZE}
  Learning Rate: ${LEARNING_RATE}
  Data Limit: ${DATA_LIMIT}
  Validation Split: ${VALIDATION_SPLIT}
  Input Channels: ${INPUT_CHANNELS}
  Output Channels: ${OUTPUT_CHANNELS}
  Gradient Clip: ${GRADIENT_CLIP}
  Device: ${DEVICE}
  TensorBoard: ${TENSORBOARD}

Files:
  Log: logs/training_${TIMESTAMP}.log
  Models: models/
  History: histories/history_${TIMESTAMP}.npy
  TensorBoard: tensorboard/train_${TIMESTAMP}/ (if enabled)
  Backup: backup_${TIMESTAMP}.tar.gz

Training Status: $([ $TRAIN_STATUS -eq 0 ] && echo "SUCCESS" || echo "FAILED (${TRAIN_STATUS})")
EOF

echo "Manifest saved to: ${MANIFEST_FILE}" | tee -a "$LOG_FILE"

# Create compressed backup
tar -czf "${BACKUP_FILE}" \
    -C "${RUNS_DIR}" \
    "logs/training_${TIMESTAMP}.log" \
    "models" \
    "histories/history_${TIMESTAMP}.npy" \
    "manifest_${TIMESTAMP}.txt" 2>/dev/null || {
    # If tar fails (e.g., some files missing), try with available files
    echo "Some files missing, creating partial backup..." | tee -a "$LOG_FILE"
    
    TAR_FILES=()
    [ -f "${RUNS_DIR}/logs/training_${TIMESTAMP}.log" ] && TAR_FILES+=("logs/training_${TIMESTAMP}.log")
    [ -d "${RUNS_DIR}/models" ] && TAR_FILES+=("models")
    [ -f "${RUNS_DIR}/histories/history_${TIMESTAMP}.npy" ] && TAR_FILES+=("histories/history_${TIMESTAMP}.npy")
    TAR_FILES+=("manifest_${TIMESTAMP}.txt")
    
    # Add TensorBoard logs if they exist
    if [ "${TENSORBOARD}" = "true" ] && [ -d "${RUNS_DIR}/tensorboard" ]; then
        TB_DIR=$(ls -td ${RUNS_DIR}/tensorboard/train_* 2>/dev/null | head -1)
        if [ -n "$TB_DIR" ]; then
            TAR_FILES+=("${TB_DIR#${RUNS_DIR}/}")
            echo "Including TensorBoard logs in backup..." | tee -a "$LOG_FILE"
        fi
    fi
    
    tar -czf "${BACKUP_FILE}" -C "${RUNS_DIR}" "${TAR_FILES[@]}" 2>/dev/null || {
        echo "Warning: Backup creation failed" | tee -a "$LOG_FILE"
    }
}

if [ -f "${BACKUP_FILE}" ]; then
    BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    echo "Backup created successfully: ${BACKUP_FILE} (${BACKUP_SIZE})" | tee -a "$LOG_FILE"
fi

# Clean up old backups (keep last 10)
echo "" | tee -a "$LOG_FILE"
echo "Cleaning up old backups (keeping last 10)..." | tee -a "$LOG_FILE"
cd "${RUNS_DIR}"
ls -t backup_*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm
cd "${PROJECT_DIR}"

# Print summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Training Run Summary" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Log File: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "Models Directory: ${RUNS_DIR}/models/" | tee -a "$LOG_FILE"
echo "Training History: ${RUNS_DIR}/histories/history_${TIMESTAMP}.npy" | tee -a "$LOG_FILE"
echo "Backup File: ${BACKUP_FILE}" | tee -a "$LOG_FILE"
echo "Manifest: ${MANIFEST_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To view the log file:" | tee -a "$LOG_FILE"
echo "  cat ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
if [ "${TENSORBOARD}" = "true" ]; then
    echo "To view TensorBoard:" | tee -a "$LOG_FILE"
    echo "  tensorboard --logdir ${RUNS_DIR}/tensorboard" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi
echo "To extract the backup:" | tee -a "$LOG_FILE"
echo "  tar -xzf ${BACKUP_FILE} -C /your/destination/" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $TRAIN_STATUS
