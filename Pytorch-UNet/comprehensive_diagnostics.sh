#!/bin/bash

# Comprehensive UNet Training Diagnostics Script
# Performs all tests from TRAINING_TIPS.md and saves results

echo "=== UNet Training Diagnostics Started ==="
echo "Timestamp: $(date)"
echo

# Setup environment and directories
source /share/home/zjm/anaconda3/bin/activate zjm
cd /share/home/zjm/ORO/Pytorch-UNet

# Create diagnostics directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIAG_DIR="diagnostics_$TIMESTAMP"
mkdir -p "$DIAG_DIR"
mkdir -p "$DIAG_DIR/logs"
mkdir -p "$DIAG_DIR/results"
mkdir -p "$DIAG_DIR/samples"

echo "Created diagnostics directory: $DIAG_DIR"
echo

# Function to log output
log_output() {
    local test_name="$1"
    local command="$2"
    local log_file="$DIAG_DIR/logs/${test_name}.log"
    
    echo "Running: $test_name"
    echo "Command: $command"
    echo "Log file: $log_file"
    echo "---" >> "$log_file"
    echo "Test: $test_name" >> "$log_file"
    echo "Time: $(date)" >> "$log_file"
    echo "Command: $command" >> "$log_file"
    echo "---" >> "$log_file"
    
    # Execute command and capture output
    eval "$command" 2>&1 | tee -a "$log_file"
    echo "Completed: $test_name"
    echo
}

# Test 1: Data Loading Debug
echo "=== Test 1: Data Loading Debug ==="
log_output "data_debug" "python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/"

# Test 2: Sample Data Inspection
echo "=== Test 2: Sample Data Inspection ==="
log_output "sample_inspection" "python -c \"
import torch
from utils.data_loading import BasicDataset
dataset = BasicDataset('./data/imgs/', './data/masks/', scale=0.5)
sample = dataset[0]
print('Image shape:', sample['image'].shape)
print('Mask shape:', sample['mask'].shape)
print('Image range:', sample['image'].min(), 'to', sample['image'].max())
print('Mask range:', sample['mask'].min(), 'to', sample['mask'].max())
print('Unique mask values:', torch.unique(sample['mask']))
\""

# Test 3: Mask Analysis
echo "=== Test 3: Mask Channel Analysis ==="
log_output "mask_analysis" "python mask_analysis.py"

# Test 4: Small Scale Training Test
echo "=== Test 4: Small Scale Training ==="
log_output "small_training" "python train.py --epochs 3 --batch-size 2 --learning-rate 1e-4 --subset 0.05 --validation 15"

# Test 5: Different Learning Rates
echo "=== Test 5: Learning Rate Comparison ==="
log_output "lr_test_high" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-3 --subset 0.02 --validation 20"
log_output "lr_test_medium" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --subset 0.02 --validation 20"
log_output "lr_test_low" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-5 --subset 0.02 --validation 20"

# Test 6: Batch Size Testing
echo "=== Test 6: Batch Size Testing ==="
log_output "batch_test_small" "python train.py --epochs 2 --batch-size 1 --learning-rate 1e-4 --subset 0.03 --validation 15"
log_output "batch_test_medium" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --subset 0.03 --validation 15"
log_output "batch_test_large" "python train.py --epochs 2 --batch-size 4 --learning-rate 1e-4 --subset 0.03 --validation 15"

# Test 7: Scale Testing
echo "=== Test 7: Image Scale Testing ==="
log_output "scale_test_full" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --scale 1.0 --subset 0.02 --validation 20"
log_output "scale_test_half" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --scale 0.5 --subset 0.02 --validation 20"
log_output "scale_test_quarter" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --scale 0.25 --subset 0.02 --validation 20"

# Test 8: Class Count Testing
echo "=== Test 8: Class Count Testing ==="
log_output "class_test_binary" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --classes 1 --subset 0.02 --validation 20"
log_output "class_test_multi" "python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --classes 4 --subset 0.02 --validation 20"

# Test 9: TensorBoard Setup Test
echo "=== Test 9: TensorBoard Test ==="
log_output "tensorboard_test" "timeout 30 tensorboard --logdir=runs --host=localhost --port=6006 &"

# Collect system information
echo "=== System Information ==="
{
    echo "System Info:"
    uname -a
    echo
    echo "Python Version:"
    python --version
    echo
    echo "PyTorch Version:"
    python -c "import torch; print(torch.__version__)"
    echo
    echo "CUDA Available:"
    python -c "import torch; print(torch.cuda.is_available())"
    echo
    echo "GPU Count:"
    python -c "import torch; print(torch.cuda.device_count())"
    echo
    echo "Data Directory Contents:"
    ls -la ./data/
    echo
    echo "Image Files Count:"
    find ./data/imgs/ -type f | wc -l
    echo
    echo "Mask Files Count:"
    find ./data/masks/ -type f | wc -l
} > "$DIAG_DIR/system_info.txt"

# Collect training tips as reference
cp TRAINING_TIPS.md "$DIAG_DIR/"

# Create summary report
echo "=== Creating Summary Report ==="
{
    echo "# UNet Training Diagnostics Report"
    echo "Generated on: $(date)"
    echo "Directory: $DIAG_DIR"
    echo
    echo "## Test Results Summary"
    echo
    echo "### Data Tests:"
    echo "- Data loading debug: See logs/data_debug.log"
    echo "- Sample inspection: See logs/sample_inspection.log" 
    echo "- Mask analysis: See logs/mask_analysis.log"
    echo
    echo "### Training Tests:"
    echo "- Small scale training: See logs/small_training.log"
    echo "- Learning rate tests: See logs/lr_test_*.log"
    echo "- Batch size tests: See logs/batch_test_*.log"
    echo "- Scale tests: See logs/scale_test_*.log"
    echo "- Class count tests: See logs/class_test_*.log"
    echo
    echo "### System Information:"
    echo "- See system_info.txt for detailed specs"
    echo
    echo "## Recommendations:"
    echo "1. Check logs for any error messages"
    echo "2. Review sample inspection results for data quality"
    echo "3. Compare different hyperparameter test results"
    echo "4. Use the most stable configuration for full training"
} > "$DIAG_DIR/SUMMARY.md"

# Save sample outputs if they exist
echo "=== Saving Sample Outputs ==="
if [ -f "sample_input.png" ]; then
    cp sample_input.png "$DIAG_DIR/samples/"
fi
if [ -f "sample_mask_true.png" ]; then
    cp sample_mask_true.png "$DIAG_DIR/samples/"
fi
if [ -f "sample_mask_pred.png" ]; then
    cp sample_mask_pred.png "$DIAG_DIR/samples/"
fi

# Create compressed archive
echo "=== Creating Compressed Archive ==="
ARCHIVE_NAME="unet_diagnostics_${TIMESTAMP}.tar.gz"
tar -czf "$ARCHIVE_NAME" "$DIAG_DIR"

echo
echo "=== Diagnostics Complete ==="
echo "Results saved to: $DIAG_DIR"
echo "Compressed archive: $ARCHIVE_NAME"
echo "Summary report: $DIAG_DIR/SUMMARY.md"
echo
echo "Next steps:"
echo "1. Extract the archive: tar -xzf $ARCHIVE_NAME"
echo "2. Review SUMMARY.md for key findings"
echo "3. Check logs/ directory for detailed test outputs"
echo "4. Examine samples/ for visual verification"
echo
echo "Recommended configuration based on test results:"
echo "- Optimal learning rate: [determined from lr_test_* logs]"
echo "- Best batch size: [determined from batch_test_* logs]" 
echo "- Ideal scale factor: [determined from scale_test_* logs]"
echo "- Proper class count: [determined from class_test_* logs]"