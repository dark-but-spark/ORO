#!/bin/bash

# Quick UNet Diagnostics Script
# Runs essential tests and creates compressed results

echo "=== Quick UNet Diagnostics ==="
echo "Timestamp: $(date)"

# Setup
source /share/home/zjm/anaconda3/bin/activate zjm
cd /share/home/zjm/ORO/Pytorch-UNet

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="quick_diagnostics_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "Results directory: $RESULTS_DIR"

# Run essential tests
echo "1. Running data debug..."
python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/ > "$RESULTS_DIR/data_debug.log" 2>&1

echo "2. Checking sample data..."
python -c "
import torch
from utils.data_loading import BasicDataset
dataset = BasicDataset('./data/imgs/', './data/masks/', scale=0.5)
sample = dataset[0]
print('Image shape:', sample['image'].shape)
print('Mask shape:', sample['mask'].shape)
print('Image range:', sample['image'].min(), 'to', sample['image'].max())
print('Mask range:', sample['mask'].min(), 'to', sample['mask'].max())
print('Unique values:', torch.unique(sample['mask']))
" > "$RESULTS_DIR/sample_check.log" 2>&1

echo "3. Analyzing masks..."
python mask_analysis.py > "$RESULTS_DIR/mask_analysis.log" 2>&1

echo "4. Running short training test..."
python train.py --epochs 2 --batch-size 2 --learning-rate 1e-4 --subset 0.05 --validation 15 > "$RESULTS_DIR/training_test.log" 2>&1

# Collect system info
echo "5. Collecting system information..."
{
    echo "System: $(uname -a)"
    echo "Python: $(python --version)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "Images: $(find ./data/imgs/ -type f | wc -l)"
    echo "Masks: $(find ./data/masks/ -type f | wc -l)"
} > "$RESULTS_DIR/system_info.txt"

# Create summary
echo "6. Creating summary..."
{
    echo "# Quick Diagnostics Summary"
    echo "Run time: $(date)"
    echo ""
    echo "## Key Files:"
    echo "- data_debug.log: Data loading verification"
    echo "- sample_check.log: Sample data inspection" 
    echo "- mask_analysis.log: Mask format analysis"
    echo "- training_test.log: Short training validation"
    echo "- system_info.txt: Environment details"
} > "$RESULTS_DIR/README.md"

# Package results
ARCHIVE="quick_diagnostics_${TIMESTAMP}.tar.gz"
tar -czf "$ARCHIVE" "$RESULTS_DIR"

echo ""
echo "=== Quick Diagnostics Complete ==="
echo "Results: $RESULTS_DIR"
echo "Archive: $ARCHIVE"
echo ""
echo "Check the README.md in the results directory for next steps."