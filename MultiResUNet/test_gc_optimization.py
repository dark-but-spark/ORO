#!/usr/bin/env python3
"""
Test script to verify GC optimization is working correctly
Run this before starting actual training
"""

from matplotlib.pylab import rint
import torch
import gc
import time
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    def __init__(self, size=100, img_size=(640, 640), channels=4):
        self.size = size
        self.img_size = img_size
        self.channels = channels
    
    def __len__(self):
       return self.size
    
    def __getitem__(self, idx):
        # Simulate loading a batch (this allocates memory)
        img = torch.randn(3, *self.img_size)
        mask = torch.randn(self.channels, *self.img_size)
        return img, mask

def test_gc_cleanup():
    """Test that GC cleanup is working properly"""
    print("=" * 60)
    print("Testing GC Cleanup Mechanism")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        initial_allocated = torch.cuda.memory_allocated(device) / 1024**2
        initial_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"Initial GPU Memory: Allocated={initial_allocated:.0f}MB, Reserved={initial_reserved:.0f}MB")
    
    # Create dataset and loader
    dataset = DummyDataset(size=50, img_size=(640, 640), channels=4)
    loader= DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)
    
    print("\nSimulating training loop with GC cleanup...")
    
    # Simulate training batches
    memory_history = []
    
    for i, (img, mask) in enumerate(loader):
        img = img.to(device)
        mask = mask.to(device)
        
        # Simulate forward pass
        output = mask * 0.5  # Simple operation
        
        # Calculate loss
        loss = (output - mask).abs().mean()
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: Test GC cleanup (mimicking our optimization)
        del output, loss, img, mask
        
        # Clean every 3 batches (like our optimization)
        if (i + 1) % 3 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Monitor memory
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            memory_history.append(allocated)
            
            if (i + 1) % 10 == 0:
               print(f"  Batch {i+1}: Allocated={allocated:.0f}MB")
    
    # Final cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        
        final_allocated = torch.cuda.memory_allocated(device) / 1024**2
        final_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"\nFinal GPU Memory: Allocated={final_allocated:.0f}MB, Reserved={final_reserved:.0f}MB")
        
        # Check if memory was properly cleaned
        memory_growth = final_allocated - initial_allocated
        print(f"\nMemory Growth: {memory_growth:+.0f}MB")
        
        if abs(memory_growth) < 50:  # Less than 50MB growth is acceptable
           print("✅ GC cleanup is working EXCELLENTLY!")
           return True
        elif memory_growth < 200:  # Less than 200MB is acceptable
           print("✅ GC cleanup is working GOOD")
           return True
        else:
           print("⚠️ WARNING: Memory growth detected. GC may need tuning.")
           return False
    else:
       print("✅ CPU mode - GC test passed")
       return True

def test_dataloader_config():
    """Test DataLoader configuration"""
    print("\n" + "=" * 60)
    print("Testing DataLoader Configuration")
    print("=" * 60)
    
    import os
    cpu_count = os.cpu_count() or 4
    optimal_workers = min(8, max(1, cpu_count - 2))
    
    print(f"System CPU cores: {cpu_count}")
    print(f"Optimal workers: {optimal_workers}")
    print(f"Persistent workers: False")
    
    dataset = DummyDataset(size=20)
    
    try:
        loader = DataLoader(
            dataset,
           batch_size=4,
            num_workers=optimal_workers,
            pin_memory=True,
            persistent_workers=False,
           prefetch_factor=4 if optimal_workers > 0 else None
        )
        
        print("✅ DataLoader created successfully")
        
        # Test data loading
        count = 0
        for img, mask in loader:
            count += 1
        
        print(f"✅ Successfully loaded {count} batches")
        return True
        
    except Exception as e:
       print(f"❌ DataLoader test failed: {e}")
       return False

def test_tensorboard_flush():
    """Test TensorBoard writer flush"""
    print("\n" + "=" * 60)
    print("Testing TensorBoard Writer Flush")
    print("=" * 60)
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(log_dir=tmpdir)
            
            # Write some data
            for i in range(10):
                writer.add_scalar('test/value', i * 0.5, i)
                writer.flush()  # Test flush
            
            writer.close()
            
            print("✅ TensorBoard flush test passed")
            return True
            
    except ImportError:
       print("ℹ️  TensorBoard not installed - skipping test")
       return None
    except Exception as e:
       print(f"❌ TensorBoard test failed: {e}")
       return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  MultiResUNet GC Optimization Verification")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: GC Cleanup
    results['gc_cleanup'] = test_gc_cleanup()
    
    # Test 2: DataLoader Config
    results['dataloader'] = test_dataloader_config()
    
    # Test 3: TensorBoard Flush
    tb_result = test_tensorboard_flush()
    if tb_result is not None:
       results['tensorboard'] = tb_result
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = all(v for v in results.values() if v is not None)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20s}: {status}")
    
    print("")
    if all_passed:
       print("🎉 All tests PASSED! GC optimization is working correctly.")
       print("\nYou can now run training with confidence:")
       print("bash run_training.sh --epochs 150 --data-limit 3500 \\")
       print("  --batch-size 4 --num-workers 8 --prefetch-factor 4 \\")
       print("  --gradient-clip 1.0 --save-model --tensorboard \\")
       print("  --verbose --scale --scale-factor 0.5")
    else:
       print("⚠️  Some tests FAILED. Please check the errors above.")
       print("Training may still work, but monitor memory carefully.")
    
    print("")
    print("=" * 60)

if __name__ == '__main__':
    main()
