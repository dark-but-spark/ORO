#!/usr/bin/env python3
"""
Memory Diagnostic Tool for MultiResUNet Training
Checks system memory, GPU status, and provides optimization recommendations
"""

import os
import sys
import gc
import psutil
import subprocess

def get_system_memory():
    """Get system memory information"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print("=" * 60)
    print("SYSTEM MEMORY STATUS")
    print("=" * 60)
    print(f"Total RAM:      {mem.total / 1024**3:.1f} GB")
    print(f"Available RAM:  {mem.available / 1024**3:.1f} GB ({mem.available*100/mem.total:.1f}%)")
    print(f"Used RAM:       {mem.used / 1024**3:.1f} GB ({mem.percent:.1f}%)")
    print(f"Swap Total:     {swap.total / 1024**3:.1f} GB")
    print(f"Swap Used:      {swap.used / 1024**3:.1f} GB ({swap.percent:.1f}%)")
    print("")
    
    return mem

def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        print("=" * 60)
        print("GPU STATUS")
        print("=" * 60)
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 5:
                gpu_id, name, total, used, free = parts
                print(f"GPU {gpu_id}: {name}")
                print(f"  Total: {total} MB | Used: {used} MB | Free: {free} MB")
                print(f"  Usage: {float(used)/float(total)*100:.1f}%")
        print("")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  No NVIDIA GPU detected or nvidia-smi not available")
        print("")
        return False

def get_cpu_info():
    """Get CPU information"""
    cpu_count = os.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    print("=" * 60)
    print("CPU STATUS")
    print("=" * 60)
    print(f"CPU Cores:      {cpu_count}")
    if cpu_freq:
        print(f"CPU Frequency:  {cpu_freq.current:.0f} MHz (Max: {cpu_freq.max:.0f} MHz)")
    
    # Per-core usage
    percpu = psutil.cpu_percent(percpu=True, interval=0.5)
    print(f"Core Usage:     {percpu}")
    print(f"Overall Usage:  {psutil.cpu_percent(interval=0.5):.1f}%")
    print("")
    
    return cpu_count

def check_training_process():
    """Check if training process is running"""
    training_pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'python' in proc.info['name'].lower() and 'train.py' in cmdline:
                training_pids.append({
                    'pid': proc.info['pid'],
                    'rss_mb': proc.info['memory_info'].rss / 1024**2,
                    'cpu': proc.cpu_percent(interval=0.5),
                    'cmdline': cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print("=" * 60)
    print("TRAINING PROCESS STATUS")
    print("=" * 60)
    
    if training_pids:
        for proc in training_pids:
            print(f"PID: {proc['pid']}")
            print(f"  RSS Memory: {proc['rss_mb']:.0f} MB")
            print(f"  CPU Usage:  {proc['cpu']:.1f}%")
            print(f"  Command:    {' '.join(proc['cmdline'].split()[:5])}...")
            
            # Memory assessment
            if proc['rss_mb'] > 2000:
                print(f"  ⚠️  WARNING: High memory usage (>2GB)")
            elif proc['rss_mb'] > 1000:
                print(f"  ℹ️  INFO: Moderate memory usage")
            else:
                print(f"  ✓ Memory usage looks good")
            print("")
    else:
        print("No training process currently running")
        print("Start training with:")
        print("  bash run_training.sh --epochs 150 --data-limit 3500 \\")
        print("    --batch-size 4 --num-workers 8 --verbose")
        print("")
    
    return training_pids

def estimate_memory_requirements(data_limit=3500, batch_size=4, scale_factor=0.5):
    """Estimate memory requirements for training"""
    print("=" * 60)
    print("MEMORY REQUIREMENTS ESTIMATION")
    print("=" * 60)
    
    # Base estimates (per sample @ 640x640)
    base_mem_per_sample_mb = 11.5  # RGB image + 4-channel mask
    
    # With scale factor
    scaled_mem_per_sample = base_mem_per_sample_mb * (scale_factor ** 2)
    
    print(f"Data samples:   {data_limit}")
    print(f"Batch size:     {batch_size}")
    print(f"Scale factor:   {scale_factor}")
    print("")
    
    print("Per-sample memory:")
    print(f"  @ 640×640:  ~{base_mem_per_sample_mb:.1f} MB")
    print(f"  @ 320×320:  ~{scaled_mem_per_sample:.1f} MB (with scale={scale_factor})")
    print("")
    
    # DataLoader memory (workers * batch_size * 2 for ping-pong buffering)
    num_workers = min(8, os.cpu_count() - 2) if os.cpu_count() else 6
    prefetch_factor = 4
    
    worker_memory = num_workers * batch_size * scaled_mem_per_sample * prefetch_factor * 1.2  # 20% overhead
    
    print("Estimated memory breakdown:")
    print(f"  Data cache:        ~{data_limit * scaled_mem_per_sample:.0f} MB (if fully loaded)")
    print(f"  DataLoader workers: ~{worker_memory:.0f} MB")
    print(f"  Model + optimizer:  ~500-1000 MB")
    print(f"  GPU buffers:        ~200-500 MB")
    print("")
    
    total_estimate = worker_memory + 750 + 350  # + model + GPU
    print(f"TOTAL ESTIMATE:     ~{total_estimate/1024:.1f} GB")
    print("")
    
    # Recommendations
    available_ram = psutil.virtual_memory().available / 1024**3
    
    print("Recommendations:")
    if available_ram < total_estimate / 1024 * 1.5:
        print(f"  ⚠️  WARNING: Available RAM ({available_ram:.1f}GB) may be insufficient")
        print("  → Reduce --data-limit or enable --scale")
        print("  → Reduce --batch-size to 2")
        print("  → Reduce --num-workers to 4")
    elif available_ram < total_estimate / 1024 * 2:
        print(f"  ℹ️  INFO: Available RAM ({available_ram:.1f}GB) is adequate")
        print("  → Monitor memory usage during training")
        print("  → Consider enabling --scale as precaution")
    else:
        print(f"  ✓ Available RAM ({available_ram:.1f}GB) is sufficient")
        print("  → You can use recommended settings")
    
    print("")

def provide_optimization_recommendations():
    """Provide specific optimization recommendations"""
    print("=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    mem = get_system_memory()
    available_gb = mem.available / 1024**3
    
    recommendations = []
    
    # Check if scale should be enabled
    if available_gb < 16:
        recommendations.append(
            "🔴 CRITICAL: Enable --scale --scale-factor 0.5 (reduces memory by 75%)"
        )
    
    # Batch size recommendation
    if available_gb < 8:
        recommendations.append(
            "🔴 CRITICAL: Use --batch-size 2 (very limited RAM)"
        )
    elif available_gb < 16:
        recommendations.append(
            "🟡 WARNING: Use --batch-size 4 (moderate RAM)"
        )
    
    # Workers recommendation
    cpu_count = os.cpu_count() or 4
    if cpu_count <= 8:
        recommendations.append(
            f"🟡 WARNING: Limited CPU cores ({cpu_count}). Use --num-workers {max(1, cpu_count-2)}"
        )
    
    # General recommendations
    recommendations.append("✅ Always use --gradient-clip 1.0")
    recommendations.append("✅ Use --prefetch-factor 2-4 (balance speed vs memory)")
    recommendations.append("✅ Run ./monitor_memory.sh in background")
    recommendations.append("✅ Use screen/tmux to prevent SSH disconnect")
    
    if available_gb > 32:
        recommendations.append("✅ You have plenty of RAM - can use aggressive settings")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("")
    
    # Suggested command
    print("SUGGESTED TRAINING COMMAND:")
    print("bash run_training.sh --epochs 150 --data-limit 3500 \\")
    
    if available_gb < 16:
        print("  --batch-size 4 --num-workers 6 --prefetch-factor 2 \\")
    else:
        print("  --batch-size 8 --num-workers 8 --prefetch-factor 4 \\")
    
    print("  --gradient-clip 1.0 --save-model --tensorboard \\")
    print("  --verbose --scale --scale-factor 0.5")
    print("")

def check_oom_history():
    """Check system logs for recent OOM kills"""
    print("=" * 60)
    print("OOM KILL HISTORY")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['dmesg', '-T'],
            capture_output=True, text=True, timeout=5
        )
        
        oom_lines = [line for line in result.stdout.split('\n') 
                     if 'killed process' in line.lower() and 'python' in line.lower()]
        
        if oom_lines[-5:]:
            print("Recent OOM kills detected:")
            for line in oom_lines[-5:]:
                print(f"  {line}")
            print("")
            print("⚠️  Your system has killed Python processes due to OOM!")
            print("   Reduce memory usage immediately.")
        else:
            print("✓ No recent OOM kills detected")
        print("")
    except Exception as e:
        print(f"Could not check dmesg: {e}")
        print("")

def main():
    """Main diagnostic function"""
    print("\n" + "=" * 60)
    print("  MultiResUNet Memory Diagnostic Tool v3.1")
    print("=" * 60 + "\n")
    
    # Run all diagnostics
    get_system_memory()
    get_gpu_info()
    get_cpu_info()
    check_training_process()
    check_oom_history()
    
    # Provide recommendations
    provide_optimization_recommendations()
    
    # Estimate requirements for typical workload
    print("\n")
    estimate_memory_requirements(data_limit=3500, batch_size=4, scale_factor=0.5)
    
    print("=" * 60)
    print("Diagnostic complete.")
    print("=" * 60)
    print("\nFor real-time monitoring, run:")
    print("  ./monitor_memory.sh")
    print("\nFor training with optimal GC, use:")
    print("  bash run_training.sh --epochs 150 --data-limit 3500 \\")
    print("    --batch-size 4 --num-workers 8 --prefetch-factor 4 \\")
    print("    --gradient-clip 1.0 --save-model --tensorboard \\")
    print("    --verbose --scale --scale-factor 0.5")
    print("")

if __name__ == '__main__':
    main()
