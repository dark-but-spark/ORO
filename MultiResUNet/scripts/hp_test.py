#!/usr/bin/env python3
"""Hyperparameter smoke-test runner

Usage:
  python scripts/hp_test.py [SCENARIO]

If no SCENARIO is provided the script will prompt interactively.
Scenarios are integer keys that select small test configurations.
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
RUNS_DIR = os.path.join(PROJECT_DIR, 'runs', 'test_hp')
os.makedirs(RUNS_DIR, exist_ok=True)

SCENARIOS = {
    1: {'name': 'baseline', 'epochs': 2, 'batch_size': 2, 'lr': 1e-4, 'data_limit': 8, 'scale': False},
    2: {'name': 'bigger_batch', 'epochs': 2, 'batch_size': 4, 'lr': 1e-4, 'data_limit': 8, 'scale': False},
    3: {'name': 'scale_on', 'epochs': 2, 'batch_size': 2, 'lr': 1e-4, 'data_limit': 8, 'scale': True, 'scale_factor': 0.5},
    4: {'name': 'high_lr', 'epochs': 2, 'batch_size': 2, 'lr': 1e-3, 'data_limit': 8, 'scale': False},
    5: {'name': 'sweep_lr', 'epochs': 2, 'batch_size': 2, 'lr': 5e-5, 'data_limit': 8, 'scale': False},
}

def build_cmd(cfg, log_dir):
    cmd = [sys.executable, '-u', os.path.join(PROJECT_DIR, 'train.py')]
    cmd += ['--epochs', str(cfg.get('epochs', 2))]
    cmd += ['--batch-size', str(cfg.get('batch_size', 2))]
    cmd += ['--learning-rate', str(cfg.get('lr', 1e-4))]
    cmd += ['--data-limit', str(cfg.get('data_limit', 8))]
    cmd += ['--validation-split', '0.1']
    if cfg.get('scale'):
        cmd += ['--scale', '--scale-factor', str(cfg.get('scale_factor', 0.5))]
    cmd += ['--tensorboard', '--log-dir', log_dir]
    cmd += ['--save-model', '--verbose', '--debug']
    return cmd

def run_scenario(idx, cfg):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"sc{idx}_{cfg.get('name','run')}_{ts}"
    log_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(log_dir, exist_ok=True)

    cmd = build_cmd(cfg, log_dir)
    print(f"Running scenario {idx}: {cfg.get('name')} -> log: {log_dir}")
    print('Command:', ' '.join(cmd))

    stdout_file = os.path.join(log_dir, 'stdout.log')
    stderr_file = os.path.join(log_dir, 'stderr.log')

    start = time.time()
    try:
        with open(stdout_file, 'wb') as out, open(stderr_file, 'wb') as err:
            proc = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=out, stderr=err, timeout=300)
        status = proc.returncode
    except subprocess.TimeoutExpired:
        status = -1
        with open(stderr_file, 'a') as err:
            err.write('\nTIMEOUT\n')
    duration = time.time() - start

    # Try to load training_history.npy from PROJECT_DIR
    history_path = os.path.join(PROJECT_DIR, 'training_history.npy')
    metrics = {}
    if os.path.exists(history_path):
        try:
            import numpy as np
            hist = np.load(history_path, allow_pickle=True).item()
            metrics['last_val_dice'] = float(hist.get('val_dice')[-1]) if hist.get('val_dice') else None
            metrics['last_val_jaccard'] = float(hist.get('val_jaccard')[-1]) if hist.get('val_jaccard') else None
            # move history into run dir for bookkeeping
            dst = os.path.join(log_dir, 'training_history.npy')
            os.replace(history_path, dst)
        except Exception:
            pass
    else:
        # try to extract final metrics from stdout
        try:
            with open(stdout_file, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            for line in txt.splitlines()[::-1]:
                if 'Final Dice:' in line:
                    try:
                        metrics['final_dice'] = float(line.split(':')[-1].strip())
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    summary = {
        'scenario': idx,
        'config': cfg,
        'log_dir': log_dir,
        'status': status,
        'duration_s': duration,
        'metrics': metrics,
    }
    with open(os.path.join(log_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('Done — status', status, 'duration', int(duration), 's')
    if metrics:
        print('Metrics:', metrics)
    print('Logs at', log_dir)
    return summary


def main():
    arg = None
    if len(sys.argv) > 1:
        try:
            arg = int(sys.argv[1])
        except Exception:
            arg = sys.argv[1]

    if arg is None:
        print('Available scenarios:')
        for k, v in SCENARIOS.items():
            print(f"  {k}: {v['name']} (bs={v['batch_size']}, lr={v['lr']}, scale={v.get('scale',False)})")
        s = input('Enter scenario number (or "all"): ').strip()
        if s == 'all':
            choices = list(SCENARIOS.keys())
        else:
            choices = [int(s)]
    elif arg == 'all' or arg == 'All':
        choices = list(SCENARIOS.keys())
    else:
        choices = [int(arg)]

    results = []
    for idx in choices:
        cfg = SCENARIOS.get(idx)
        if cfg is None:
            print('Unknown scenario', idx)
            continue
        res = run_scenario(idx, cfg)
        results.append(res)

    # write aggregated summary
    agg_file = os.path.join(RUNS_DIR, 'aggregate_results.json')
    with open(agg_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\nAll done. Aggregate results written to', agg_file)


if __name__ == '__main__':
    main()
