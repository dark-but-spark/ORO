#!/usr/bin/env python3
"""Small helper to inspect training outputs: TensorBoard logs, checkpoints, and basic diagnostics.

Usage:
  python scripts/inspect_runs.py            # run diagnostics (no TB)
  python scripts/inspect_runs.py --tb       # start TensorBoard (background)
  python scripts/inspect_runs.py --list-ckpt
  python scripts/inspect_runs.py --inspect-latest

The script is conservative: it will not modify checkpoints.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import webbrowser
import time


def find_repo_root():
    return Path(__file__).resolve().parents[1]


def start_tensorboard(logdir: Path, port: int = 6006, open_browser: bool = True):
    if not logdir.exists():
        print(f'No TensorBoard logs found at {logdir}')
        return
    cmd = ['tensorboard', '--logdir', str(logdir), '--port', str(port)]
    print('Starting TensorBoard:',' '.join(cmd))
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        url = f'http://localhost:{port}'
        print(f'TensorBoard started (pid={p.pid}). URL: {url}')
        if open_browser:
            time.sleep(1)
            webbrowser.open(url)
    except FileNotFoundError:
        print('`tensorboard` not found on PATH. Install with `pip install tensorboard` or run:')
        print(f'  tensorboard --logdir {logdir} --port {port}')


def list_event_files(logdir: Path):
    if not logdir.exists():
        print(f'No logs at {logdir}')
        return
    print(f'Listing TensorBoard event files under {logdir}:')
    for p in sorted(logdir.rglob('events*')):
        print('-', p)


def list_checkpoints(ckpt_dir: Path):
    if not ckpt_dir.exists():
        print(f'No checkpoints directory at {ckpt_dir}')
        return []
    ckpts = sorted(ckpt_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        print('No .pth checkpoints found')
        return []
    print('Found checkpoints:')
    for p in ckpts:
        print(f'- {p.name}  ({p.stat().st_size//1024} KB)')
    return ckpts


def inspect_checkpoint(path: Path):
    try:
        import torch
    except Exception:
        print('torch not available; cannot inspect checkpoint tensors. Install torch to use this feature.')
        return
    print(f'Loading checkpoint {path} (read-only)')
    sd = torch.load(str(path), map_location='cpu')
    if isinstance(sd, dict):
        print('Keys in checkpoint:', list(sd.keys()))
    else:
        print('Checkpoint is not a dict; type=', type(sd))
        return

    # report mask metadata if present
    if 'mask_channels' in sd:
        print('mask_channels:', sd.get('mask_channels'))
    if 'mask_values' in sd:
        print('mask_values (length):', len(sd.get('mask_values')) if sd.get('mask_values') is not None else None)

    # Quick sanity check of weights for NaN/Inf
    n_tensors = 0
    n_bad = 0
    for k, v in sd.items():
        if hasattr(v, 'dtype'):
            # a tensor-like object
            try:
                t = v
                if not isinstance(t, torch.Tensor):
                    continue
                n_tensors += 1
                has_finite = torch.isfinite(t).all().item()
                if not has_finite:
                    n_bad += 1
                    print(f'  Non-finite values in key: {k}')
            except Exception:
                continue
    print(f'Inspected {n_tensors} tensors; {n_bad} had non-finite values')


def basic_diagnostics(root: Path):
    print('Running basic diagnostics...')
    # 1) Check logs
    logs = root / 'runs' / 'exp_local'
    list_event_files(logs)

    # 2) Check checkpoints
    ckpts = list_checkpoints(root / 'checkpoints')
    if ckpts:
        latest = ckpts[0]
        print('\nInspecting latest checkpoint:', latest.name)
        inspect_checkpoint(latest)

    # 3) Environment hints
    print('\nEnvironment hints:')
    try:
        import torch
        print('torch version:', torch.__version__)
        print('CUDA available:', torch.cuda.is_available())
    except Exception:
        print('torch not installed or failed to import')

    print('\nIf you saw NaN/Inf in weights or loss during training:')
    print('- try reducing --learning-rate and/or --batch-size')
    print('- try disabling AMP (remove --amp) or enabling it if disabled')
    print('- try smaller subset (--subset) during debugging')


def main():
    parser = argparse.ArgumentParser(description='Inspect training outputs and optionally start TensorBoard')
    parser.add_argument('--tb', action='store_true', help='Start TensorBoard (background)')
    parser.add_argument('--port', type=int, default=6006, help='Port for TensorBoard')
    parser.add_argument('--list-ckpt', action='store_true', help='List checkpoints')
    parser.add_argument('--inspect-latest', action='store_true', help='Inspect latest checkpoint (requires torch)')
    parser.add_argument('--inspect', type=str, help='Inspect specific checkpoint path')
    args = parser.parse_args()

    root = find_repo_root()

    if args.tb:
        start_tensorboard(root / 'runs' / 'exp_local', port=args.port)
    if args.list_ckpt:
        list_checkpoints(root / 'checkpoints')
    if args.inspect_latest:
        ckpts = list_checkpoints(root / 'checkpoints')
        if ckpts:
            inspect_checkpoint(ckpts[0])
    if args.inspect:
        inspect_checkpoint(Path(args.inspect))

    # default: run basic diagnostics when no flags
    if not any([args.tb, args.list_ckpt, args.inspect_latest, args.inspect]):
        basic_diagnostics(root)


if __name__ == '__main__':
    main()
