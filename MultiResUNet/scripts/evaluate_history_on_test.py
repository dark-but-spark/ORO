import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloading import create_single_dataset
from pytorch.MultiResUNet import dice_coef, jacard, per_class_segmentation_metrics, predict_prob_with_tta
from train import create_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate historical best_model.pth checkpoints on one fixed test split."
    )
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=["runs", "runsTemp/runsABCtest/logs"],
        help="Directories recursively scanned for models/best_model.pth.",
    )
    parser.add_argument("--test-img-dir", default="data/test/images")
    parser.add_argument("--test-mask-dir", default="data/test/masks")
    parser.add_argument("--output-csv", default="runs/history_test_evaluation.csv")
    parser.add_argument("--output-json", default="runs/history_test_evaluation.json")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", default="none", choices=["none", "flips"])
    parser.add_argument("--metric-ignore-classes", type=int, nargs="*", default=[2])
    parser.add_argument("--scale", action="store_true", default=None, help="Override checkpoint config scale=True.")
    parser.add_argument("--no-scale", action="store_false", dest="scale", help="Override checkpoint config scale=False.")
    parser.add_argument("--scale-factor", type=float, default=None, help="Override checkpoint config scale_factor.")
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--output-channels", type=int, default=None)
    parser.add_argument("--default-model-architecture", default="smp_unet", choices=["multiresunet", "smp_unet"])
    parser.add_argument("--default-encoder-name", default="resnet34")
    parser.add_argument("--default-encoder-weights", default="imagenet")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered checkpoints.")
    return parser.parse_args()


def read_json(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def discover_checkpoints(run_roots):
    checkpoints = []
    seen = set()
    for root in run_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for ckpt in root_path.rglob("best_model.pth"):
            resolved = ckpt.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dir = ckpt.parent.parent if ckpt.parent.name == "models" else ckpt.parent
            checkpoints.append((run_dir, ckpt))
    return sorted(checkpoints, key=lambda item: str(item[0]))


def config_value(config, args, key, default):
    override = getattr(args, key, None)
    if override is not None:
        return override
    value = config.get(key, default)
    if value == -1:
        return default
    return value


def make_model_args(config, args):
    return SimpleNamespace(
        model_architecture=config.get("model_architecture", args.default_model_architecture),
        encoder_name=config.get("encoder_name", args.default_encoder_name),
        encoder_weights=config.get("encoder_weights", args.default_encoder_weights),
        input_channels=int(config_value(config, args, "input_channels", 3)),
        output_channels=int(config_value(config, args, "output_channels", 4)),
        dropout_rate=float(config.get("dropout_rate", 0.2)),
    )


def load_state_dict(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def evaluate(model, loader, device, output_channels, threshold=0.5, tta="none", ignore_classes=None):
    model.eval()
    ignore_classes = sorted({int(idx) for idx in (ignore_classes or []) if 0 <= int(idx) < output_channels})
    keep_classes = [idx for idx in range(output_channels) if idx not in ignore_classes]

    totals = {
        "dice": 0.0,
        "jaccard": 0.0,
        "dice_ignore_classes": 0.0,
        "jaccard_ignore_classes": 0.0,
    }
    class_dice_sum = None
    class_jaccard_sum = None
    batches = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            probs = predict_prob_with_tta(model, images, tta)
            preds = (probs >= threshold).float()

            totals["dice"] += dice_coef(targets, preds).item()
            totals["jaccard"] += jacard(targets, preds).item()
            class_dice, class_jaccard = per_class_segmentation_metrics(targets, preds)
            class_dice_sum = class_dice if class_dice_sum is None else class_dice_sum + class_dice
            class_jaccard_sum = class_jaccard if class_jaccard_sum is None else class_jaccard_sum + class_jaccard

            if keep_classes and len(keep_classes) != output_channels:
                totals["dice_ignore_classes"] += dice_coef(targets[:, keep_classes], preds[:, keep_classes]).item()
                totals["jaccard_ignore_classes"] += jacard(targets[:, keep_classes], preds[:, keep_classes]).item()
            batches += 1

    if batches == 0:
        raise ValueError("Test loader is empty")

    result = {
        "test_dice": totals["dice"] / batches,
        "test_jaccard": totals["jaccard"] / batches,
        "test_class_dice": (class_dice_sum / batches).tolist(),
        "test_class_jaccard": (class_jaccard_sum / batches).tolist(),
        "test_batches": batches,
    }
    if keep_classes and len(keep_classes) != output_channels:
        result["metric_ignore_classes"] = ignore_classes
        result["test_dice_ignore_classes"] = totals["dice_ignore_classes"] / batches
        result["test_jaccard_ignore_classes"] = totals["jaccard_ignore_classes"] / batches
    return result


def evaluate_checkpoint(run_dir, ckpt, dataset_cache, args, device):
    config = read_json(run_dir / "history" / "config.json")
    summary = read_json(run_dir / "history" / "summary.json")
    existing_test = read_json(run_dir / "history" / "test_metrics.json")
    model_args = make_model_args(config, args)

    scale = config_value(config, args, "scale", True)
    scale_factor = float(config_value(config, args, "scale_factor", 0.75))
    cache_key = (scale, scale_factor)
    if cache_key not in dataset_cache:
        dataset_cache[cache_key] = create_single_dataset(
            img_dir=args.test_img_dir,
            mask_dir=args.test_mask_dir,
            scale=scale,
            scale_factor=scale_factor,
            apply_augmentation=False,
        )
    dataset = dataset_cache[cache_key]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, model_name = create_model(model_args)
    model.load_state_dict(load_state_dict(ckpt, device))
    model.to(device)

    metrics = evaluate(
        model,
        loader,
        device,
        output_channels=model_args.output_channels,
        threshold=args.threshold,
        tta=args.tta,
        ignore_classes=args.metric_ignore_classes,
    )

    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "status": "ok",
        "model_name": model_name,
        "model_architecture": model_args.model_architecture,
        "encoder_name": getattr(model_args, "encoder_name", None),
        "encoder_weights": getattr(model_args, "encoder_weights", None),
        "scale": scale,
        "scale_factor": scale_factor,
        "threshold": args.threshold,
        "tta": args.tta,
        "test_samples": len(dataset),
        "best_val_dice": summary.get("best_val_dice"),
        "best_epoch": summary.get("best_epoch"),
        "class_weights": config.get("class_weights"),
        "oversample_factor": config.get("oversample_factor"),
        "previous_test_dice": existing_test.get("dice"),
        "previous_test_dice_ignore_classes": existing_test.get("dice_ignore_classes"),
    }
    row.update(metrics)
    return row


def write_outputs(rows, output_csv, output_json):
    output_csv = Path(output_csv)
    output_json = Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "run_name",
        "status",
        "test_dice",
        "test_jaccard",
        "test_dice_ignore_classes",
        "best_val_dice",
        "best_epoch",
        "previous_test_dice",
        "model_architecture",
        "encoder_name",
        "scale_factor",
        "class_weights",
        "oversample_factor",
        "checkpoint",
        "error",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoints = discover_checkpoints(args.run_roots)
    print(f"Discovered checkpoints: {len(checkpoints)}")
    for run_dir, ckpt in checkpoints:
        print(f"  {run_dir.name}: {ckpt}")

    if args.dry_run:
        return

    rows = []
    dataset_cache = {}
    for run_dir, ckpt in checkpoints:
        print(f"\nEvaluating {run_dir.name}")
        try:
            row = evaluate_checkpoint(run_dir, ckpt, dataset_cache, args, device)
            print(f"  test_dice={row['test_dice']:.4f}, test_jaccard={row['test_jaccard']:.4f}")
            if "test_dice_ignore_classes" in row:
                print(f"  test_dice_ignore_classes={row['test_dice_ignore_classes']:.4f}")
        except Exception as exc:
            row = {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt),
                "status": "error",
                "error": repr(exc),
            }
            print(f"  ERROR: {exc}")
        rows.append(row)

    rows.sort(key=lambda item: item.get("test_dice") if isinstance(item.get("test_dice"), float) else -1, reverse=True)
    write_outputs(rows, args.output_csv, args.output_json)


if __name__ == "__main__":
    main()
