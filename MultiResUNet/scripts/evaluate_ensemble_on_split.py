"""Evaluate a probability-averaged ensemble on one fixed validation/test split.

Each checkpoint keeps its own training resolution. Predictions are resized to a
shared evaluation resolution before averaging, so scaled and full-resolution
models can be combined safely.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloading import create_single_dataset
from pytorch.MultiResUNet import dice_coef, jacard, predict_prob_with_tta
from scripts.evaluate_history_on_test import load_state_dict, make_model_args, read_json
from train import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an exact checkpoint ensemble on a fixed split.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing history/config.json and models/best_model.pth. Repeat for each model.",
    )
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-json", default="runs/debug_eval/ensemble_metrics.json")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta", default="flips", choices=["none", "flips"])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5],
        help="One shared threshold or one threshold per output class.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional non-negative model weights; defaults to equal averaging.",
    )
    parser.add_argument(
        "--class-model-weights",
        action="append",
        default=[],
        help="Per-class weights for one model as comma-separated values; repeat in --run-dir order.",
    )
    parser.add_argument(
        "--inference-scales",
        type=float,
        nargs="+",
        default=None,
        help="Optional absolute input scales averaged for every model, e.g. 0.75 0.875 1.0.",
    )
    parser.add_argument(
        "--evaluation-scale",
        type=float,
        default=0.75,
        help="Resolution used for targets and final probability averaging (default: 0.75).",
    )
    return parser.parse_args()


def _construction_args():
    # Evaluation must not download ImageNet weights; the checkpoint replaces
    # every learned parameter immediately after construction.
    return SimpleNamespace(
        encoder_weights="none",
        default_model_architecture="smp_unet",
        default_encoder_name="resnet34",
        default_encoder_weights="none",
        input_channels=None,
        output_channels=None,
        scale=None,
        scale_factor=None,
    )


def _resize(tensor, size, mode):
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


def load_ensemble(run_dirs, device, weights, class_model_weights):
    members = []
    construction_args = _construction_args()
    for value in run_dirs:
        run_dir = Path(value)
        checkpoint = run_dir / "models" / "best_model.pth"
        config_path = run_dir / "history" / "config.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config: {config_path}")

        config = read_json(config_path)
        model_args = make_model_args(config, construction_args)
        model, model_name = create_model(model_args)
        model.load_state_dict(load_state_dict(checkpoint, device))
        model.to(device).eval()
        input_scale = float(config.get("scale_factor", 1.0)) if config.get("scale", False) else 1.0
        members.append({
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "model": model,
            "model_name": model_name,
            "input_scale": input_scale,
            "output_channels": int(model_args.output_channels),
        })

    output_channels = {member["output_channels"] for member in members}
    if len(output_channels) != 1:
        raise ValueError(f"Ensemble output channel mismatch: {sorted(output_channels)}")
    if weights is None:
        weights = [1.0] * len(members)
    if len(weights) != len(members) or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("--weights must contain one non-negative value per --run-dir and sum to > 0")
    if class_model_weights:
        if len(class_model_weights) != len(members):
            raise ValueError("Repeat --class-model-weights once per --run-dir")
        parsed = []
        for value in class_model_weights:
            row = [float(item) for item in value.split(",")]
            if len(row) != next(iter(output_channels)) or any(item < 0 for item in row):
                raise ValueError("Each --class-model-weights row needs one non-negative value per class")
            parsed.append(row)
        per_class_totals = [sum(row[class_idx] for row in parsed) for class_idx in range(len(parsed[0]))]
        if any(total <= 0 for total in per_class_totals):
            raise ValueError("Per-class model weights must sum to > 0 for every class")
        for member, row in zip(members, parsed):
            member["class_weights"] = [
                row[class_idx] / per_class_totals[class_idx] for class_idx in range(len(row))
            ]
            member["weight"] = None
    else:
        weight_sum = float(sum(weights))
        for member, weight in zip(members, weights):
            member["weight"] = float(weight) / weight_sum
            member["class_weights"] = None
    return members, output_channels.pop()


def evaluate(members, loader, device, output_channels, thresholds, evaluation_scale, tta, inference_scales=None):
    if len(thresholds) == 1:
        thresholds = thresholds * output_channels
    if len(thresholds) != output_channels or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("--thresholds must be one value or one value in [0,1] per output class")
    threshold_tensor = torch.tensor(thresholds, device=device).view(1, output_channels, 1, 1)

    total_dice = 0.0
    total_jaccard = 0.0
    class_intersection_sum = torch.zeros(output_channels, dtype=torch.float64)
    class_true_sum = torch.zeros(output_channels, dtype=torch.float64)
    class_pred_sum = torch.zeros(output_channels, dtype=torch.float64)
    samples = 0
    batches = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating ensemble"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            native_h, native_w = images.shape[-2:]
            eval_size = (
                max(1, int(native_h * evaluation_scale)),
                max(1, int(native_w * evaluation_scale)),
            )
            eval_targets = _resize(targets, eval_size, "nearest")
            ensemble_probs = torch.zeros(
                (images.size(0), output_channels, *eval_size), device=device, dtype=images.dtype
            )

            for member in members:
                scales = inference_scales or [member["input_scale"]]
                if any(scale <= 0 for scale in scales):
                    raise ValueError("All --inference-scales values must be > 0")
                if member["class_weights"] is None:
                    model_weights = torch.full(
                        (1, output_channels, 1, 1), member["weight"], device=device, dtype=images.dtype
                    )
                else:
                    model_weights = torch.tensor(
                        member["class_weights"], device=device, dtype=images.dtype
                    ).view(1, output_channels, 1, 1)
                for input_scale in scales:
                    input_size = (
                        max(1, int(native_h * input_scale)),
                        max(1, int(native_w * input_scale)),
                    )
                    model_images = _resize(images, input_size, "bicubic")
                    probs = predict_prob_with_tta(member["model"], model_images, tta)
                    resized_probs = _resize(probs, eval_size, "bilinear")
                    ensemble_probs += resized_probs * model_weights / len(scales)

            preds = (ensemble_probs >= threshold_tensor).float()
            batch_samples = targets.size(0)
            total_dice += dice_coef(eval_targets, preds).item() * batch_samples
            total_jaccard += jacard(eval_targets, preds).item() * batch_samples

            targets_by_class = eval_targets.reshape(batch_samples, output_channels, -1)
            preds_by_class = preds.reshape(batch_samples, output_channels, -1)
            class_intersection_sum += (targets_by_class * preds_by_class).sum(dim=(0, 2)).cpu().double()
            class_true_sum += targets_by_class.sum(dim=(0, 2)).cpu().double()
            class_pred_sum += preds_by_class.sum(dim=(0, 2)).cpu().double()
            samples += batch_samples
            batches += 1

    if samples == 0:
        raise ValueError("Evaluation dataset is empty")
    smooth = 1e-6
    class_dice = (2 * class_intersection_sum + smooth) / (class_true_sum + class_pred_sum + smooth)
    class_jaccard = (class_intersection_sum + smooth) / (
        class_true_sum + class_pred_sum - class_intersection_sum + smooth
    )
    return {
        "dice": total_dice / samples,
        "jaccard": total_jaccard / samples,
        "class_dice": class_dice.tolist(),
        "class_jaccard": class_jaccard.tolist(),
        "thresholds": thresholds,
        "evaluation_scale": evaluation_scale,
        "tta": tta,
        "inference_scales": inference_scales,
        "samples": samples,
        "batches": batches,
    }


def main():
    args = parse_args()
    if args.evaluation_scale <= 0:
        raise ValueError("--evaluation-scale must be > 0")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    members, output_channels = load_ensemble(
        args.run_dir, device, args.weights, args.class_model_weights
    )
    dataset = create_single_dataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        scale=False,
        apply_augmentation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(
        members,
        loader,
        device,
        output_channels,
        args.thresholds,
        args.evaluation_scale,
        args.tta,
        args.inference_scales,
    )
    metrics["members"] = [
        {key: value for key, value in member.items() if key != "model"}
        for member in members
    ]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main()
