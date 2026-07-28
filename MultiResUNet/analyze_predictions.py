import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloading import create_datasets, create_single_dataset
from pytorch.MultiResUNet import predict_prob_with_tta
from train import create_model


CLASS_COLORS = np.array(
    [
        [255, 64, 64],
        [64, 220, 64],
        [64, 128, 255],
        [255, 210, 64],
        [255, 64, 255],
        [64, 255, 255],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze validation predictions, export per-image metrics and worst-case panels."
    )
    parser.add_argument("--model-path", required=True, help="Path to a saved model state_dict, e.g. runs/.../models/best_model.pth")
    parser.add_argument("--output-dir", default="runs/diagnostics/best_validation", help="Directory for CSV/JSON/images")
    parser.add_argument("--img-dir", default="data/imgs", help="Image directory used by training")
    parser.add_argument("--mask-dir", default="data/masks", help="Mask .npz directory used by training")
    parser.add_argument("--model-architecture", default="smp_unet", choices=["multiresunet", "smp_unet"])
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--output-channels", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--dataset-mode", default="validation", choices=["validation", "single"],
                        help="validation: old random validation split; single: analyze all files in img-dir/mask-dir")
    parser.add_argument("--split-name", default="validation", help="Name printed in reports, e.g. validation or test")
    parser.add_argument("--scale", action="store_true", default=True)
    parser.add_argument("--no-scale", action="store_false", dest="scale")
    parser.add_argument("--scale-factor", type=float, default=0.75)
    parser.add_argument("--data-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-tta", default="flips", choices=["none", "flips"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worst-count", type=int, default=50)
    parser.add_argument("--best-count", type=int, default=12)
    parser.add_argument("--class-worst-count", type=int, default=50)
    parser.add_argument("--class-names", nargs="*", default=None)
    return parser.parse_args()


def ensure_class_names(class_names, output_channels):
    if class_names:
        names = list(class_names)
    else:
        names = [f"class_{idx}" for idx in range(output_channels)]
    if len(names) < output_channels:
        names.extend(f"class_{idx}" for idx in range(len(names), output_channels))
    return names[:output_channels]


def build_model(args, device):
    model, model_name = create_model(args)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded {model_name} from {args.model_path}")
    return model


def make_validation_loader(args):
    if args.dataset_mode == "single":
        val_dataset = create_single_dataset(
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            limit=args.data_limit,
            scale=args.scale,
            scale_factor=args.scale_factor,
            apply_augmentation=False,
            augmentation_strength="mild",
        )
        n_val = len(val_dataset)
    else:
        train_ratio = 1.0 - args.validation_split
        _, val_dataset, _, n_val = create_datasets(
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            train_ratio=train_ratio,
            val_ratio=args.validation_split,
            limit=args.data_limit,
            scale=args.scale,
            scale_factor=args.scale_factor,
            repeat_factor=1,
            train_apply_augmentation=False,
            val_apply_augmentation=False,
            shuffle=True,
            seed=args.seed,
            augmentation_strength="mild",
        )
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return val_dataset, loader, n_val


def per_sample_metrics(target, pred, prob):
    smooth = 1e-6
    target_f = target.reshape(target.shape[0], target.shape[1], -1)
    pred_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
    prob_f = prob.reshape(prob.shape[0], prob.shape[1], -1)

    inter = (target_f * pred_f).sum(axis=2)
    true_sum = target_f.sum(axis=2)
    pred_sum = pred_f.sum(axis=2)
    union = true_sum + pred_sum - inter
    dice = (2.0 * inter + smooth) / (true_sum + pred_sum + smooth)
    jaccard = (inter + smooth) / (union + smooth)

    fp = np.clip(pred_f - target_f, 0, 1).sum(axis=2)
    fn = np.clip(target_f - pred_f, 0, 1).sum(axis=2)
    prob_mean = prob_f.mean(axis=2)

    sample_inter = inter.sum(axis=1)
    sample_true_sum = true_sum.sum(axis=1)
    sample_pred_sum = pred_sum.sum(axis=1)
    sample_union = sample_true_sum + sample_pred_sum - sample_inter
    overall_dice = (2.0 * sample_inter + smooth) / (sample_true_sum + sample_pred_sum + smooth)
    overall_jaccard = (sample_inter + smooth) / (sample_union + smooth)
    return {
        "dice": dice,
        "jaccard": jaccard,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "true_pixels": true_sum,
        "pred_pixels": pred_sum,
        "prob_mean": prob_mean,
        "overall_dice": overall_dice,
        "overall_jaccard": overall_jaccard,
        "macro_class_dice": dice.mean(axis=1),
        "macro_class_jaccard": jaccard.mean(axis=1),
    }


def normalize_image(image_chw):
    image = np.asarray(image_chw[:3], dtype=np.float32)
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def mask_to_rgb(mask_chw):
    mask = np.asarray(mask_chw, dtype=np.float32)
    h, w = mask.shape[1], mask.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(mask.shape[0]):
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        channel = mask[c] > 0.5
        rgb[channel] = np.maximum(rgb[channel], color)
    return rgb


def error_to_rgb(target_chw, pred_chw):
    fp = np.clip(pred_chw - target_chw, 0, 1).max(axis=0)
    fn = np.clip(target_chw - pred_chw, 0, 1).max(axis=0)
    rgb = np.zeros((target_chw.shape[1], target_chw.shape[2], 3), dtype=np.uint8)
    rgb[..., 0] = (fp * 255).astype(np.uint8)
    rgb[..., 1] = (fn * 255).astype(np.uint8)
    return rgb


def add_label(panel, text):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_panel(image, target, pred, prob, title):
    image_rgb = normalize_image(image)
    target_rgb = mask_to_rgb(target)
    pred_rgb = mask_to_rgb(pred)
    prob_rgb = mask_to_rgb(prob)
    error_rgb = error_to_rgb(target, pred)
    gt_overlay = cv2.addWeighted(image_rgb, 0.65, target_rgb, 0.35, 0)
    pred_overlay = cv2.addWeighted(image_rgb, 0.65, pred_rgb, 0.35, 0)

    panels = [
        add_label(image_rgb, "image"),
        add_label(target_rgb, "ground_truth"),
        add_label(pred_rgb, "prediction"),
        add_label(prob_rgb, "probability"),
        add_label(error_rgb, "error red=FP green=FN"),
        add_label(gt_overlay, "gt_overlay"),
        add_label(pred_overlay, "pred_overlay"),
    ]
    combined = np.concatenate(panels, axis=1)
    cv2.rectangle(combined, (0, 0), (combined.shape[1] - 1, combined.shape[0] - 1), (255, 255, 255), 1)
    cv2.putText(combined, title, (8, combined.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return combined


def save_ranked_panels(rows, samples, out_dir, count, prefix):
    target_dir = out_dir / prefix
    target_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows[:count], start=1):
        sample = samples[row["sample_index"]]
        panel = make_panel(
            sample["image"],
            sample["target"],
            sample["pred"],
            sample["prob"],
            f"{prefix} rank={rank} dice={row['overall_dice']:.4f} file={row['image_file']}",
        )
        out_path = target_dir / f"{rank:03d}_dice_{row['overall_dice']:.4f}_{Path(row['image_file']).stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ensure_class_names(args.class_names, args.output_channels)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    val_dataset, val_loader, n_val = make_validation_loader(args)
    print(f"{args.split_name.capitalize()} samples for analysis: {n_val}")

    rows = []
    samples = {}
    sample_offset = 0
    class_dice_values = []
    class_jaccard_values = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Analyzing validation predictions"):
            images_device = images.to(device)
            probs = predict_prob_with_tta(model, images_device, args.val_tta).detach().cpu().numpy()
            targets_np = targets.numpy()
            images_np = images.numpy()
            preds_np = (probs >= args.threshold).astype(np.float32)

            metrics = per_sample_metrics(targets_np, preds_np, probs)
            class_dice_values.append(metrics["dice"])
            class_jaccard_values.append(metrics["jaccard"])

            for batch_idx in range(images_np.shape[0]):
                dataset_idx = sample_offset + batch_idx
                image_file = val_dataset.img_files[dataset_idx]
                mask_file = val_dataset.mask_files[dataset_idx]
                row = {
                    "sample_index": dataset_idx,
                    "image_file": image_file,
                    "mask_file": mask_file,
                    "overall_dice": float(metrics["overall_dice"][batch_idx]),
                    "overall_jaccard": float(metrics["overall_jaccard"][batch_idx]),
                    "macro_class_dice": float(metrics["macro_class_dice"][batch_idx]),
                    "macro_class_jaccard": float(metrics["macro_class_jaccard"][batch_idx]),
                }
                for class_idx, class_name in enumerate(class_names):
                    row[f"{class_name}_dice"] = float(metrics["dice"][batch_idx, class_idx])
                    row[f"{class_name}_jaccard"] = float(metrics["jaccard"][batch_idx, class_idx])
                    row[f"{class_name}_fp_pixels"] = float(metrics["fp_pixels"][batch_idx, class_idx])
                    row[f"{class_name}_fn_pixels"] = float(metrics["fn_pixels"][batch_idx, class_idx])
                    row[f"{class_name}_true_pixels"] = float(metrics["true_pixels"][batch_idx, class_idx])
                    row[f"{class_name}_pred_pixels"] = float(metrics["pred_pixels"][batch_idx, class_idx])
                    row[f"{class_name}_prob_mean"] = float(metrics["prob_mean"][batch_idx, class_idx])
                rows.append(row)
                samples[dataset_idx] = {
                    "image": images_np[batch_idx],
                    "target": targets_np[batch_idx],
                    "pred": preds_np[batch_idx],
                    "prob": probs[batch_idx],
                }
            sample_offset += images_np.shape[0]

    rows_sorted = sorted(rows, key=lambda item: item["overall_dice"])
    csv_path = out_dir / "validation_error_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    class_dice = np.concatenate(class_dice_values, axis=0) if class_dice_values else np.empty((0, args.output_channels))
    class_jaccard = np.concatenate(class_jaccard_values, axis=0) if class_jaccard_values else np.empty((0, args.output_channels))
    summary = {
        "model_path": args.model_path,
        "split_name": args.split_name,
        "dataset_mode": args.dataset_mode,
        "samples": len(rows),
        "validation_samples": len(rows),
        "threshold": args.threshold,
        "val_tta": args.val_tta,
        "mean_overall_dice": float(np.mean([row["overall_dice"] for row in rows])) if rows else None,
        "mean_overall_jaccard": float(np.mean([row["overall_jaccard"] for row in rows])) if rows else None,
        "worst_overall_dice": float(rows_sorted[0]["overall_dice"]) if rows_sorted else None,
        "best_overall_dice": float(rows_sorted[-1]["overall_dice"]) if rows_sorted else None,
        "per_class": {},
    }
    for class_idx, class_name in enumerate(class_names):
        nonempty_dice = [
            float(row[f"{class_name}_dice"])
            for row in rows
            if float(row[f"{class_name}_true_pixels"]) > 0
        ]
        nonempty_jaccard = [
            float(row[f"{class_name}_jaccard"])
            for row in rows
            if float(row[f"{class_name}_true_pixels"]) > 0
        ]
        summary["per_class"][class_name] = {
            "mean_dice": float(class_dice[:, class_idx].mean()) if class_dice.size else None,
            "std_dice": float(class_dice[:, class_idx].std()) if class_dice.size else None,
            "mean_jaccard": float(class_jaccard[:, class_idx].mean()) if class_jaccard.size else None,
            "std_jaccard": float(class_jaccard[:, class_idx].std()) if class_jaccard.size else None,
            "gt_nonempty_count": len(nonempty_dice),
            "gt_nonempty_mean_dice": float(np.mean(nonempty_dice)) if nonempty_dice else None,
            "gt_nonempty_std_dice": float(np.std(nonempty_dice)) if nonempty_dice else None,
            "gt_nonempty_mean_jaccard": float(np.mean(nonempty_jaccard)) if nonempty_jaccard else None,
            "gt_nonempty_std_jaccard": float(np.std(nonempty_jaccard)) if nonempty_jaccard else None,
        }

    summary_path = out_dir / "per_class_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

    save_ranked_panels(rows_sorted, samples, out_dir, args.worst_count, "worst_cases")
    save_ranked_panels(list(reversed(rows_sorted)), samples, out_dir, args.best_count, "best_cases")
    for class_idx, class_name in enumerate(class_names):
        class_rows = [
            row for row in rows
            if float(row.get(f"{class_name}_true_pixels", 0.0)) > 0
        ]
        class_rows = sorted(class_rows, key=lambda item: item[f"{class_name}_dice"])
        if class_rows:
            save_ranked_panels(
                class_rows,
                samples,
                out_dir,
                args.class_worst_count,
                f"class_{class_idx}_{class_name}_worst",
            )

    print(f"Saved CSV report: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved worst panels: {out_dir / 'worst_cases'}")
    print(f"Saved best panels: {out_dir / 'best_cases'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
