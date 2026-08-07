"""Mine the hardest training images for one segmentation class."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloading import create_single_dataset
from pytorch.MultiResUNet import predict_prob_with_tta
from scripts.evaluate_history_on_test import load_state_dict, make_model_args, read_json
from train import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Build a hard-sample JSON manifest from one checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--class-index", type=int, default=2)
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--tta", choices=["none", "flips"], default="flips")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")
    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "history" / "config.json")
    construction = SimpleNamespace(
        encoder_weights="none", default_model_architecture="smp_unet",
        default_encoder_name="resnet34", default_encoder_weights="none",
        input_channels=None, output_channels=None, scale=None, scale_factor=None,
    )
    model_args = make_model_args(config, construction)
    if not 0 <= args.class_index < model_args.output_channels:
        raise ValueError("--class-index is outside the checkpoint output range")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, model_name = create_model(model_args)
    model.load_state_dict(load_state_dict(run_dir / "models" / "best_model.pth", device))
    model.to(device).eval()

    scale = bool(config.get("scale", False))
    scale_factor = float(config.get("scale_factor", 1.0))
    dataset = create_single_dataset(
        args.img_dir, args.mask_dir, scale=scale, scale_factor=scale_factor,
        apply_augmentation=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    scores = []
    smooth = 1e-6
    with torch.no_grad():
        for sample_idx, (images, targets) in enumerate(tqdm(loader, desc="Mining hard samples")):
            images = images.to(device)
            target = targets[:, args.class_index:args.class_index + 1].to(device)
            probs = predict_prob_with_tta(model, images, args.tta)
            pred = (probs[:, args.class_index:args.class_index + 1] >= args.threshold).float()
            intersection = (target * pred).sum().item()
            true_sum = target.sum().item()
            pred_sum = pred.sum().item()
            dice = (2 * intersection + smooth) / (true_sum + pred_sum + smooth)
            false_negative = (target * (1 - pred)).sum().item()
            false_positive = ((1 - target) * pred).sum().item()
            fn_rate = false_negative / max(true_sum, 1.0)
            fp_rate = false_positive / max(pred_sum, 1.0)
            hardness = (1 - dice) + 0.25 * fn_rate + 0.10 * fp_rate
            image_file = dataset.img_files[sample_idx]
            scores.append({
                "image_file": image_file,
                "stem": Path(image_file).stem,
                "class_dice": dice,
                "fn_rate": fn_rate,
                "fp_rate": fp_rate,
                "hardness": hardness,
            })

    scores.sort(key=lambda item: item["hardness"], reverse=True)
    selected_count = max(1, int(round(len(scores) * args.top_fraction)))
    selected = scores[:selected_count]
    output = {
        "source_run_dir": str(run_dir),
        "model_name": model_name,
        "class_index": args.class_index,
        "threshold": args.threshold,
        "tta": args.tta,
        "top_fraction": args.top_fraction,
        "selected_count": selected_count,
        "total_count": len(scores),
        "selected_stems": [item["stem"] for item in selected],
        "scores": scores,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"Selected {selected_count}/{len(scores)} hard samples")
    print(f"Saved manifest: {output_path}")


if __name__ == "__main__":
    main()
