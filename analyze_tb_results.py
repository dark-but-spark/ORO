from pathlib import Path
import math
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_history(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.4f}"


def summarize_history(name: str, hist_path: Path) -> dict:
    hist = load_history(hist_path)
    val_dice = list(hist.get("val_dice", []))
    val_jaccard = list(hist.get("val_jaccard", []))
    train_dice = list(hist.get("dice", hist.get("train_dice", [])))
    train_jaccard = list(hist.get("jaccard", hist.get("train_jaccard", [])))
    loss = list(hist.get("loss", hist.get("train_loss", [])))
    val_loss = list(hist.get("val_loss", []))

    best_idx = max(range(len(val_dice)), key=lambda i: val_dice[i]) if val_dice else None
    last_idx = len(val_dice) - 1 if val_dice else None
    return {
        "name": name,
        "epochs": max((len(v) for v in hist.values() if hasattr(v, "__len__")), default=0),
        "best_epoch": None if best_idx is None else best_idx + 1,
        "best_val_dice": None if best_idx is None else val_dice[best_idx],
        "best_val_jaccard": None if best_idx is None or not val_jaccard else val_jaccard[best_idx],
        "final_val_dice": None if last_idx is None else val_dice[last_idx],
        "final_val_jaccard": None if last_idx is None or not val_jaccard else val_jaccard[last_idx],
        "final_train_dice": None if not train_dice else train_dice[-1],
        "final_train_jaccard": None if not train_jaccard else train_jaccard[-1],
        "final_loss": None if not loss else loss[-1],
        "final_val_loss": None if not val_loss else val_loss[-1],
        "gap_final_dice": None
        if not train_dice or last_idx is None
        else train_dice[-1] - val_dice[last_idx],
    }


def print_table(title: str, rows: list[dict]):
    print(f"\n{title}")
    print(
        "name,epochs,best_epoch,best_val_dice,best_val_jaccard,"
        "final_val_dice,final_val_jaccard,final_train_dice,gap_final_dice"
    )
    for r in rows:
        print(
            f"{r['name']},{r['epochs']},{r['best_epoch']},"
            f"{fmt(r['best_val_dice'])},{fmt(r['best_val_jaccard'])},"
            f"{fmt(r['final_val_dice'])},{fmt(r['final_val_jaccard'])},"
            f"{fmt(r['final_train_dice'])},{fmt(r['gap_final_dice'])}"
        )


def load_event_run(run_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    scalars = {}
    for event_path in sorted(run_dir.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            scalars.setdefault(tag, [])
            for item in acc.Scalars(tag):
                scalars[tag].append((item.step, item.wall_time, item.value))

    deduped = {}
    for tag, items in scalars.items():
        latest_by_step = {}
        for step, wall_time, value in items:
            old = latest_by_step.get(step)
            if old is None or wall_time >= old[0]:
                latest_by_step[step] = (wall_time, value)
        deduped[tag] = [(step, wall, value) for step, (wall, value) in sorted(latest_by_step.items())]
    return deduped


def summarize_events(name: str, run_dir: Path) -> dict:
    scalars = load_event_run(run_dir)

    def values(tag):
        return [v for _, _, v in scalars.get(tag, [])]

    val_dice = values("Metrics/val_dice")
    val_jaccard = values("Metrics/val_jaccard")
    train_dice = values("Metrics/dice")
    train_jaccard = values("Metrics/jaccard")
    loss = values("Loss/train")
    val_loss = values("Loss/validation")

    best_idx = max(range(len(val_dice)), key=lambda i: val_dice[i]) if val_dice else None
    last_idx = len(val_dice) - 1 if val_dice else None
    return {
        "name": name,
        "epochs": len(val_dice),
        "best_epoch": None if best_idx is None else best_idx + 1,
        "best_val_dice": None if best_idx is None else val_dice[best_idx],
        "best_val_jaccard": None if best_idx is None or not val_jaccard else val_jaccard[best_idx],
        "final_val_dice": None if last_idx is None else val_dice[last_idx],
        "final_val_jaccard": None if last_idx is None or not val_jaccard else val_jaccard[last_idx],
        "final_train_dice": None if not train_dice else train_dice[-1],
        "final_train_jaccard": None if not train_jaccard else train_jaccard[-1],
        "final_loss": None if not loss else loss[-1],
        "final_val_loss": None if not val_loss else val_loss[-1],
        "gap_final_dice": None
        if not train_dice or last_idx is None
        else train_dice[-1] - val_dice[last_idx],
    }


def main():
    print("EVENT_SCALARS")
    for root in [Path("MultiResUNet/ABtest"), Path("MultiResUNet/kfold")]:
        run_events = {}
        for event_path in sorted(root.rglob("events.out.tfevents.*")):
            run_name = event_path.relative_to(root).parts[0]
            try:
                acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
                acc.Reload()
                tags = acc.Tags().get("scalars", [])
                if tags:
                    run_events.setdefault(run_name, {})
                    for tag in tags:
                        run_events[run_name][tag] = run_events[run_name].get(tag, 0) + len(acc.Scalars(tag))
            except Exception as exc:
                print(f"{root.name},{run_name},{event_path.name},ERROR,{type(exc).__name__}:{exc}")
        for run_name, tags in sorted(run_events.items()):
            print(f"{root.name},{run_name},{tags}")

    ab_rows = []
    for p in sorted(Path("MultiResUNet/ABtest").glob("*/training_history.npy")):
        ab_rows.append(summarize_history(p.parent.name, p))
    ab_rows.sort(key=lambda r: (r["best_val_dice"] or -1), reverse=True)
    print_table("ABTEST", ab_rows)

    kfold_rows = []
    for p in sorted(Path("MultiResUNet/kfold").glob("*/training_history.npy")):
        kfold_rows.append(summarize_history(p.parent.name, p))
    kfold_rows.sort(key=lambda r: r["name"])
    print_table("KFOLD", kfold_rows)

    vals = np.array([r["best_val_dice"] for r in kfold_rows if r["best_val_dice"] is not None])
    ious = np.array([r["best_val_jaccard"] for r in kfold_rows if r["best_val_jaccard"] is not None])
    finals = np.array([r["final_val_dice"] for r in kfold_rows if r["final_val_dice"] is not None])
    if len(vals):
        print("\nKFOLD_AGG")
        print(f"best_val_dice_mean,{vals.mean():.4f}")
        print(f"best_val_dice_std,{vals.std(ddof=1) if len(vals) > 1 else 0:.4f}")
        print(f"best_val_jaccard_mean,{ious.mean():.4f}")
        print(f"best_val_jaccard_std,{ious.std(ddof=1) if len(ious) > 1 else 0:.4f}")
        print(f"final_val_dice_mean,{finals.mean():.4f}")
        print(f"fold_count,{len(vals)}")

    event_ab_rows = []
    for run_dir in sorted(p for p in Path("MultiResUNet/ABtest").iterdir() if p.is_dir()):
        event_ab_rows.append(summarize_events(run_dir.name, run_dir))
    event_ab_rows.sort(key=lambda r: (r["best_val_dice"] or -1), reverse=True)
    print_table("EVENT_ABTEST_DEDUPED", event_ab_rows)

    event_kfold_rows = []
    for run_dir in sorted(p for p in Path("MultiResUNet/kfold").iterdir() if p.is_dir()):
        event_kfold_rows.append(summarize_events(run_dir.name, run_dir))
    event_kfold_rows.sort(key=lambda r: r["name"])
    print_table("EVENT_KFOLD_DEDUPED", event_kfold_rows)

    event_vals = np.array([r["best_val_dice"] for r in event_kfold_rows if r["best_val_dice"] is not None])
    event_ious = np.array([r["best_val_jaccard"] for r in event_kfold_rows if r["best_val_jaccard"] is not None])
    event_finals = np.array([r["final_val_dice"] for r in event_kfold_rows if r["final_val_dice"] is not None])
    if len(event_vals):
        print("\nEVENT_KFOLD_AGG")
        print(f"best_val_dice_mean,{event_vals.mean():.4f}")
        print(f"best_val_dice_std,{event_vals.std(ddof=1) if len(event_vals) > 1 else 0:.4f}")
        print(f"best_val_jaccard_mean,{event_ious.mean():.4f}")
        print(f"best_val_jaccard_std,{event_ious.std(ddof=1) if len(event_ious) > 1 else 0:.4f}")
        print(f"final_val_dice_mean,{event_finals.mean():.4f}")
        print(f"fold_count,{len(event_vals)}")


if __name__ == "__main__":
    main()
