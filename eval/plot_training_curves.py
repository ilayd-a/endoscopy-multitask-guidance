from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EPOCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+):\s+"
    r"Train Loss\s+(?P<train_loss>[0-9.]+)\s+\|\s+"
    r"Val Loss\s+(?P<val_loss>[0-9.]+)"
    r"(?:\s+\|\s+Val Dice\s+(?P<val_dice>[0-9.]+))?"
)


def load_training_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        required = {"epoch", "train_loss", "val_loss"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        return frame

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        row = {
            "epoch": int(match.group("epoch")),
            "train_loss": float(match.group("train_loss")),
            "val_loss": float(match.group("val_loss")),
        }
        if match.group("val_dice") is not None:
            row["val_dice"] = float(match.group("val_dice"))
        records.append(row)

    if not records:
        raise ValueError(f"No epoch records found in {path}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train/val loss curves from model training logs."
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Training Curves")
    parser.add_argument("--export-csv", type=Path, default=None)
    args = parser.parse_args()

    labels = args.labels or [path.stem for path in args.logs]
    if len(labels) != len(args.logs):
        raise ValueError("--labels must match the number of log files")

    frames = []
    has_val_dice = False
    for path, label in zip(args.logs, labels):
        frame = load_training_frame(path).copy()
        frame["label"] = label
        has_val_dice = has_val_dice or ("val_dice" in frame.columns)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if args.export_csv is not None:
        combined.to_csv(args.export_csv, index=False)
        print(f"Saved parsed log CSV -> {args.export_csv}")

    subplot_count = 2 if has_val_dice else 1
    fig, axes = plt.subplots(1, subplot_count, figsize=(7 * subplot_count, 4.5))
    if subplot_count == 1:
        axes = [axes]

    ax_loss = axes[0]
    for frame, label in zip(frames, labels):
        ax_loss.plot(frame["epoch"], frame["train_loss"], label=f"{label} train")
        ax_loss.plot(
            frame["epoch"], frame["val_loss"], linestyle="--", label=f"{label} val"
        )
    ax_loss.set_title("Train / Val Loss", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend()

    if has_val_dice:
        ax_dice = axes[1]
        for frame, label in zip(frames, labels):
            if "val_dice" not in frame.columns:
                continue
            ax_dice.plot(frame["epoch"], frame["val_dice"], label=label)
        ax_dice.set_title("Validation Dice", fontweight="bold")
        ax_dice.set_xlabel("Epoch")
        ax_dice.set_ylabel("Dice")
        ax_dice.grid(alpha=0.25)
        ax_dice.legend()

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved training curve plot -> {args.output}")


if __name__ == "__main__":
    main()
