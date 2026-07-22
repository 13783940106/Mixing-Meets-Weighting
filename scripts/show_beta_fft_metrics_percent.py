#!/usr/bin/env python3
"""Read beta-FFT raw metric logs/CSVs and display Dice/IoU as percentages.

This script is read-only: it does not modify logs, CSVs, TensorBoard files, or checkpoints.
Raw evaluator/training values are expected to remain in [0,1].
"""
import argparse
import csv
import re
from pathlib import Path

RAW_LOG_PATTERNS = {
    "model1_val_mDice": [r"model1_mean_dice\s*:\s*([0-9.]+)", r"model1_val_mDice\s*:\s*([0-9.]+)%"],
    "model1_val_mIoU": [r"model1_mean_iou\s*:\s*([0-9.]+)", r"model1_val_mIoU\s*:\s*([0-9.]+)%"],
    "model1_test_mDice": [r"model1_mean_dice_test\s*:\s*([0-9.]+)", r"model1_test_mDice\s*:\s*([0-9.]+)%"],
    "model1_test_mIoU": [r"model1_mean_iou_test\s*:\s*([0-9.]+)", r"model1_test_mIoU\s*:\s*([0-9.]+)%"],
    "model2_val_mDice": [r"model2_mean_dice\s*:\s*([0-9.]+)", r"model2_val_mDice\s*:\s*([0-9.]+)%"],
    "model2_val_mIoU": [r"model2_mean_iou\s*:\s*([0-9.]+)", r"model2_val_mIoU\s*:\s*([0-9.]+)%"],
    "model2_test_mDice": [r"model2_mean_dice_test\s*:\s*([0-9.]+)", r"model2_test_mDice\s*:\s*([0-9.]+)%"],
    "model2_test_mIoU": [r"model2_mean_iou_test\s*:\s*([0-9.]+)", r"model2_test_mIoU\s*:\s*([0-9.]+)%"],
}

CSV_IOU_FIELDS = [
    "model1_val_miou", "model1_test_miou", "model2_val_miou", "model2_test_miou",
    "model1_val_rv_iou", "model1_val_myo_iou", "model1_val_lv_iou",
    "model2_val_rv_iou", "model2_val_myo_iou", "model2_val_lv_iou",
]


def metric_to_percent(value: float) -> float:
    raw = float(value)
    if -1e-8 <= raw <= 1.0 + 1e-8:
        return raw * 100.0
    if -1e-8 <= raw <= 100.0 + 1e-8:
        # Already a percent from a display log; do not multiply again.
        return raw
    raise ValueError(f"metric value must be raw [0,1] or displayed percent [0,100], got {raw}")


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    values = {}
    for name, patterns in RAW_LOG_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                value = float(matches[-1])
                values[name] = metric_to_percent(value)
                break
    return values


def parse_dynamics(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    latest = None
    for row in reversed(rows):
        if any(row.get(field) for field in CSV_IOU_FIELDS):
            latest = row
            break
    if latest is None:
        return {}
    out = {}
    for field in CSV_IOU_FIELDS:
        value = latest.get(field, "")
        if value != "":
            out[field] = metric_to_percent(float(value))
    return out


def print_values(title, values):
    if not values:
        return
    print(title)
    for key in sorted(values):
        print(f"  {key} = {values[key]:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, help="train.log or console log with raw/display metrics")
    ap.add_argument("--dynamics", type=Path, help="training_dynamics.csv with raw IoU fields")
    args = ap.parse_args()
    if not args.log and not args.dynamics:
        ap.error("provide --log and/or --dynamics")
    if args.log:
        print_values(f"Log display metrics from {args.log}", parse_log(args.log))
    if args.dynamics:
        print_values(f"Dynamics display metrics from {args.dynamics}", parse_dynamics(args.dynamics))


if __name__ == "__main__":
    main()
