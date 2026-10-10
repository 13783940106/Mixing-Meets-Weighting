#!/usr/bin/env python3
"""V22.0 ACDC TEST-only evaluator for factorial arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from v14_mask_adamix_components import v14_bcp_net  # noqa: E402

CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_metrics(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    if pred.any() and truth.any():
        inter = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return (
            2.0 * inter / (int(pred.sum()) + int(truth.sum())),
            inter / union if union else 0.0,
            float(metric.binary.hd95(pred, truth)),
            float(metric.binary.asd(pred, truth)),
        )
    if not pred.any() and not truth.any():
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def infer_case(model, data_root: Path, case: str, device: str):
    with h5py.File(data_root / "data" / f"{case}.h5", "r") as handle:
        image = handle["image"][:]
        truth = handle["label"][:]
    pred_volume = np.zeros_like(truth)
    with torch.no_grad():
        for index, source in enumerate(image):
            height, width = source.shape
            resized = zoom(source, (256 / height, 256 / width), order=0)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_volume[index] = zoom(pred, (height / 256, width / 256), order=0)
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        dice, iou, hd95, asd = safe_metrics(pred_volume == class_id, truth == class_id)
        rows.append({"case": case, "class_id": class_id, "class_name": class_name, "dice": dice, "iou": iou, "hd95": hd95, "asd": asd})
    return rows


def write_tsv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {key: float(np.mean([float(row[key]) for row in rows])) for key in ("dice", "iou", "hd95", "asd")}
    for class_name in ("RV", "MYO", "LV"):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        for key in ("dice", "iou", "hd95", "asd"):
            summary[f"{class_name}_{key}"] = float(np.mean([float(row[key]) for row in class_rows]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["dry-run", "run"], default="run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads((ROOT / args.config).read_text())
    checkpoint_path = ROOT / args.checkpoint
    output_dir = ROOT / args.output_dir
    test_manifest = Path(config["test_manifest"])
    payload = {
        "task_id": config["task_id"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path.exists() else "MISSING",
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": sha256(test_manifest) if test_manifest.exists() else "MISSING",
        "mask_token_enabled": bool(config["mask_token_enabled"]),
        "boundary_unit": "pixel-unit",
        "TEST_ONLY_MODE": True,
        "training_started": False,
        "optimizer_created": False,
        "backward_called": False,
    }
    if args.mode == "dry-run":
        print(json.dumps({"status": "DRY_RUN_PASS", **payload}, indent=2))
        return

    if not checkpoint_path.exists():
        raise SystemExit(f"CHECKPOINT_MISSING:{checkpoint_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUESTED_BUT_NOT_AVAILABLE")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model = v14_bcp_net(in_chns=1, class_num=4, ema=False, mask_token_enabled=bool(config["mask_token_enabled"]), mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.to(args.device)
    model.eval()
    cases = [line.strip() for line in test_manifest.read_text().splitlines() if line.strip()]
    rows = []
    for case in cases:
        rows.extend(infer_case(model, Path(config["dataset_root"]), case, args.device))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "case_class_metrics.tsv", rows, ["case", "class_id", "class_name", "dice", "iou", "hd95", "asd"])
    summary = summarize(rows)
    write_tsv(output_dir / "class_metrics.tsv", [
        {"class_name": cls, "dice": summary[f"{cls}_dice"], "iou": summary[f"{cls}_iou"], "hd95": summary[f"{cls}_hd95"], "asd": summary[f"{cls}_asd"]}
        for cls in ("RV", "MYO", "LV")
    ], ["class_name", "dice", "iou", "hd95", "asd"])
    receipt = {"status": "TEST_PASS", "created_at": datetime.now(timezone.utc).isoformat(), **payload, "metrics": summary}
    (output_dir / "TEST_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
