#!/usr/bin/env python3
"""Read-only PROMISE12 PB2D/v2 fixed parent blend evaluator.

This script is intentionally separate from training code. It loads two frozen
parents and evaluates a fixed probability blend on validation by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import torch
from medpy import metric


ROOT = Path("/home/renguanming/jupyterlab/rgm/beta-FFT-main")
DATA = ROOT / "data_staging/promise12/20260624_141558/data_promise12"
NPY = DATA / "npy_image"
V2_WORK = ROOT / "runs/pb25d_improvement_prediction_disagreement_neighbor_regularization_v2_seed1337_20260626_172000/work"
PB2D_CKPT = ROOT / "runs/promise12_track_pb_2d_seed1337_20260624_160237/work/beta_FFT/PROMISE12_bcp_fft_test_seed_train_PROMISE12_7/self_train/model1_iter_18000_dice_0.8787.pth"
V2_CKPT = ROOT / "runs/pb25d_improvement_prediction_disagreement_neighbor_regularization_v2_seed1337_20260626_172000/work/beta_FFT/PROMISE12_bcp_fft_test_seed_pb25d_improvement_prediction_disagreement_neighbor_regularization_v2_seed1337_20260626_172000_full_7/self_train/model1_iter_29800_dice_0.88.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only fixed PB2D/v2 parent blend evaluator for PROMISE12 validation."
    )
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Split to evaluate. test is refused unless --allow-test is set.")
    parser.add_argument("--allow-test", action="store_true", help="Explicitly allow test split. Not for current P-B protocol state.")
    parser.add_argument("--alpha-v2", type=float, default=0.5, help="Probability weight for v2 parent. Default is protocol-fixed 0.5.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to audit/pb25d_v2_fixed_parent_blend_eval_<timestamp>.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Evaluation device.")
    return parser.parse_args()


def load_net_factory():
    sys.path.insert(0, str(V2_WORK))
    from networks.net_factory import net_factory  # noqa: E402

    return net_factory


def load_state(path: Path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    return state


def make_model(net_factory, in_chns: int, ckpt: Path, device: torch.device):
    model = net_factory(net_type="unet", in_chns=in_chns, class_num=2)
    model.load_state_dict(load_state(ckpt), strict=True)
    model.to(device).eval()
    return model


def binary_metrics(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, float("inf"), float("inf")
    return float(metric.binary.dc(pred, gt)), float(metric.binary.asd(pred, gt)), float(metric.binary.hd95(pred, gt))


def slice_tensor(image: np.ndarray, ind: int, mode: str, device: torch.device):
    if mode == "2d":
        arr = image[ind][None, :, :]
    else:
        prev_ind = ind - 1 if ind > 0 else ind
        next_ind = ind + 1 if ind < image.shape[0] - 1 else ind
        arr = np.stack([image[prev_ind], image[ind], image[next_ind]], axis=0)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)


def evaluate_case(pb2d_model, v2_model, image: np.ndarray, label: np.ndarray, alpha_v2: float, device: torch.device):
    pred = np.zeros_like(label, dtype=np.uint8)
    pred2 = np.zeros_like(label, dtype=np.uint8)
    pred25 = np.zeros_like(label, dtype=np.uint8)
    disagreement = []
    with torch.no_grad():
        for ind in range(image.shape[0]):
            soft2 = torch.softmax(pb2d_model(slice_tensor(image, ind, "2d", device)), dim=1)
            soft25 = torch.softmax(v2_model(slice_tensor(image, ind, "25d_3ch", device)), dim=1)
            blend = (1.0 - alpha_v2) * soft2 + alpha_v2 * soft25
            p = torch.argmax(blend, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            p2 = torch.argmax(soft2, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            p25 = torch.argmax(soft25, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred[ind] = p
            pred2[ind] = p2
            pred25[ind] = p25
            disagreement.append(float((p2 != p25).mean()))
    dsc, asd, hd95 = binary_metrics(pred, label)
    dsc2, _, _ = binary_metrics(pred2, label)
    dsc25, _, _ = binary_metrics(pred25, label)
    return {
        "dsc": dsc,
        "asd": asd,
        "hd95": hd95,
        "pb2d_dsc": dsc2,
        "v2_dsc": dsc25,
        "delta_vs_pb2d": dsc - dsc2,
        "delta_vs_v2": dsc - dsc25,
        "pred_fg_ratio": float((pred > 0).mean()),
        "pred_2d_v2_disagreement_ratio": float(np.mean(disagreement)),
    }


def write_tsv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test evaluation without --allow-test. Current P-B protocol has not approved test access.")
    if not (0.0 <= args.alpha_v2 <= 1.0):
        raise SystemExit("--alpha-v2 must be in [0, 1].")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = args.output_dir or (ROOT / "audit" / f"pb25d_v2_fixed_parent_blend_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=False)

    net_factory = load_net_factory()
    pb2d_model = make_model(net_factory, 1, PB2D_CKPT, device)
    v2_model = make_model(net_factory, 3, V2_CKPT, device)

    split_file = DATA / f"{args.split}.list"
    cases = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]
    rows = []
    for case in cases:
        image = np.load(NPY / f"{case}.npy")
        label = np.load(NPY / f"{case}_segmentation.npy")
        row = evaluate_case(pb2d_model, v2_model, image, label, args.alpha_v2, device)
        row.update({"case": case, "split": args.split, "alpha_v2": args.alpha_v2})
        rows.append(row)

    summary = {
        "split": args.split,
        "alpha_v2": args.alpha_v2,
        "n_cases": len(rows),
        "mean_dsc": float(np.mean([r["dsc"] for r in rows])),
        "mean_asd": float(np.mean([r["asd"] for r in rows])),
        "mean_hd95": float(np.mean([r["hd95"] for r in rows])),
        "positive_cases_vs_pb2d": int(sum(r["delta_vs_pb2d"] >= 0 for r in rows)),
        "positive_cases_vs_v2": int(sum(r["delta_vs_v2"] >= 0 for r in rows)),
        "pb2d_checkpoint": str(PB2D_CKPT.relative_to(ROOT)),
        "v2_checkpoint": str(V2_CKPT.relative_to(ROOT)),
        "protocol_note": "Read-only evaluator. Default alpha=0.5. Test split is refused unless --allow-test is explicit.",
    }
    write_tsv(out / "case_metrics.tsv", rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
