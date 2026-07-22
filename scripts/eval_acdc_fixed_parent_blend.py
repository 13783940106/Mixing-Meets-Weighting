#!/usr/bin/env python3
"""Read-only ACDC 2D/ordinary-3ch-2.5D fixed parent blend evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import h5py
import numpy as np
import torch
from scipy.ndimage import zoom
from medpy import metric


ROOT = Path("/home/renguanming/jupyterlab/rgm/beta-FFT-main")
TRAIN = ROOT / "train_ACDC"
DATA_DIR = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/data")
CKPT_2D = ROOT / "runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/checkpoints/best.ckpt"
CKPT_25D = ROOT / "runs/beta_fft_acdc_25d_3ch_official_full_seed42/checkpoints/best.ckpt"
PATCH_SIZE = (256, 256)
CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only ACDC fixed parent blend evaluator.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--alpha-25d", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def load_bcp_net():
    sys.path.insert(0, str(TRAIN))
    from networks.net_factory import BCP_net  # noqa: E402

    return BCP_net


def load_model(ckpt_path: Path, branch: str, in_chns: int, device: str):
    BCP_net = load_bcp_net()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = BCP_net(in_chns=in_chns, class_num=4, ema=False).to(device)
    sd = {k.replace("module.", ""): v for k, v in ckpt[branch].items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING load {ckpt_path.name}/{branch}: missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()
    return net, ckpt


def class_metrics(pred, gt):
    rows = {}
    for c in range(1, 4):
        pred_c = pred == c
        gt_c = gt == c
        if pred_c.sum() > 0 and gt_c.sum() > 0:
            dice = float(metric.binary.dc(pred_c, gt_c))
            hd95 = float(metric.binary.hd95(pred_c, gt_c))
            asd = float(metric.binary.asd(pred_c, gt_c))
            inter = float(np.logical_and(pred_c, gt_c).sum())
            union = float(np.logical_or(pred_c, gt_c).sum())
            iou = inter / union if union else 0.0
        elif pred_c.sum() == 0 and gt_c.sum() == 0:
            dice, iou, hd95, asd = 1.0, 1.0, 0.0, 0.0
        else:
            dice, iou, hd95, asd = 0.0, 0.0, 0.0, 0.0
        rows[c] = {"dice": dice, "iou": iou, "hd95": hd95, "asd": asd}
    return rows


def eval_case(path: Path, net2d, net25d, alpha: float, device: str):
    with h5py.File(path, "r") as f:
        image = f["image"][:]
        gt = f["label"][:]
    pred = np.zeros_like(gt)
    pred2 = np.zeros_like(gt)
    pred25 = np.zeros_like(gt)
    disagreement = []
    for ind in range(image.shape[0]):
        center = image[ind]
        h, w = center.shape
        prev_i = ind - 1 if ind > 0 else ind
        next_i = ind + 1 if ind < image.shape[0] - 1 else ind
        center_s = zoom(center, (PATCH_SIZE[0] / h, PATCH_SIZE[1] / w), order=0)
        prev_s = zoom(image[prev_i], (PATCH_SIZE[0] / h, PATCH_SIZE[1] / w), order=0)
        next_s = zoom(image[next_i], (PATCH_SIZE[0] / h, PATCH_SIZE[1] / w), order=0)
        x2 = torch.from_numpy(center_s[None, None].astype(np.float32)).to(device)
        x25 = torch.from_numpy(np.stack([prev_s, center_s, next_s], axis=0)[None].astype(np.float32)).to(device)
        with torch.no_grad():
            s2 = torch.softmax(net2d(x2), dim=1)
            s25 = torch.softmax(net25d(x25), dim=1)
            sb = (1.0 - alpha) * s2 + alpha * s25
            p2 = torch.argmax(s2, dim=1).squeeze(0).cpu().numpy()
            p25 = torch.argmax(s25, dim=1).squeeze(0).cpu().numpy()
            pb = torch.argmax(sb, dim=1).squeeze(0).cpu().numpy()
        pred2[ind] = zoom(p2, (h / PATCH_SIZE[0], w / PATCH_SIZE[1]), order=0)
        pred25[ind] = zoom(p25, (h / PATCH_SIZE[0], w / PATCH_SIZE[1]), order=0)
        pred[ind] = zoom(pb, (h / PATCH_SIZE[0], w / PATCH_SIZE[1]), order=0)
        disagreement.append(float((p2 != p25).mean()))
    return pred, pred2, pred25, gt, float(np.mean(disagreement))


def aggregate(case_rows):
    out = {}
    for prefix in ["blend", "parent2d", "parent25d"]:
        out[f"{prefix}_mDice"] = float(np.mean([r[f"{prefix}_mDice"] for r in case_rows]))
        out[f"{prefix}_mIoU"] = float(np.mean([r[f"{prefix}_mIoU"] for r in case_rows]))
        out[f"{prefix}_HD95"] = float(np.mean([r[f"{prefix}_HD95"] for r in case_rows]))
        out[f"{prefix}_ASD"] = float(np.mean([r[f"{prefix}_ASD"] for r in case_rows]))
    out["positive_cases_vs_2d_mDice"] = int(sum(r["blend_mDice"] >= r["parent2d_mDice"] for r in case_rows))
    out["positive_cases_vs_25d_mDice"] = int(sum(r["blend_mDice"] >= r["parent25d_mDice"] for r in case_rows))
    return out


def main():
    args = parse_args()
    if not 0.0 <= args.alpha_25d <= 1.0:
        raise SystemExit("--alpha-25d must be in [0, 1]")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.output_dir or (ROOT / "audit" / f"acdc_fixed_parent_blend_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=False)

    net2d, ckpt2d = load_model(CKPT_2D, "model_1", 1, device)
    net25d, ckpt25d = load_model(CKPT_25D, "model_1", 3, device)
    list_file = DATA_DIR.parent / f"{args.split}.list"
    cases = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]

    case_rows = []
    class_rows = []
    for case in cases:
        pred, pred2, pred25, gt, dis = eval_case(DATA_DIR / f"{case}.h5", net2d, net25d, args.alpha_25d, device)
        metrics = {
            "blend": class_metrics(pred, gt),
            "parent2d": class_metrics(pred2, gt),
            "parent25d": class_metrics(pred25, gt),
        }
        row = {"case": case, "split": args.split, "alpha_25d": args.alpha_25d, "pred_2d_25d_disagreement": dis}
        for prefix, vals in metrics.items():
            row[f"{prefix}_mDice"] = float(np.mean([vals[c]["dice"] for c in range(1, 4)]))
            row[f"{prefix}_mIoU"] = float(np.mean([vals[c]["iou"] for c in range(1, 4)]))
            row[f"{prefix}_HD95"] = float(np.mean([vals[c]["hd95"] for c in range(1, 4)]))
            row[f"{prefix}_ASD"] = float(np.mean([vals[c]["asd"] for c in range(1, 4)]))
            for c in range(1, 4):
                class_rows.append({"case": case, "parent": prefix, "class": CLASS_NAMES[c], **vals[c]})
        case_rows.append(row)

    summary = {
        "split": args.split,
        "alpha_25d": args.alpha_25d,
        "n_cases": len(case_rows),
        **aggregate(case_rows),
        "ckpt_2d": str(CKPT_2D.relative_to(ROOT)),
        "ckpt_25d": str(CKPT_25D.relative_to(ROOT)),
        "ckpt_2d_step": ckpt2d.get("global_step"),
        "ckpt_25d_step": ckpt25d.get("global_step"),
    }
    with (out / "case_metrics.tsv").open("w", newline="") as f:
        fields = list(case_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(case_rows)
    with (out / "class_metrics.tsv").open("w", newline="") as f:
        fields = list(class_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(class_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
