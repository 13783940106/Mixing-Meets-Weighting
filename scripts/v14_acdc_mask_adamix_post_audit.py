#!/usr/bin/env python3
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from networks.net_factory import BCP_net as parent_bcp_net
from v14_mask_adamix_components import v14_bcp_net


RUN = ROOT / "runs" / "beta_fft_acdc_10pct_mask_token_adamix_full_seed42"
DATA = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
AUDIT = ROOT / "audit"
PARENT_CASES = AUDIT / "beta_fft_acdc_full_case_metrics.csv"
PARENT = {
    "best_val_mDice": 0.9009226236805246,
    "best_val_mIoU": 0.8271301955872562,
    "best_val_HD95": 2.180820531326562,
    "best_val_ASD": 0.49881470603716577,
    "final_val_mDice": 0.8900385427769114,
    "RV_Dice": 0.8782994042927565,
    "MYO_Dice": 0.9011183049564909,
    "LV_Dice": 0.9233501617923265,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_metrics(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    pred_sum, truth_sum = int(pred.sum()), int(truth.sum())
    if pred_sum and truth_sum:
        inter = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return 2 * inter / (pred_sum + truth_sum), inter / union, float(metric.binary.hd95(pred, truth)), float(metric.binary.asd(pred, truth))
    if pred_sum == 0 and truth_sum == 0:
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def infer_case(model, case):
    import h5py
    with h5py.File(DATA / "data" / f"{case}.h5", "r") as handle:
        image = handle["image"][:]
        truth = handle["label"][:]
    prediction = np.zeros_like(truth)
    model.eval()
    with torch.no_grad():
        for index in range(image.shape[0]):
            source = image[index]
            height, width = source.shape
            resized = zoom(source, (256 / height, 256 / width), order=0)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()
            output = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
            prediction[index] = zoom(output, (height / 256, width / 256), order=0)
    rows = []
    for class_id, name in ((1, "RV"), (2, "MYO"), (3, "LV")):
        dice, iou, hd95, asd = safe_metrics(prediction == class_id, truth == class_id)
        rows.append({"case": case, "class_id": class_id, "class_name": name, "dice": dice, "iou": iou, "hd95": hd95, "asd": asd})
    return rows


def evaluate(checkpoint_path, split_name):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = v14_bcp_net(in_chns=1, class_num=4, ema=False, mask_token_enabled=True, mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(checkpoint["model_1"])
    cases = [line.strip() for line in (DATA / f"{split_name}.list").read_text().splitlines() if line.strip()]
    rows = [row for case in cases for row in infer_case(model, case)]
    summary = {metric_name: float(np.mean([row[metric_name] for row in rows])) for metric_name in ("dice", "iou", "hd95", "asd")}
    summary.update({f"{name}_Dice": float(np.mean([row["dice"] for row in rows if row["class_name"] == name])) for name in ("RV", "MYO", "LV")})
    del model
    torch.cuda.empty_cache()
    return checkpoint, rows, summary


def parent_case_means():
    grouped = {}
    with PARENT_CASES.open() as handle:
        for row in csv.DictReader(handle):
            if row["checkpoint"] == "best" and row["branch"] == "model_1" and row["split"] == "val" and row["postprocess"] == "raw":
                grouped.setdefault(row["case"], {})[row["class_name"]] = {key: float(row[key]) for key in ("dice", "jaccard_iou", "hd95", "asd")}
    return grouped


def parse_endpoints():
    mask_rows, mix_rows = [], []
    for line in (RUN / "train.log").read_text().splitlines():
        if "mask_token_endpoint=" in line:
            mask_rows.append(json.loads(line.split("mask_token_endpoint=", 1)[1]))
        if "adamix_endpoint=" in line:
            mix_rows.append(json.loads(line.split("adamix_endpoint=", 1)[1]))
    mask_pass = bool(mask_rows) and any(abs(row["model1"].get("gamma", 0)) > 0 and row["model1"].get("token_gradient_norm", 0) > 0 for row in mask_rows) and all(row["model1"].get("finite") for row in mask_rows)
    mix_pass = bool(mix_rows) and any(row.get("patch_area_ratio", 0) > 0 and row.get("mixed_loss_contribution", 0) > 0 and row.get("model1_total_gradient_norm", 0) > 0 for row in mix_rows) and all(row.get("finite") for row in mix_rows)
    return {"mask_token": {"records": len(mask_rows), "pass": mask_pass, "last": mask_rows[-1] if mask_rows else None}, "adamix": {"records": len(mix_rows), "pass": mix_pass, "last": mix_rows[-1] if mix_rows else None}, "pass": mask_pass and mix_pass}


def main():
    best_path, last_path = RUN / "checkpoints" / "best.ckpt", RUN / "checkpoints" / "last.ckpt"
    best_ckpt, val_rows, best_val = evaluate(best_path, "val")
    last_ckpt, _, final_val = evaluate(last_path, "val")
    # Formal test is touched only now, after validation-best checkpoint/branch is frozen.
    _, test_rows, test_summary = evaluate(best_path, "test")

    parent_cases = parent_case_means()
    candidate_by_case = {}
    for row in val_rows:
        candidate_by_case.setdefault(row["case"], {})[row["class_name"]] = row
    case_rows = []
    for case in sorted(candidate_by_case):
        p, c = parent_cases[case], candidate_by_case[case]
        parent_dice = float(np.mean([p[name]["dice"] for name in ("RV", "MYO", "LV")]))
        candidate_dice = float(np.mean([c[name]["dice"] for name in ("RV", "MYO", "LV")]))
        case_rows.append({
            "case": case, "parent_Dice": parent_dice, "candidate_Dice": candidate_dice, "delta_Dice": candidate_dice - parent_dice,
            "RV_delta": c["RV"]["dice"] - p["RV"]["dice"], "MYO_delta": c["MYO"]["dice"] - p["MYO"]["dice"], "LV_delta": c["LV"]["dice"] - p["LV"]["dice"],
            "HD95_delta": float(np.mean([c[name]["hd95"] - p[name]["hd95"] for name in ("RV", "MYO", "LV")])),
            "ASD_delta": float(np.mean([c[name]["asd"] - p[name]["asd"] for name in ("RV", "MYO", "LV")])),
        })
    with (AUDIT / "acdc_10pct_mask_token_adamix_case_safety.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(case_rows)

    endpoint = parse_endpoints()
    delta = {
        "mDice_pp": 100 * (best_val["dice"] - PARENT["best_val_mDice"]),
        "mIoU_pp": 100 * (best_val["iou"] - PARENT["best_val_mIoU"]),
        "HD95": best_val["hd95"] - PARENT["best_val_HD95"],
        "ASD": best_val["asd"] - PARENT["best_val_ASD"],
        "RV_pp": 100 * (best_val["RV_Dice"] - PARENT["RV_Dice"]),
        "MYO_pp": 100 * (best_val["MYO_Dice"] - PARENT["MYO_Dice"]),
        "LV_pp": 100 * (best_val["LV_Dice"] - PARENT["LV_Dice"]),
    }
    positive = sum(row["delta_Dice"] > 0 for row in case_rows)
    negative = sum(row["delta_Dice"] < 0 for row in case_rows)
    severe_negative_ratio = sum(row["delta_Dice"] <= -0.05 for row in case_rows) / len(case_rows)
    class_safe = min(delta["RV_pp"], delta["MYO_pp"], delta["LV_pp"]) > -1.0 and delta["MYO_pp"] > -0.5 and not (delta["HD95"] > 0.1 * PARENT["best_val_HD95"] and delta["ASD"] > 0.1 * PARENT["best_val_ASD"])
    case_safe = float(np.median([row["delta_Dice"] for row in case_rows])) >= 0 and severe_negative_ratio <= 0.20
    safety_pass = class_safe and case_safe
    if delta["mDice_pp"] >= 1.0 and safety_pass and endpoint["pass"]:
        decision = "ACDC_MASK_ADAMIX_STRONG_TRANSFER_POSITIVE"
    elif delta["mDice_pp"] > 0 and safety_pass and endpoint["pass"]:
        decision = "ACDC_MASK_ADAMIX_TRANSFER_POSITIVE"
    elif abs(delta["mDice_pp"]) <= 0.1:
        decision = "ACDC_MASK_ADAMIX_TRANSFER_NEUTRAL"
    else:
        decision = "ACDC_MASK_ADAMIX_TRANSFER_NEGATIVE"

    payload = {
        "status": decision, "run": str(RUN.relative_to(ROOT)), "seed": 42, "split": "official_labelnum7",
        "best_checkpoint": str(best_path.relative_to(ROOT)), "best_checkpoint_sha256": sha(best_path), "best_iteration": int(best_ckpt["global_step"]),
        "best_validation_raw": best_val, "final_validation_raw": final_val, "formal_test_raw_after_freeze": test_summary,
        "delta_vs_parent": delta, "case_safety": {"positive": positive, "negative": negative, "median_delta": float(np.median([row["delta_Dice"] for row in case_rows])), "worst": min(case_rows, key=lambda row: row["delta_Dice"]), "best": max(case_rows, key=lambda row: row["delta_Dice"]), "pass": case_safe},
        "class_safety_pass": class_safe, "mechanism_endpoint": endpoint,
    }
    (AUDIT / "acdc_10pct_mask_token_adamix_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = f"""# ACDC 10% Mask Token + AdaMix completion report

- natural completion: PASS, global step {last_ckpt['global_step']}
- validation-selected best: {best_val['dice']:.9f} @ {best_ckpt['global_step']}
- final validation mDice: {final_val['dice']:.9f}
- delta vs parent: {delta['mDice_pp']:+.4f} pp mDice; {delta['mIoU_pp']:+.4f} pp mIoU; HD95 {delta['HD95']:+.6f}; ASD {delta['ASD']:+.6f}
- class deltas: RV {delta['RV_pp']:+.4f} pp; MYO {delta['MYO_pp']:+.4f} pp; LV {delta['LV_pp']:+.4f} pp
- paired cases: {positive} positive / {negative} negative; worst `{payload['case_safety']['worst']['case']}` {payload['case_safety']['worst']['delta_Dice']:+.6f}
- class safety: {'PASS' if class_safe else 'FAIL'}; case safety: {'PASS' if case_safe else 'FAIL'}
- mechanism endpoint: {'PASS' if endpoint['pass'] else 'FAIL'}
- formal decision: `{decision}`
- test policy: evaluated once only after validation-best identity was frozen; never used for selection or decision
"""
    (AUDIT / "acdc_10pct_mask_token_adamix_completion_report.md").write_text(report)
    (AUDIT / "acdc_10pct_mask_token_adamix_decision.md").write_text(f"# ACDC Mask Token + AdaMix decision\n\n`{decision}`\n\nValidation-only transfer delta: {delta['mDice_pp']:+.4f} pp. Class safety: {'PASS' if class_safe else 'FAIL'}. Case safety: {'PASS' if case_safe else 'FAIL'}. Mechanism endpoint: {'PASS' if endpoint['pass'] else 'FAIL'}.\n")

    hash_paths = [best_path, last_path, ROOT / "train_ACDC/v14_acdc_mask_adamix_transfer.py", ROOT / "train_ACDC/v14_mask_adamix_components.py", AUDIT / "acdc_10pct_mask_token_adamix_metrics.json", AUDIT / "acdc_10pct_mask_token_adamix_case_safety.tsv"]
    with (AUDIT / "acdc_10pct_mask_token_adamix_sha256.tsv").open("w") as handle:
        handle.write("path\tsha256\n")
        for path in hash_paths:
            handle.write(f"{path.relative_to(ROOT)}\t{sha(path)}\n")
    state_path = ROOT / "experiments/acdc_mask_adamix_transfer_state.json"
    state = json.loads(state_path.read_text())
    state.update({"updated_at": datetime.now().astimezone().isoformat(), "stage": "ACDC_10PCT_COMPLETE_V14_1_5PCT_AUTOCONTINUE_PENDING", "training_authorized": False, "best_validation_mDice": best_val["dice"], "best_iteration": int(best_ckpt["global_step"]), "decision": decision, "active_launcher_pid": None, "active_controller_pid": None, "active_training_pid": None, "gpu_concurrency": 0, "next_action": "V14_1_ACDC_5PCT_BASELINE_REUSE_AND_UNCHANGED_MASK_ADAMIX_TRANSFER"})
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    with (ROOT / "experiments/acdc_mask_adamix_transfer_registry.tsv").open("a") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat()}\tFULL\tCOMPLETE\t{RUN.relative_to(ROOT)}\tACDC_10P_BETA_FFT_2D_MASK_TOKEN_ADAMIX\t42\tofficial_labelnum7\t{best_val['dice']}\t{best_ckpt['global_step']}\t{decision}\tvalidation-only decision; formal test deferred until freeze\n")
    continuation_log = AUDIT / "v14_1_acdc_5pct_autocontinue_from_10pct.log"
    with continuation_log.open("w") as output:
        continuation = subprocess.run(
            ["conda", "run", "-n", "beta_fft", "python", str(ROOT / "scripts/v14_1_acdc_5pct_autocontinue.py")],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    if continuation.returncode != 0:
        (AUDIT / "V14_1_ACDC_5PCT_AUTOCONTINUE_FAILED").write_text(str(continuation.returncode) + "\n")
        raise RuntimeError(f"V14.1 ACDC 5% autocontinue failed: {continuation.returncode}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
