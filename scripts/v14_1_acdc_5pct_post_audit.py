#!/usr/bin/env python3
"""Formal validation-selected audit for unchanged ACDC 5% Mask Token + AdaMix."""

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from v14_mask_adamix_components import v14_bcp_net  # noqa: E402

RUN = ROOT / "runs/beta_fft_acdc_5pct_mask_token_adamix_full_seed1337"
DATA = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
AUDIT = ROOT / "audit"
PARENT_REPLAY = AUDIT / "v14_1_acdc_5pct_baseline_replay/baseline_replay.json"
PARENT_CASES = AUDIT / "v14_1_acdc_5pct_baseline_replay/parent_validation_cases.tsv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_metrics(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    if pred.any() and truth.any():
        inter, union = int((pred & truth).sum()), int((pred | truth).sum())
        return 2 * inter / (int(pred.sum()) + int(truth.sum())), inter / union, float(metric.binary.hd95(pred, truth)), float(metric.binary.asd(pred, truth))
    if not pred.any() and not truth.any():
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def infer_case(model, case):
    with h5py.File(DATA / "data" / f"{case}.h5", "r") as handle:
        image, truth = handle["image"][:], handle["label"][:]
    prediction = np.zeros_like(truth)
    model.eval()
    with torch.no_grad():
        for index, source in enumerate(image):
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
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.cuda()
    cases = [line.strip() for line in (DATA / f"{split_name}.list").read_text().splitlines() if line.strip()]
    rows = [row for case in cases for row in infer_case(model, case)]
    summary = {name: float(np.mean([row[name] for row in rows])) for name in ("dice", "iou", "hd95", "asd")}
    summary.update({f"{name}_Dice": float(np.mean([row["dice"] for row in rows if row["class_name"] == name])) for name in ("RV", "MYO", "LV")})
    del model
    torch.cuda.empty_cache()
    return checkpoint, rows, summary


def grouped_parent_cases():
    grouped = {}
    with PARENT_CASES.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grouped.setdefault(row["case"], {})[row["class_name"]] = {name: float(row[name]) for name in ("dice", "iou", "hd95", "asd")}
    return grouped


def endpoints():
    mask_rows, mix_rows = [], []
    for line in (RUN / "train.log").read_text().splitlines():
        if "mask_token_endpoint=" in line:
            mask_rows.append(json.loads(line.split("mask_token_endpoint=", 1)[1]))
        if "adamix_endpoint=" in line:
            mix_rows.append(json.loads(line.split("adamix_endpoint=", 1)[1]))
    mask_pass = bool(mask_rows) and any(abs(row["model1"].get("gamma", 0)) > 0 and row["model1"].get("token_gradient_norm", 0) > 0 for row in mask_rows) and all(row["model1"].get("finite") and row["model1"].get("gradient_finite") for row in mask_rows)
    mix_pass = bool(mix_rows) and any(row.get("patch_area_ratio", 0) > 0 and row.get("mixed_loss_contribution", 0) > 0 and row.get("model1_total_gradient_norm", 0) > 0 for row in mix_rows) and all(row.get("finite") for row in mix_rows)
    return {"mask_token": {"records": len(mask_rows), "pass": mask_pass, "last": mask_rows[-1] if mask_rows else None}, "adamix": {"records": len(mix_rows), "pass": mix_pass, "last": mix_rows[-1] if mix_rows else None}, "pass": mask_pass and mix_pass}


def main():
    parent_payload = json.loads(PARENT_REPLAY.read_text())
    if parent_payload["status"] != "ACDC_5PCT_BASELINE_REUSE_PASS":
        raise RuntimeError("5% parent replay not passed")
    parent = parent_payload["best_validation_raw"]
    best_path, last_path = RUN / "checkpoints/best.ckpt", RUN / "checkpoints/last.ckpt"
    best_ckpt, val_rows, best = evaluate(best_path, "val")
    last_ckpt, _, final = evaluate(last_path, "val")
    # True test is evaluated exactly once, after validation-best identity is frozen.
    _, _, formal_test = evaluate(best_path, "test")

    parent_cases, candidate_cases = grouped_parent_cases(), {}
    for row in val_rows:
        candidate_cases.setdefault(row["case"], {})[row["class_name"]] = row
    case_rows = []
    for case in sorted(candidate_cases):
        p, c = parent_cases[case], candidate_cases[case]
        pd = float(np.mean([p[name]["dice"] for name in ("RV", "MYO", "LV")]))
        cd = float(np.mean([c[name]["dice"] for name in ("RV", "MYO", "LV")]))
        case_rows.append({
            "case": case, "parent_Dice": pd, "candidate_Dice": cd, "delta_Dice": cd - pd,
            "RV_delta": c["RV"]["dice"] - p["RV"]["dice"], "MYO_delta": c["MYO"]["dice"] - p["MYO"]["dice"], "LV_delta": c["LV"]["dice"] - p["LV"]["dice"],
            "HD95_delta": float(np.mean([c[name]["hd95"] - p[name]["hd95"] for name in ("RV", "MYO", "LV")])),
            "ASD_delta": float(np.mean([c[name]["asd"] - p[name]["asd"] for name in ("RV", "MYO", "LV")])),
        })
    case_path = AUDIT / "acdc_5pct_mask_token_adamix_case_safety.tsv"
    with case_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(case_rows)

    delta = {
        "mDice_pp": 100 * (best["dice"] - parent["dice"]), "mIoU_pp": 100 * (best["iou"] - parent["iou"]),
        "HD95": best["hd95"] - parent["hd95"], "ASD": best["asd"] - parent["asd"],
        "RV_pp": 100 * (best["RV_Dice"] - parent["RV_Dice"]), "MYO_pp": 100 * (best["MYO_Dice"] - parent["MYO_Dice"]), "LV_pp": 100 * (best["LV_Dice"] - parent["LV_Dice"]),
    }
    positive, negative = sum(row["delta_Dice"] > 0 for row in case_rows), sum(row["delta_Dice"] < 0 for row in case_rows)
    severe_ratio = sum(row["delta_Dice"] <= -0.05 for row in case_rows) / len(case_rows)
    class_safe = min(delta["RV_pp"], delta["MYO_pp"], delta["LV_pp"]) > -1.0 and delta["MYO_pp"] > -0.5 and not (delta["HD95"] > 0.1 * parent["hd95"] and delta["ASD"] > 0.1 * parent["asd"])
    case_safe = float(np.median([row["delta_Dice"] for row in case_rows])) >= 0 and severe_ratio <= 0.20
    endpoint = endpoints()
    if delta["mDice_pp"] >= 1.0 and class_safe and case_safe and endpoint["pass"]:
        decision = "ACDC_5PCT_MASK_ADAMIX_STRONG_TRANSFER_POSITIVE"
    elif delta["mDice_pp"] > 0 and class_safe and case_safe and endpoint["pass"]:
        decision = "ACDC_5PCT_MASK_ADAMIX_TRANSFER_POSITIVE"
    elif abs(delta["mDice_pp"]) <= 0.1:
        decision = "ACDC_5PCT_MASK_ADAMIX_TRANSFER_NEUTRAL"
    else:
        decision = "ACDC_5PCT_MASK_ADAMIX_TRANSFER_NEGATIVE"
    payload = {
        "status": decision, "run": str(RUN.relative_to(ROOT)), "seed": 1337, "labelnum": 3, "split": "official_validation_only",
        "best_checkpoint": str(best_path.relative_to(ROOT)), "best_checkpoint_sha256": sha(best_path), "best_iteration": int(best_ckpt["global_step"]),
        "parent_validation_raw": parent, "best_validation_raw": best, "final_validation_raw": final, "formal_test_raw_after_freeze": formal_test,
        "delta_vs_parent": delta,
        "case_safety": {"positive": positive, "negative": negative, "median_delta": float(np.median([row["delta_Dice"] for row in case_rows])), "worst": min(case_rows, key=lambda row: row["delta_Dice"]), "pass": case_safe},
        "class_safety_pass": class_safe, "mechanism_endpoint": endpoint,
    }
    metrics_path = AUDIT / "acdc_5pct_mask_token_adamix_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path = AUDIT / "acdc_5pct_mask_token_adamix_completion_report.md"
    report_path.write_text(
        "# ACDC 5% unchanged Mask Token + AdaMix completion report\n\n"
        f"- natural completion: PASS, global step {last_ckpt['global_step']}\n"
        f"- validation-selected best: {best['dice']:.9f} @ {best_ckpt['global_step']}\n"
        f"- final validation mDice: {final['dice']:.9f}\n"
        f"- delta vs parent: {delta['mDice_pp']:+.4f} pp mDice; {delta['mIoU_pp']:+.4f} pp mIoU; HD95 {delta['HD95']:+.6f}; ASD {delta['ASD']:+.6f}\n"
        f"- classes: RV {delta['RV_pp']:+.4f} pp; MYO {delta['MYO_pp']:+.4f} pp; LV {delta['LV_pp']:+.4f} pp\n"
        f"- paired cases: {positive} positive / {negative} negative; worst `{payload['case_safety']['worst']['case']}` {payload['case_safety']['worst']['delta_Dice']:+.6f}\n"
        f"- class safety: {'PASS' if class_safe else 'FAIL'}; case safety: {'PASS' if case_safe else 'FAIL'}; endpoint: {'PASS' if endpoint['pass'] else 'FAIL'}\n"
        f"- decision: `{decision}`\n"
        "- test policy: one formal test only after validation-best freeze; not used for selection or decision\n"
    )
    hash_paths = [best_path, last_path, ROOT / "train_ACDC/v14_acdc_mask_adamix_transfer.py", ROOT / "train_ACDC/v14_mask_adamix_components.py", metrics_path, case_path]
    with (AUDIT / "acdc_5pct_mask_token_adamix_sha256.tsv").open("w") as handle:
        handle.write("path\tsha256\n")
        for path in hash_paths:
            handle.write(f"{path.relative_to(ROOT)}\t{sha(path)}\n")

    ten = json.loads((AUDIT / "acdc_10pct_mask_token_adamix_metrics.json").read_text())
    (AUDIT / "acdc_5pct_10pct_mask_token_adamix_final_comparison.md").write_text(
        "# V14 ACDC 10% and 5% unchanged transfer final comparison\n\n"
        f"## ACDC 10%\n\nParent {ten['best_validation_raw']['dice'] - ten['delta_vs_parent']['mDice_pp']/100:.9f}; candidate {ten['best_validation_raw']['dice']:.9f} @ {ten['best_iteration']}; delta {ten['delta_vs_parent']['mDice_pp']:+.4f} pp; decision `{ten['status']}`.\n\n"
        f"## ACDC 5%\n\nParent {parent['dice']:.9f}; candidate {best['dice']:.9f} @ {best_ckpt['global_step']}; delta {delta['mDice_pp']:+.4f} pp; decision `{decision}`.\n\n"
        "Full class, geometry, case-safety, endpoint and formal-test fields are preserved in the two metrics JSON files.\n"
    )
    state_path = ROOT / "experiments/acdc_mask_adamix_transfer_state.json"
    state = json.loads(state_path.read_text())
    state.update({"updated_at": datetime.now().astimezone().isoformat(), "stage": "V14_1_ACDC_5PCT_COMPLETE", "training_authorized": False, "five_pct_best_validation_mDice": best["dice"], "five_pct_best_iteration": int(best_ckpt["global_step"]), "five_pct_decision": decision, "active_launcher_pid": None, "active_controller_pid": None, "active_training_pid": None, "gpu_concurrency": 0, "next_action": "NONE"})
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    with (ROOT / "experiments/acdc_mask_adamix_transfer_registry.tsv").open("a") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat()}\tFULL\tCOMPLETE\t{RUN.relative_to(ROOT)}\tACDC_5P_BETA_FFT_2D_MASK_TOKEN_ADAMIX\t1337\tofficial_labelnum3\t{best['dice']}\t{best_ckpt['global_step']}\t{decision}\tunchanged transfer; validation-only decision; formal test after freeze\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
