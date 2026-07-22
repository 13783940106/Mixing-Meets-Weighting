#!/usr/bin/env python3
"""Validation-only replay of the frozen historical ACDC 5% 2D parent."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from networks.net_factory import BCP_net  # noqa: E402

DATA = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
RUN = ROOT / "runs/acdc_5pct_2d_matched_control_seed1337_valonly_20260628_094308"
OUT = ROOT / "audit/v14_1_acdc_5pct_baseline_replay"
EXPECTED = {
    "source_snapshot_sha256": "a4ad0567a02bb8dca6757fd9e45c333d5f1df70201d3bdce171511e22f89fc4a",
    "best_checkpoint_sha256": "7b73685aee068249c76aa12ae0567373420cdd214a82a3af3e3ab31a6a680e56",
    "last_checkpoint_sha256": "53cb20fd2fccfc06d0f5c42adce503913d4683093cbfc3288fda4860a23af490",
    "val_list_sha256": "61b935c936d8e12f7889e82d954f421fa25fb95305e92a476096b3242de8777d",
    "recorded_best_mDice": 0.8925191580834845,
    "recorded_best_iteration": 24400,
    "recorded_final_mDice": 0.8853426918578148,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_metrics(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    if pred.any() and truth.any():
        inter = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return (
            2 * inter / (int(pred.sum()) + int(truth.sum())),
            inter / union,
            float(metric.binary.hd95(pred, truth)),
            float(metric.binary.asd(pred, truth)),
        )
    if not pred.any() and not truth.any():
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def infer_case(model, case):
    with h5py.File(DATA / "data" / f"{case}.h5", "r") as handle:
        image, truth = handle["image"][:], handle["label"][:]
    prediction = np.zeros_like(truth)
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


def evaluate(path):
    checkpoint = torch.load(path, map_location="cpu")
    if any(not torch.isfinite(value).all() for value in checkpoint["model_1"].values() if torch.is_tensor(value)):
        raise RuntimeError(f"non-finite checkpoint: {path}")
    model = BCP_net(in_chns=1, class_num=4, ema=False).cuda()
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.eval()
    cases = [line.strip() for line in (DATA / "val.list").read_text().splitlines() if line.strip()]
    rows = [row for case in cases for row in infer_case(model, case)]
    del model
    torch.cuda.empty_cache()
    summary = {name: float(np.mean([row[name] for row in rows])) for name in ("dice", "iou", "hd95", "asd")}
    summary.update({f"{name}_Dice": float(np.mean([row["dice"] for row in rows if row["class_name"] == name])) for name in ("RV", "MYO", "LV")})
    return checkpoint, rows, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    identities = {
        "source_snapshot_sha256": sha(RUN / "source_snapshot.py"),
        "best_checkpoint_sha256": sha(RUN / "checkpoints/best.ckpt"),
        "last_checkpoint_sha256": sha(RUN / "checkpoints/last.ckpt"),
        "val_list_sha256": sha(DATA / "val.list"),
    }
    for key, actual in identities.items():
        if actual != EXPECTED[key]:
            raise RuntimeError(f"5% parent identity mismatch {key}: {actual}")
    best_ckpt, best_rows, best = evaluate(RUN / "checkpoints/best.ckpt")
    last_ckpt, _, final = evaluate(RUN / "checkpoints/last.ckpt")
    # The legacy closure and this replay use equivalent raw validation semantics;
    # allow only numerical implementation noise, never a material metric drift.
    replay_pass = (
        int(best_ckpt["global_step"]) == EXPECTED["recorded_best_iteration"]
        and abs(best["dice"] - EXPECTED["recorded_best_mDice"]) <= 1e-6
        and abs(final["dice"] - EXPECTED["recorded_final_mDice"]) <= 1e-6
        and int(last_ckpt["global_step"]) == 30000
    )
    with (OUT / "parent_validation_cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(best_rows)
    payload = {
        "status": "ACDC_5PCT_BASELINE_REUSE_PASS" if replay_pass else "ACDC_5PCT_BASELINE_REPLAY_FAIL",
        "rerun": False if replay_pass else None,
        "seed": 1337,
        "labelnum": 3,
        "split": "official_validation_only",
        "run": str(RUN.relative_to(ROOT)),
        "best_iteration": int(best_ckpt["global_step"]),
        "best_validation_raw": best,
        "final_validation_raw": final,
        "identity": identities,
        "recorded": EXPECTED,
    }
    (OUT / "baseline_replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "baseline_reuse_decision.md").write_text(
        f"# V14.1 ACDC 5% parent replay\n\n`{payload['status']}`\n\n"
        f"Seed 1337, labelnum=3, best validation mDice {best['dice']:.9f} @ {best_ckpt['global_step']}. "
        f"Baseline training rerun: {'NO_RERUN' if replay_pass else 'NOT_DECIDED'}.\n"
    )
    print(json.dumps(payload, indent=2))
    if not replay_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
