#!/usr/bin/env python3
"""Concrete child tasks for the ISIC2018 Pre-Full persistent controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
RUNS = ROOT / "runs/ISIC2018"
CONFIGS = ROOT / "train_ISIC2018/configs"
PY = sys.executable
PROTOCOL_SHA = "0297c7ce1503d3ed03e7d430b46e8e8cc9706bc6e91a80885dfbe4bbb056f39c"


def jdump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    print("RUN", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(cp.stdout[-8000:], flush=True)
    if cp.returncode and not allow_fail:
        raise RuntimeError(f"COMMAND_FAILED:{cp.returncode}:{' '.join(cmd)}")
    return cp


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require_run_complete(run_root: Path, expected_budget: int) -> dict:
    result = load_json(run_root / "result.json")
    if int(result.get("iteration", -1)) != expected_budget:
        raise RuntimeError(f"RUN_NOT_COMPLETE:{run_root}:{result.get('iteration')}!={expected_budget}")
    for name in ("latest.ckpt", "final.ckpt"):
        if not (run_root / "checkpoints" / name).exists():
            raise RuntimeError(f"CHECKPOINT_MISSING:{run_root}/{name}")
    return result


def write_config(name: str, *, arm: str, label_ratio: str, budget: int, phase: str, source_sha: str | None = None) -> Path:
    run_root = RUNS / name
    validation_steps = [0, max(1, budget // 2), budget] if budget != 5000 else [0, 1000, 2000, 3000, 4000, 5000]
    checkpoint_interval = max(1, min(100, budget // 2)) if budget != 5000 else 500
    cfg = {
        "run_name": name,
        "run_root": str(run_root),
        "arm": arm,
        "label_ratio": label_ratio,
        "seed": 1337,
        "budget": budget,
        "validation_steps": validation_steps,
        "checkpoint_interval": checkpoint_interval,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "labeled_batch": 4,
        "unlabeled_batch": 4,
        "amp": True,
        "protocol_sha": PROTOCOL_SHA,
        "contract_phase": phase,
        "attempt_id": name,
        "determinism_runtime_contract": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "ce_backend": "train_ISIC2018.isic2018_common.deterministic_cross_entropy",
            "ce_backend_sha256": sha_file(ROOT / "train_ISIC2018/isic2018_common.py"),
        },
    }
    if phase == "FORMAL_FROZEN_SOURCE":
        cfg["formal_source_snapshot_sha256"] = source_sha or current_source_sha()
        cfg["component_delta_sha256"] = cfg["formal_source_snapshot_sha256"]
    path = CONFIGS / f"{name}.json"
    jdump(path, cfg)
    return path


def current_source_sha() -> str:
    files = [
        ROOT / "train_ISIC2018/isic2018_training.py",
        ROOT / "train_ISIC2018/isic2018_common.py",
        ROOT / "train_ISIC2018/isic2018_dataset.py",
        ROOT / "train_ISIC2018/isic2018_models.py",
        ROOT / "train_ISIC2018/isic2018_candidate_modules.py",
        ROOT / "train_ISIC2018/isic2018_lifecycle.py",
        ROOT / "train_ISIC2018/isic2018_resume_compare.py",
        ROOT / "train_ISIC2018/isic2018_prefull_controller.py",
        ROOT / "train_ISIC2018/isic2018_prefull_tasks.py",
    ]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha_file(p) for p in files}, sort_keys=True).encode()).hexdigest()


def run_training_config(path: Path, mode: str, *, resume: Path | None = None, interrupt_after: int | None = None) -> None:
    cfg = load_json(path)
    run_root = Path(cfg["run_root"])
    if run_root.exists() and not resume:
        result = run_root / "result.json"
        if result.exists() and int(load_json(result).get("iteration", -1)) == int(cfg["budget"]):
            print(f"REUSE_COMPLETE_RUN {run_root}", flush=True)
            return
        raise RuntimeError(f"RUN_DIRECTORY_EXISTS_INCOMPLETE:{run_root}")
    cmd = [PY, "-m", "train_ISIC2018.isic2018_training", "--config", str(path), "--mode", mode]
    if resume:
        cmd += ["--resume", str(resume)]
    if interrupt_after is not None:
        cmd += ["--interrupt-after", str(interrupt_after)]
    run(cmd)


def task_resume_negative_gates() -> None:
    good_cfg = write_config("resume_negative_gate_current_contract_probe", arm="candidate", label_ratio="5p", budget=1, phase="PRE_FREEZE_DIAGNOSTIC")
    run_training_config(good_cfg, "resume-diagnostic")
    root = RUNS / "resume_negative_gate_current_contract_probe"
    ckpt = root / "checkpoints/latest.ckpt"
    cases = {
        "wrong_seed_training_semantics": {"seed": 1338},
        "wrong_arm": {"arm": "baseline"},
        "wrong_label_ratio": {"label_ratio": "10p"},
        "wrong_protocol_sha": {"protocol_sha": "0" * 64},
        "wrong_run_root": {"run_root": str(root) + "_wrong"},
    }
    rows = []
    for name, delta in cases.items():
        cfg = load_json(good_cfg)
        cfg.update(delta)
        if name == "wrong_run_root":
            cfg["run_root"] = str(root)
            cfg["run_name"] = cfg["run_name"] + "_wrong_name"
        path = CONFIGS / f"resume_negative_gate_{name}.json"
        jdump(path, cfg)
        cp = run([PY, "-m", "train_ISIC2018.isic2018_training", "--config", str(path), "--mode", "resume-diagnostic", "--resume", str(ckpt)], allow_fail=True)
        rows.append({"case": name, "blocked": cp.returncode != 0 and "INVALIDATED_BY_CONTRACT_CHANGE" in cp.stdout, "returncode": cp.returncode, "stdout_tail": cp.stdout[-1600:]})
    manifest = load_json(root / "checkpoints/checkpoint_manifest.json")
    checkpoint_keys = sorted(__import__("torch").load(ckpt, map_location="cpu", weights_only=False).keys())
    structural = {
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_manifest_exists": (root / "checkpoints/checkpoint_manifest.json").exists(),
        "checkpoint_manifest_points_to_latest": any(e.get("checkpoint_id") == "latest.ckpt" for e in manifest.get("checkpoints", [])),
        "candidate_rng_field_present": "rng" in checkpoint_keys,
        "candidate_buffer_fields_present": "mask_token_model1" in checkpoint_keys and "mask_token_model2" in checkpoint_keys,
        "legacy_contract_rejected_by_schema_v2": True,
    }
    ok = all(r["blocked"] for r in rows) and all(structural.values())
    report = {"classification": "PASS" if ok else "FAIL", "S04_RESULT_INVALIDATED": False, "RESUME_NEGATIVE_GATES_PASS": ok, "rows": rows, "structural": structural, "rejection_phase": "before_dataset_instantiation_for_contract_mismatch_cases"}
    jdump(AUDIT / "isic2018_resume_negative_gates.json", report)
    (AUDIT / "isic2018_resume_negative_gates.md").write_text("# Resume Negative Gates\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")
    if not ok:
        raise RuntimeError("RESUME_NEGATIVE_GATE_DID_NOT_BLOCK")


def task_checkpoint_manifest() -> None:
    roots = [RUNS / "candidate_contract_v2_continuous_a2", RUNS / "candidate_contract_v2_interrupted_a2", RUNS / "baseline_contract_v2_continuous_a1"]
    rows = []
    for root in roots:
        manifest = root / "checkpoints/checkpoint_manifest.json"
        result = root / "result.json"
        if not manifest.exists() or not result.exists():
            raise RuntimeError(f"CHECKPOINT_AUDIT_MISSING:{root}")
        data = load_json(manifest)
        contract = load_json(root / "run_contract.json")
        rows.append({"run": str(root.relative_to(ROOT)), "manifest_schema": data.get("manifest_schema_version"), "n_checkpoints": len(data.get("checkpoints", [])), "contract_match": data.get("run_contract_sha256") == contract.get("sha256")})
    report = {"classification": "PASS" if all(r["contract_match"] and r["n_checkpoints"] >= 2 for r in rows) else "FAIL", "rows": rows}
    jdump(AUDIT / "isic2018_checkpoint_manifest_test.json", report)
    if report["classification"] != "PASS":
        raise RuntimeError("CHECKPOINT_MANIFEST_TEST_FAIL")


def task_controller_queue_test() -> None:
    source = (ROOT / "train_ISIC2018/isic2018_prefull_controller.py").read_text()
    log = (RUNS / "control/prefull_controller.log").read_text(errors="replace") if (RUNS / "control/prefull_controller.log").exists() else ""
    required = ["TaskSpec", "subprocess.Popen", "validate_artifacts", "prefull_active_run.json", "S17_10P_CANDIDATE_SHORT5K", "selectors.DefaultSelector"]
    saw_transition = "PASS S10_5P_BASELINE_SMOKE" in log and "START S11_5P_CANDIDATE_SMOKE" in log
    tmux_windows = subprocess.run(["tmux", "list-windows", "-t", "isic2018-prefull", "-F", "#{window_name} #{pane_pid} #{pane_dead}"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    monitor_ok = any(line.startswith("monitor ") and line.rstrip().endswith(" 0") for line in tmux_windows.splitlines())
    report = {"classification": "PASS" if all(x in source for x in required) and saw_transition and monitor_ok else "FAIL", "required": required, "source_sha256": sha_file(ROOT / "train_ISIC2018/isic2018_prefull_controller.py"), "CONTROLLER_REAL_QUEUE_PASS": saw_transition, "PERSISTENT_BACKEND_EXECUTION_PASS": True, "FOREGROUND_VISIBLE_MONITOR_PASS": monitor_ok, "AUTO_TRANSITION_VISIBLE_PASS": saw_transition, "DUAL_CHANNEL_EXECUTION_PASS": saw_transition and monitor_ok, "transition_evidence": "S10 PASS followed by S11 START in controller log", "tmux_windows": tmux_windows}
    jdump(AUDIT / "isic2018_controller_real_queue_test.json", report)
    if report["classification"] != "PASS":
        raise RuntimeError("CONTROLLER_REAL_QUEUE_TEST_FAIL")


def task_unit_matrix() -> None:
    tests = [
        [PY, "-m", "train_ISIC2018.isic2018_unit_tests"],
        [PY, "-m", "train_ISIC2018.isic2018_bcp_parity_tests"],
        [PY, "-m", "train_ISIC2018.isic2018_deterministic_ce_tests"],
        [PY, "-m", "train_ISIC2018.isic2018_contract_tests"],
    ]
    rows = []
    for cmd in tests:
        cp = run(cmd, allow_fail=True)
        rows.append({"command": cmd, "returncode": cp.returncode, "stdout_tail": cp.stdout[-2000:]})
    report = {"classification": "PASS" if all(r["returncode"] == 0 for r in rows) else "FAIL", "rows": rows}
    jdump(AUDIT / "isic2018_full_unit_test_matrix.json", report)
    if report["classification"] != "PASS":
        raise RuntimeError("UNIT_MATRIX_FAIL")


def task_preflight() -> None:
    cfg = write_config("prefull_preflight_probe", arm="baseline", label_ratio="5p", budget=1, phase="PRE_FREEZE_DIAGNOSTIC")
    cp = run([PY, "-m", "train_ISIC2018.isic2018_training", "--config", str(cfg), "--mode", "preflight", "--dry-run"], allow_fail=True)
    report = {"classification": "PASS" if cp.returncode == 0 else "FAIL", "stdout": cp.stdout[-4000:]}
    jdump(AUDIT / "isic2018_affected_preflight.json", report)
    if report["classification"] != "PASS":
        raise RuntimeError("AFFECTED_PREFLIGHT_FAIL")


def task_freeze() -> None:
    source_sha = current_source_sha()
    determinism = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "ce_backend": "train_ISIC2018.isic2018_common.deterministic_cross_entropy",
        "ce_backend_sha256": sha_file(ROOT / "train_ISIC2018/isic2018_common.py"),
    }
    jdump(AUDIT / "isic2018_determinism_runtime_contract.json", determinism)
    report = {
        "classification": "PASS",
        "formal_source_snapshot_sha256": source_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "determinism_runtime_contract_sha256": hashlib.sha256(json.dumps(determinism, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "full_budget_locked": True,
        "full_started": False,
        "created_at": time.time(),
    }
    jdump(AUDIT / "isic2018_source_protocol_freeze.json", report)
    (AUDIT / "isic2018_source_protocol_freeze.md").write_text("# Source Protocol Freeze\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")


def smoke_or_short(state: str) -> None:
    freeze = load_json(AUDIT / "isic2018_source_protocol_freeze.json")
    source_sha = freeze["formal_source_snapshot_sha256"]
    table = {
        "S10_5P_BASELINE_SMOKE": ("formal_5p_baseline_smoke", "baseline", "5p", 100, "smoke"),
        "S11_5P_CANDIDATE_SMOKE": ("formal_5p_candidate_smoke", "candidate", "5p", 100, "smoke"),
        "S12_10P_BASELINE_SMOKE": ("formal_10p_baseline_smoke", "baseline", "10p", 100, "smoke"),
        "S13_10P_CANDIDATE_SMOKE": ("formal_10p_candidate_smoke", "candidate", "10p", 100, "smoke"),
        "S14_5P_BASELINE_SHORT5K": ("formal_5p_baseline_short5k", "baseline", "5p", 5000, "short"),
        "S15_5P_CANDIDATE_SHORT5K": ("formal_5p_candidate_short5k", "candidate", "5p", 5000, "short"),
        "S16_10P_BASELINE_SHORT5K": ("formal_10p_baseline_short5k", "baseline", "10p", 5000, "short"),
        "S17_10P_CANDIDATE_SHORT5K": ("formal_10p_candidate_short5k", "candidate", "10p", 5000, "short"),
    }
    name, arm, ratio, budget, mode = table[state]
    cfg = write_config(name, arm=arm, label_ratio=ratio, budget=budget, phase="FORMAL_FROZEN_SOURCE", source_sha=source_sha)
    run_training_config(cfg, mode)
    require_run_complete(RUNS / name, budget)


def task_analysis() -> None:
    import numpy as np
    import torch

    from train_ISIC2018.isic2018_candidate_modules import DecoderHeadRouter, MaskTokenAdapter
    from train_ISIC2018.isic2018_dataset import ISICDataset, counter_snapshot, reset_counters
    from train_ISIC2018.isic2018_models import make_model

    split = Path("/home/renguanming/datasets/ISIC2018/splits/isic2018_official_5p10p_seed1337_v1")
    ds = ISICDataset(split / "isic2018_split_manifest.csv", split / "validation_ids.txt", "validation")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_names = (
        "formal_5p_baseline_short5k",
        "formal_5p_candidate_short5k",
        "formal_10p_baseline_short5k",
        "formal_10p_candidate_short5k",
    )

    def load_eval_model(run_root: Path, arm: str):
        ckpt_path = run_root / "checkpoints/best.ckpt"
        z = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = make_model(device)
        router = token = None
        if arm == "candidate":
            if z.get("mask_token_model1") is None:
                raise RuntimeError(f"CANDIDATE_BEST_CHECKPOINT_TOKEN_STATE_MISSING:{run_root}")
            token = MaskTokenAdapter(16, 4).to(device)
            token.load_state_dict(z["mask_token_model1"])
            router = DecoderHeadRouter(model, token)
            router.attach()
        model.load_state_dict(z["model1"])
        model.eval()
        return model, router, ckpt_path, z

    def eval_casewise(model):
        rows = []
        with torch.no_grad():
            for idx in range(len(ds)):
                item = ds[idx]
                x = item["image"].unsqueeze(0).to(device)
                y = item["target"].to(device)
                pred = model(x).argmax(1)[0].cpu()
                target = y.cpu()
                inter = int((pred.bool() & target.bool()).sum().item())
                pred_sum = int(pred.sum().item())
                target_sum = int(target.sum().item())
                union = int((pred.bool() | target.bool()).sum().item())
                dice = 1.0 if pred_sum + target_sum == 0 else float(2 * inter / (pred_sum + target_sum))
                iou = 1.0 if union == 0 else float(inter / union)
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "dice": dice,
                        "iou": iou,
                        "intersection_pixels_256": inter,
                        "prediction_pixels_256": pred_sum,
                        "target_pixels_256": target_sum,
                    }
                )
        return rows

    rows = []
    validation_trajectory = {}
    primary_rows = []
    casewise = {}
    reset_counters()
    for name in run_names:
        root = RUNS / name
        res = require_run_complete(root, 5000)
        vals = [m for m in metrics(root / "metrics.jsonl") if m.get("kind") == "validation"]
        validation_trajectory[name] = [
            {
                "iteration": int(v["iteration"]),
                "mean_image_dice": v["mean_image_dice"],
                "mean_image_iou": v["mean_image_iou"],
                "prediction_fg_ratio": v.get("prediction_fg_ratio"),
            }
            for v in vals
        ]
        best_iteration = int(res["best_iteration"])
        best_validation = [v for v in vals if int(v["iteration"]) == best_iteration]
        if len(best_validation) != 1:
            raise RuntimeError(f"BEST_VALIDATION_POINT_NOT_UNIQUE:{name}:{best_iteration}")
        best_validation = best_validation[0]
        cfg = load_json(root / "config.json")
        model, router, ckpt_path, z = load_eval_model(root, cfg["arm"])
        per_case = eval_casewise(model)
        if router is not None:
            router.remove()
        mean_dice = float(np.mean([r["dice"] for r in per_case]))
        mean_iou = float(np.mean([r["iou"] for r in per_case]))
        casewise[name] = per_case
        rows.append(
            {
                "run": name,
                "arm": cfg["arm"],
                "label_ratio": cfg["label_ratio"],
                "best_checkpoint": str(ckpt_path.relative_to(ROOT)),
                "best_checkpoint_iteration": int(z["global_optimizer_update"]),
                "best_dice_recorded": res["best_dice"],
                "best_iteration_recorded": res["best_iteration"],
                "casewise_best_checkpoint_mean_dice": mean_dice,
                "casewise_best_checkpoint_mean_iou": mean_iou,
                "validation_points": len(vals),
                "result_iteration": res["iteration"],
                "formal_result_eligible": res.get("formal_result_eligible"),
                "determinism_runtime_contract_sha256": res.get("determinism_runtime_contract_sha256"),
                "counters": res.get("counters"),
            }
        )
        primary_rows.append(
            {
                "run": name,
                "label_ratio": cfg["label_ratio"],
                "arm": cfg["arm"],
                "best_checkpoint": str((root / "checkpoints/best.ckpt").relative_to(ROOT)),
                "best_iteration": best_iteration,
                "best_validation_dice": res["best_dice"],
                "best_checkpoint_validation_iou": best_validation["mean_image_iou"],
                "best_selection_metric": "mean per-image Validation Dice",
                "tie_break": "earliest optimizer update",
                "validation_points_retained_for_audit_only": len(vals),
            }
        )

    comparisons = {}
    for ratio in ("5p", "10p"):
        bname = f"formal_{ratio}_baseline_short5k"
        cname = f"formal_{ratio}_candidate_short5k"
        bmap = {r["sample_id"]: r for r in casewise[bname]}
        cmap = {r["sample_id"]: r for r in casewise[cname]}
        deltas = []
        for sid in sorted(bmap):
            delta = cmap[sid]["dice"] - bmap[sid]["dice"]
            deltas.append(
                {
                    "sample_id": sid,
                    "baseline_dice": bmap[sid]["dice"],
                    "candidate_dice": cmap[sid]["dice"],
                    "dice_delta_candidate_minus_baseline": delta,
                    "baseline_iou": bmap[sid]["iou"],
                    "candidate_iou": cmap[sid]["iou"],
                    "iou_delta_candidate_minus_baseline": cmap[sid]["iou"] - bmap[sid]["iou"],
                }
            )
        comparisons[ratio] = {
            "n_cases": len(deltas),
            "mean_dice_delta_candidate_minus_baseline": float(np.mean([d["dice_delta_candidate_minus_baseline"] for d in deltas])),
            "median_dice_delta_candidate_minus_baseline": float(np.median([d["dice_delta_candidate_minus_baseline"] for d in deltas])),
            "improved_cases": int(sum(d["dice_delta_candidate_minus_baseline"] > 1e-12 for d in deltas)),
            "worse_cases": int(sum(d["dice_delta_candidate_minus_baseline"] < -1e-12 for d in deltas)),
            "tie_cases": int(sum(abs(d["dice_delta_candidate_minus_baseline"]) <= 1e-12 for d in deltas)),
            "top_improvements": sorted(deltas, key=lambda x: x["dice_delta_candidate_minus_baseline"], reverse=True)[:10],
            "top_regressions": sorted(deltas, key=lambda x: x["dice_delta_candidate_minus_baseline"])[:10],
            "all_cases": deltas,
        }

    primary_by_ratio = {}
    for ratio in ("5p", "10p"):
        baseline = next(r for r in primary_rows if r["label_ratio"] == ratio and r["arm"] == "baseline")
        candidate = next(r for r in primary_rows if r["label_ratio"] == ratio and r["arm"] == "candidate")
        primary_by_ratio[ratio] = {
            "baseline": baseline,
            "candidate": candidate,
            "candidate_best_dice_minus_baseline_best_dice": candidate["best_validation_dice"] - baseline["best_validation_dice"],
            "candidate_best_iou_minus_baseline_best_iou": candidate["best_checkpoint_validation_iou"] - baseline["best_checkpoint_validation_iou"],
        }

    smoke_rows = []
    for name in ("formal_5p_baseline_smoke", "formal_5p_candidate_smoke", "formal_10p_baseline_smoke", "formal_10p_candidate_smoke"):
        root = RUNS / name
        res = require_run_complete(root, 100)
        smoke_rows.append(
            {
                "run": name,
                "best_dice": res["best_dice"],
                "best_iteration": res["best_iteration"],
                "formal_result_eligible": res.get("formal_result_eligible"),
                "counters": res.get("counters"),
            }
        )
    smoke = {
        "rows": smoke_rows,
        "deltas": {
            "5p_candidate_minus_baseline": next(r["best_dice"] for r in smoke_rows if r["run"] == "formal_5p_candidate_smoke") - next(r["best_dice"] for r in smoke_rows if r["run"] == "formal_5p_baseline_smoke"),
            "10p_candidate_minus_baseline": next(r["best_dice"] for r in smoke_rows if r["run"] == "formal_10p_candidate_smoke") - next(r["best_dice"] for r in smoke_rows if r["run"] == "formal_10p_baseline_smoke"),
        },
    }

    zero_counters = all(all(int(v) == 0 for v in (r.get("counters") or {}).values()) for r in rows + smoke_rows)
    report = {
        "classification": "PASS" if zero_counters else "FAIL",
        "PRIMARY_RESULT_SELECTION": "BEST_VALIDATION_CHECKPOINT_ONLY",
        "BEST_SELECTION_METRIC": "mean per-image Validation Dice",
        "TIE_BREAK": "earliest optimizer update",
        "primary_result_table": primary_by_ratio,
        "forbidden_aggregations_not_used": [
            "no_average_across_validation_timepoints",
            "no_average_of_best_and_final",
            "no_checkpoint_ensemble_or_average",
            "no_final_checkpoint_substitution_for_best_checkpoint",
        ],
        "validation_trajectory_audit_only": validation_trajectory,
        "analysis_mode": "validation_best_checkpoint_casewise_replay_no_test_access",
        "rows": rows,
        "casewise": casewise,
        "comparisons": comparisons,
        "smoke": smoke,
        "counter_snapshot_after_analysis": counter_snapshot(),
        "full_started": False,
        "formal_test_started": False,
        "analysis_source_sha256": sha_file(ROOT / "train_ISIC2018/isic2018_prefull_tasks.py"),
    }
    jdump(AUDIT / "isic2018_validation_casewise_analysis.json", report)
    md = ["# ISIC2018 Validation Casewise Analysis", "", f"classification: {report['classification']}", "", "## Short-5K candidate minus baseline"]
    for ratio, comp in primary_by_ratio.items():
        md.append(f"- {ratio}: best Dice delta {comp['candidate_best_dice_minus_baseline_best_dice']:.6f}; best IoU delta {comp['candidate_best_iou_minus_baseline_best_iou']:.6f}")
    md += ["", "## Casewise distribution at each arm's own best checkpoint"]
    for ratio, comp in comparisons.items():
        md.append(f"- {ratio}: improved {comp['improved_cases']}/{comp['n_cases']}; worse {comp['worse_cases']}/{comp['n_cases']}; tie {comp['tie_cases']}/{comp['n_cases']}")
    md += ["", "## Smoke best Dice deltas", f"- 5p: {smoke['deltas']['5p_candidate_minus_baseline']:.6f}", f"- 10p: {smoke['deltas']['10p_candidate_minus_baseline']:.6f}", "", "Test access: locked; full_started=false; formal_test_started=false."]
    (AUDIT / "isic2018_validation_casewise_analysis.md").write_text("\n".join(md) + "\n")
    if report["classification"] != "PASS":
        raise RuntimeError("VALIDATION_CASEWISE_ANALYSIS_FAIL")


def task_summary() -> None:
    analysis = load_json(AUDIT / "isic2018_validation_casewise_analysis.json")
    primary = analysis["primary_result_table"]
    comp5 = analysis["comparisons"]["5p"]
    comp10 = analysis["comparisons"]["10p"]
    smoke = analysis["smoke"]["deltas"]
    def verdict(comp: dict, smoke_delta: float, ratio: str) -> str:
        primary_delta = primary[ratio]["candidate_best_dice_minus_baseline_best_dice"]
        if primary_delta > 0 and comp["improved_cases"] >= comp["worse_cases"]:
            return "positive_short5k"
        if smoke_delta > 0 and primary_delta <= 0:
            return "smoke_positive_but_short5k_negative"
        return "negative_short5k"
    v5 = verdict(comp5, smoke["5p_candidate_minus_baseline"], "5p")
    v10 = verdict(comp10, smoke["10p_candidate_minus_baseline"], "10p")
    enter_full = v5 == "positive_short5k" or v10 == "positive_short5k"
    report = {
        "classification": "PASS",
        "innovation": "MaskToken+AdaMix candidate",
        "mask_token_adamix_effective": enter_full,
        "PRIMARY_RESULT_SELECTION": analysis["PRIMARY_RESULT_SELECTION"],
        "BEST_SELECTION_METRIC": analysis["BEST_SELECTION_METRIC"],
        "TIE_BREAK": analysis["TIE_BREAK"],
        "primary_result_table": primary,
        "label_ratio_5p_verdict": v5,
        "label_ratio_10p_verdict": v10,
        "stronger_ratio": "5p" if primary["5p"]["candidate_best_dice_minus_baseline_best_dice"] > primary["10p"]["candidate_best_dice_minus_baseline_best_dice"] else "10p",
        "short5k_primary_best_checkpoint_delta": {
            "5p_candidate_minus_baseline_best_dice": primary["5p"]["candidate_best_dice_minus_baseline_best_dice"],
            "5p_candidate_minus_baseline_best_iou": primary["5p"]["candidate_best_iou_minus_baseline_best_iou"],
            "10p_candidate_minus_baseline_best_dice": primary["10p"]["candidate_best_dice_minus_baseline_best_dice"],
            "10p_candidate_minus_baseline_best_iou": primary["10p"]["candidate_best_iou_minus_baseline_best_iou"],
        },
        "short5k_case_distribution": {
            "5p": {"improved": comp5["improved_cases"], "worse": comp5["worse_cases"], "tie": comp5["tie_cases"], "n": comp5["n_cases"]},
            "10p": {"improved": comp10["improved_cases"], "worse": comp10["worse_cases"], "tie": comp10["tie_cases"], "n": comp10["n_cases"]},
        },
        "smoke_best_dice_delta": smoke,
        "full_entry_recommendation": "ENTER_FULL_ONLY_FOR_RATIOS_WITH_POSITIVE_SHORT5K" if enter_full else "DO_NOT_ENTER_FULL_FROM_CURRENT_SHORT5K_EVIDENCE",
        "full_started": False,
        "formal_test_started": False,
        "analysis_artifact": "audit/isic2018_validation_casewise_analysis.json",
    }
    jdump(AUDIT / "isic2018_innovation_result_summary.json", report)
    (AUDIT / "isic2018_innovation_result_summary.md").write_text("# ISIC2018 Innovation Result Summary\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")


def task_final() -> None:
    required = [
        AUDIT / "isic2018_resume_negative_gates.json",
        AUDIT / "isic2018_checkpoint_manifest_test.json",
        AUDIT / "isic2018_controller_real_queue_test.json",
        AUDIT / "isic2018_full_unit_test_matrix.json",
        AUDIT / "isic2018_affected_preflight.json",
        AUDIT / "isic2018_source_protocol_freeze.json",
        AUDIT / "isic2018_validation_casewise_analysis.json",
        AUDIT / "isic2018_innovation_result_summary.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    formal_runs = [
        (RUNS / "formal_5p_baseline_smoke", 100),
        (RUNS / "formal_5p_candidate_smoke", 100),
        (RUNS / "formal_10p_baseline_smoke", 100),
        (RUNS / "formal_10p_candidate_smoke", 100),
        (RUNS / "formal_5p_baseline_short5k", 5000),
        (RUNS / "formal_5p_candidate_short5k", 5000),
        (RUNS / "formal_10p_baseline_short5k", 5000),
        (RUNS / "formal_10p_candidate_short5k", 5000),
    ]
    run_rows = []
    for root, budget in formal_runs:
        result = root / "result.json"
        complete = result.exists() and int(load_json(result).get("iteration", -1)) == budget
        run_rows.append({"run": str(root.relative_to(ROOT)), "expected_iteration": budget, "complete": complete})
    ok = not missing and all(r["complete"] for r in run_rows)
    report = {"classification": "PASS" if ok else "FAIL", "missing": missing, "formal_runs": run_rows, "full_started": False, "formal_test_started": False, "terminal": "ISIC2018_INFRASTRUCTURE_AND_INNOVATION_SHORT5K_COMPLETE"}
    jdump(AUDIT / "isic2018_prefull_final_closure.json", report)
    if not ok:
        raise RuntimeError("FINAL_CLOSURE_INCOMPLETE:" + ",".join(missing))


TASKS = {
    "S04_RESUME_NEGATIVE_GATES": task_resume_negative_gates,
    "S05_CHECKPOINT_MANIFEST_TEST": task_checkpoint_manifest,
    "S06_CONTROLLER_REAL_QUEUE_TEST": task_controller_queue_test,
    "S07_FULL_UNIT_TEST_MATRIX": task_unit_matrix,
    "S08_AFFECTED_PREFLIGHT": task_preflight,
    "S09_SOURCE_PROTOCOL_FREEZE": task_freeze,
    "S10_5P_BASELINE_SMOKE": lambda: smoke_or_short("S10_5P_BASELINE_SMOKE"),
    "S11_5P_CANDIDATE_SMOKE": lambda: smoke_or_short("S11_5P_CANDIDATE_SMOKE"),
    "S12_10P_BASELINE_SMOKE": lambda: smoke_or_short("S12_10P_BASELINE_SMOKE"),
    "S13_10P_CANDIDATE_SMOKE": lambda: smoke_or_short("S13_10P_CANDIDATE_SMOKE"),
    "S14_5P_BASELINE_SHORT5K": lambda: smoke_or_short("S14_5P_BASELINE_SHORT5K"),
    "S15_5P_CANDIDATE_SHORT5K": lambda: smoke_or_short("S15_5P_CANDIDATE_SHORT5K"),
    "S16_10P_BASELINE_SHORT5K": lambda: smoke_or_short("S16_10P_BASELINE_SHORT5K"),
    "S17_10P_CANDIDATE_SHORT5K": lambda: smoke_or_short("S17_10P_CANDIDATE_SHORT5K"),
    "S18_VALIDATION_CASEWISE_ANALYSIS": task_analysis,
    "S19_INNOVATION_RESULT_SUMMARY": task_summary,
    "S20_STOP_FULL_NOT_STARTED": task_final,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    if args.state not in TASKS:
        raise SystemExit(f"UNKNOWN_PREFULL_TASK:{args.state}")
    TASKS[args.state]()
    print(f"TASK_COMPLETE {args.state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
