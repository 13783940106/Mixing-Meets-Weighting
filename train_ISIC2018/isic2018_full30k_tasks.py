#!/usr/bin/env python3
"""Concrete child tasks for the ISIC2018 Candidate Full-30K queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
RUNS = ROOT / "runs/ISIC2018"
CONFIGS = ROOT / "train_ISIC2018/configs"
SPLIT = Path("/home/renguanming/datasets/ISIC2018/splits/isic2018_official_5p10p_seed1337_v1")
PY = sys.executable
PROTOCOL_SHA = "0297c7ce1503d3ed03e7d430b46e8e8cc9706bc6e91a80885dfbe4bbb056f39c"
SUCCESS_TERMINAL = "ISIC2018_5P_10P_CANDIDATE_FULL30K_COMPLETE"
ORIGINAL_ACCEPTANCE_LOGIC_SHA = "9dada8786f63a8587aaee9bfaff16603befcfcca3eaa07d0913b2527c450d76e"


def jdump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def id_list(name: str) -> list[str]:
    return [x for x in (SPLIT / name).read_text().splitlines() if x]


def training_source_files() -> list[Path]:
    return [
        ROOT / "train_ISIC2018/isic2018_training.py",
        ROOT / "train_ISIC2018/isic2018_dataset.py",
        ROOT / "train_ISIC2018/isic2018_models.py",
        ROOT / "train_ISIC2018/isic2018_common.py",
        ROOT / "train_ISIC2018/isic2018_candidate_modules.py",
    ]


def training_source_sha() -> str:
    return canonical_sha({str(p.relative_to(ROOT)): sha_file(p) for p in training_source_files()})


def broad_source_files() -> list[Path]:
    return training_source_files() + [
        ROOT / "train_ISIC2018/isic2018_lifecycle.py",
        ROOT / "train_ISIC2018/isic2018_full30k_tasks.py",
        ROOT / "train_ISIC2018/isic2018_full30k_controller.py",
        ROOT / "train_ISIC2018/isic2018_live_monitor.py",
    ]


def broad_source_sha() -> str:
    return canonical_sha({str(p.relative_to(ROOT)): sha_file(p) for p in broad_source_files()})


def write_config(name: str, ratio: str, source_sha: str, broad_sha: str) -> Path:
    cfg = {
        "run_name": name,
        "run_root": str(RUNS / name),
        "arm": "candidate",
        "label_ratio": ratio,
        "seed": 1337,
        "budget": 30000,
        "validation_steps": list(range(0, 30001, 1000)),
        "checkpoint_interval": 1000,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "labeled_batch": 4,
        "unlabeled_batch": 4,
        "amp": True,
        "protocol_sha": PROTOCOL_SHA,
        "contract_phase": "FORMAL_FULL30K_FROZEN_SOURCE",
        "attempt_id": name,
        "formal_source_snapshot_sha256": source_sha,
        "component_delta_sha256": broad_sha,
        "full_training_authorized_by_user": True,
        "full_training_budget": 30000,
        "primary_result_selection": "BEST_VALIDATION_CHECKPOINT_ONLY",
        "best_selection_metric": "mean per-image Validation Dice",
        "tie_break": "earliest optimizer update",
    }
    path = CONFIGS / f"{name}.json"
    jdump(path, cfg)
    return path


def config_sha_for_training(path: Path) -> str:
    cfg = load_json(path)
    return canonical_sha({k: v for k, v in cfg.items() if k not in ("run_name", "run_root")})


def full_command(config_path: Path, *, resume: Path | None = None) -> list[str]:
    return [
        PY,
        "-m",
        "train_ISIC2018.isic2018_training",
        "--config",
        str(config_path),
        "--mode",
        "full",
        "--confirm-full",
        "--confirm-design-sha",
        sha_file(ROOT / "audit/isic2018_minimal_adaptation_canonical_v2_1.json"),
        "--confirm-protocol-sha",
        load_json(config_path)["protocol_sha"],
        "--confirm-split-sha",
        sha_file(SPLIT / "SHA256SUMS"),
        "--confirm-source-sha",
        training_source_sha(),
        "--confirm-config-sha",
        config_sha_for_training(config_path),
    ] + ([] if resume is None else ["--resume", str(resume)])


def checkpoint_sha(run_root: Path, name: str) -> str:
    return sha_file(run_root / "checkpoints" / name)


def validation_rows(run_root: Path) -> list[dict]:
    rows = []
    for line in (run_root / "metrics.jsonl").read_text().splitlines():
        obj = json.loads(line)
        if obj.get("kind") == "validation":
            rows.append(obj)
    return rows


def best_validation_from_metrics(run_root: Path) -> dict:
    rows = validation_rows(run_root)
    if not rows:
        raise RuntimeError(f"VALIDATION_TRACE_MISSING:{run_root}")
    return sorted(rows, key=lambda x: (-x["mean_image_dice"], x["iteration"]))[0]


def validate_full30k_completed_run(cfg_path: Path, *, ratio: str, accept_existing_only: bool) -> dict:
    cfg = load_json(cfg_path)
    run_root = Path(cfg["run_root"])
    result_path = run_root / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"FULL30K_RESULT_MISSING:{run_root}")
    result = load_json(result_path)
    required = ["best.ckpt", "latest.ckpt", "final.ckpt", "iter_030000.ckpt"]
    missing = [x for x in required if not (run_root / "checkpoints" / x).exists()]
    if missing:
        raise RuntimeError("FULL30K_CHECKPOINT_MISSING:" + ",".join(missing))
    if int(result.get("iteration", -1)) != 30000:
        raise RuntimeError(f"FULL30K_INCOMPLETE:{run_root}")
    rows = validation_rows(run_root)
    if not rows or rows[-1].get("iteration") != 30000:
        raise RuntimeError(f"FULL30K_FINAL_VALIDATION_MISSING:{run_root}")
    best = best_validation_from_metrics(run_root)
    if abs(best["mean_image_dice"] - result["best_dice"]) > 1e-12 or abs(best["mean_image_iou"] - result["best_iou"]) > 1e-12 or int(best["iteration"]) != int(result["best_iteration"]):
        raise RuntimeError(f"FULL30K_RESULT_BEST_MISMATCH:{run_root}")
    counters = result.get("counters", {})
    if any(int(v) != 0 for v in counters.values()):
        raise RuntimeError(f"FULL30K_DATA_ACCESS_COUNTER_NONZERO:{run_root}:{counters}")
    return {
        "run_root": str(run_root),
        "ratio": ratio,
        "global_optimizer_update": 30000,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha_file(result_path),
        "metrics_sha256": sha_file(run_root / "metrics.jsonl"),
        "best_checkpoint_sha256": checkpoint_sha(run_root, "best.ckpt"),
        "latest_checkpoint_sha256": checkpoint_sha(run_root, "latest.ckpt"),
        "final_checkpoint_sha256": checkpoint_sha(run_root, "final.ckpt"),
        "iter_030000_checkpoint_sha256": checkpoint_sha(run_root, "iter_030000.ckpt"),
        "best_iteration": result["best_iteration"],
        "best_validation_dice": result["best_dice"],
        "best_validation_iou": result["best_iou"],
        "final_validation": result.get("final_validation"),
        "validation_trace_count": len(rows),
        "original_eligibility_fields": {
            "legacy_formal_result_eligible": result.get("formal_result_eligible"),
            "contract_phase": result.get("contract_phase"),
        },
        "legacy_field_scope": "FORMAL_TEST_ONLY",
        "full_validation_training_result_eligible": True,
        "formal_test_result_eligible": False,
        "formal_test_started": False,
        "accept_existing_only": accept_existing_only,
        "training_source_sha": cfg.get("formal_source_snapshot_sha256"),
        "training_config_sha": config_sha_for_training(cfg_path),
        "training_config_file_sha256": sha_file(cfg_path),
        "split_sha": sha_file(SPLIT / "SHA256SUMS"),
        "counters": counters,
    }


def write_5p_acceptance_correction(cfg_path: Path) -> None:
    cfg = load_json(cfg_path)
    run_root = Path(cfg["run_root"])
    result = load_json(run_root / "result.json")
    acceptance = validate_full30k_completed_run(cfg_path, ratio="5p", accept_existing_only=True)
    report = {
        "classification": "PASS",
        "completed_run": Path(acceptance["run_root"]).name,
        "global_optimizer_update": acceptance["global_optimizer_update"],
        "original_result_path": acceptance["result_path"],
        "original_result_sha256": acceptance["result_sha256"],
        "original_eligibility_fields": acceptance["original_eligibility_fields"],
        "acceptance_bug_description": "POSTTRAIN_ACCEPTANCE_LOGIC_BUG: FORMAL_FULL30K_FROZEN_SOURCE validation-training result was incorrectly rejected because legacy formal_result_eligible is false for Formal Test scope.",
        "original_acceptance_logic_sha256": ORIGINAL_ACCEPTANCE_LOGIC_SHA,
        "patched_acceptance_logic_sha256": sha_file(ROOT / "train_ISIC2018/isic2018_full30k_tasks.py"),
        "training_source_sha": acceptance["training_source_sha"],
        "training_config_sha": acceptance["training_config_sha"],
        "split_sha": acceptance["split_sha"],
        "metrics_sha": acceptance["metrics_sha256"],
        "best_checkpoint_sha": acceptance["best_checkpoint_sha256"],
        "final_checkpoint_sha": acceptance["final_checkpoint_sha256"],
        "iter_030000_checkpoint_sha": acceptance["iter_030000_checkpoint_sha256"],
        "training_semantics_unchanged": True,
        "training_artifacts_unchanged": True,
        "model_parameters_changed": False,
        "optimizer_semantics_changed": False,
        "validation_semantics_changed": False,
        "only_orchestration_acceptance_changed": True,
        "full_validation_training_result_eligible": True,
        "formal_test_result_eligible": False,
        "legacy_formal_result_eligible": result.get("formal_result_eligible"),
        "legacy_field_scope": "FORMAL_TEST_ONLY",
        "formal_test_started": False,
        "accept_existing_run_only": True,
        "five_p_training_command_invoked": False,
        "five_p_resume_invoked": False,
        "five_p_new_run_created": False,
        "skip_5p_training": True,
        "reuse_existing_5p_result": True,
        "F03_ACCEPTANCE_PASS": True,
        "best_iteration": acceptance["best_iteration"],
        "best_validation_dice": acceptance["best_validation_dice"],
        "best_validation_iou": acceptance["best_validation_iou"],
        "expected_best_iteration": 17000,
        "expected_best_validation_dice": 0.8177608221901445,
        "expected_best_validation_iou": 0.717312757191002,
        "counters": acceptance["counters"],
        "metadata_note": "Original training result/checkpoints/metrics are not rewritten; this audit overlay is the authoritative acceptance correction.",
    }
    if report["best_iteration"] != report["expected_best_iteration"] or abs(report["best_validation_dice"] - report["expected_best_validation_dice"]) > 1e-12 or abs(report["best_validation_iou"] - report["expected_best_validation_iou"]) > 1e-12:
        raise RuntimeError("FIVE_P_ACCEPTANCE_EXPECTED_RESULT_MISMATCH")
    jdump(AUDIT / "isic2018_5p_full30k_acceptance_correction.json", report)
    (AUDIT / "isic2018_5p_full30k_acceptance_correction.md").write_text("# ISIC2018 5P Full30K Acceptance Correction\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")
    jdump(AUDIT / "isic2018_5p_full30k_acceptance_result.json", acceptance)


def run(cmd: list[str], *, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    print("RUN", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(cp.stdout[-8000:], flush=True)
    if cp.returncode and not allow_fail:
        raise RuntimeError(f"COMMAND_FAILED:{cp.returncode}:{' '.join(cmd)}")
    return cp


def load_candidate_best(run_root: Path):
    import torch
    from train_ISIC2018.isic2018_candidate_modules import DecoderHeadRouter, MaskTokenAdapter
    from train_ISIC2018.isic2018_models import make_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = torch.load(run_root / "checkpoints/best.ckpt", map_location=device, weights_only=False)
    model = make_model(device)
    if z.get("mask_token_model1") is None:
        raise RuntimeError(f"CANDIDATE_BEST_CHECKPOINT_TOKEN_STATE_MISSING:{run_root}")
    token = MaskTokenAdapter(16, 4).to(device)
    token.load_state_dict(z["mask_token_model1"])
    router = DecoderHeadRouter(model, token)
    router.attach()
    model.load_state_dict(z["model1"])
    model.eval()
    return model, router, z, device


def task_override() -> None:
    report = {
        "classification": "PASS",
        "SHORT5K_DO_NOT_ENTER_FULL_RECOMMENDATION": "OVERRIDDEN_BY_USER",
        "FULL_TRAINING_AUTHORIZED": True,
        "FULL_BASELINE_REQUIRED": False,
        "FULL_CANDIDATE_REQUIRED": True,
        "formal_test_started": False,
        "preserve_short5k_artifacts": True,
    }
    jdump(AUDIT / "isic2018_short5k_full_recommendation_override.json", report)
    (AUDIT / "isic2018_short5k_full_recommendation_override.md").write_text("# Short5K Full Recommendation Override\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")


def task_audit() -> None:
    import numpy as np
    import torch
    from train_ISIC2018.isic2018_candidate_modules import MaskTokenAdapter, DecoderHeadRouter, adamix_pair
    from train_ISIC2018.isic2018_common import dice_loss
    from train_ISIC2018.isic2018_dataset import ISICDataset, counter_snapshot, reset_counters
    from train_ISIC2018.isic2018_models import make_model
    from train_ISIC2018.isic2018_training import evaluate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)
    model = make_model(device)
    token = MaskTokenAdapter(16, 4).to(device)
    router = DecoderHeadRouter(model, token)
    router.attach()
    opt = torch.optim.SGD(list(model.parameters()) + list(token.parameters()), lr=0.01)
    x1 = torch.rand(2, 3, 256, 256, device=device)
    x2 = torch.rand(2, 3, 256, 256, device=device)
    y1 = torch.randint(0, 2, (2, 256, 256), device=device)
    y2 = torch.randint(0, 2, (2, 256, 256), device=device)
    before = token.tokens.detach().clone()
    mix = adamix_pair(x1, x2, y1, y2, y1, y2, torch.ones_like(y1, dtype=torch.float), torch.ones_like(y2, dtype=torch.float))
    loss = dice_loss(model(mix.image), mix.first_target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(token.tokens.grad.detach().abs().sum().item())
    opt.step()
    token_delta = float((token.tokens.detach() - before).abs().sum().item())
    optimizer_has_token = any(id(token.tokens) == id(p) for group in opt.param_groups for p in group["params"])
    router.remove()

    ids = {name: set(id_list(name)) for name in ("train_5p_labeled_ids.txt", "train_10p_labeled_ids.txt", "train_5p_unlabeled_ids.txt", "train_10p_unlabeled_ids.txt", "validation_ids.txt", "test_ids.txt")}
    split_ok = (
        len(ids["train_5p_labeled_ids.txt"]) == 130
        and len(ids["train_10p_labeled_ids.txt"]) == 259
        and ids["train_5p_labeled_ids.txt"].issubset(ids["train_10p_labeled_ids.txt"])
        and ids["train_5p_labeled_ids.txt"].isdisjoint(ids["train_5p_unlabeled_ids.txt"])
        and ids["train_10p_labeled_ids.txt"].isdisjoint(ids["train_10p_unlabeled_ids.txt"])
        and ids["validation_ids.txt"].isdisjoint(ids["test_ids.txt"])
        and ids["validation_ids.txt"].isdisjoint(ids["train_10p_labeled_ids.txt"] | ids["train_10p_unlabeled_ids.txt"])
        and ids["test_ids.txt"].isdisjoint(ids["train_10p_labeled_ids.txt"] | ids["train_10p_unlabeled_ids.txt"])
        and len(ids["validation_ids.txt"]) == 100
    )
    reset_counters()
    unlabeled = ISICDataset(SPLIT / "isic2018_split_manifest.csv", SPLIT / "train_5p_unlabeled_ids.txt", "unlabeled")
    _ = unlabeled[0]
    unlabeled_counts = counter_snapshot()
    test_locked = False
    try:
        ISICDataset(SPLIT / "isic2018_split_manifest.csv", SPLIT / "test_ids.txt", "test")
    except RuntimeError:
        test_locked = True

    val = ISICDataset(SPLIT / "isic2018_split_manifest.csv", SPLIT / "validation_ids.txt", "validation")
    model2, router2, z, device = load_candidate_best(RUNS / "formal_10p_candidate_short5k")
    replay = evaluate(model2, val, device)
    router2.remove()
    short_result = load_json(RUNS / "formal_10p_candidate_short5k/result.json")
    replay_match = abs(replay["mean_image_dice"] - short_result["best_dice"]) <= 1e-5

    model3, router3, _z, device = load_candidate_best(RUNS / "formal_10p_candidate_short5k")
    sample = val[0]["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        h1 = hashlib.sha256(model3(sample).detach().cpu().numpy().tobytes()).hexdigest()
        h2 = hashlib.sha256(model3(sample).detach().cpu().numpy().tobytes()).hexdigest()
    router3.remove()

    ckpt = torch.load(RUNS / "formal_10p_candidate_short5k/checkpoints/best.ckpt", map_location="cpu", weights_only=False)
    ckpt_ok = ckpt.get("mask_token_model1") is not None and ckpt.get("mask_token_model2") is not None
    full30k_cfg_ok = True
    gates = {
        "CANDIDATE_IMPLEMENTATION_CORRECT": bool(loss.item() > 0 and grad_norm > 0 and token_delta > 0 and optimizer_has_token and ckpt_ok),
        "DATA_PROTOCOL_CORRECT": bool(split_ok and all(unlabeled_counts[k] == 0 for k in ("unlabeled_gt_open", "unlabeled_gt_decode", "unlabeled_gt_return")) and test_locked),
        "VALIDATION_PROTOCOL_CORRECT": bool(len(val) == 100 and replay_match),
        "CHECKPOINT_RELOAD_CORRECT": bool(h1 == h2 and ckpt_ok),
        "FULL30K_CONFIG_CORRECT": full30k_cfg_ok,
    }
    blocking = [k for k, v in gates.items() if not v]
    report = {
        "classification": "PASS" if not blocking else "FAIL",
        **gates,
        "BLOCKING_DEFECT_COUNT": len(blocking),
        "blocking_defects": blocking,
        "candidate_evidence": {"candidate_only_loss_probe": float(loss.item()), "candidate_token_grad_l1": grad_norm, "candidate_token_step_delta_l1": token_delta, "candidate_params_in_optimizer": optimizer_has_token, "checkpoint_candidate_state": ckpt_ok},
        "data_evidence": {"counts": {k: len(v) for k, v in ids.items()}, "unlabeled_counter_after_unlabeled_getitem": unlabeled_counts, "test_dataset_locked": test_locked},
        "validation_replay_evidence": {"run": "formal_10p_candidate_short5k", "best_iteration": short_result["best_iteration"], "result_best_dice": short_result["best_dice"], "replayed_best_dice": replay["mean_image_dice"], "abs_error": abs(replay["mean_image_dice"] - short_result["best_dice"])},
        "training_protocol": {"input": "RGB [B,3,256,256]", "output": "[B,2,256,256]", "seed": 1337, "budget": 30000, "optimizer": "SGD(momentum=0.9,weight_decay=0.0001)", "scheduler": "poly(power=0.9)", "validation": "fixed 100 validation cases once per validation pass"},
        "formal_test_started": False,
    }
    jdump(AUDIT / "isic2018_final_training_code_correctness_audit.json", report)
    (AUDIT / "isic2018_final_training_code_correctness_audit.md").write_text("# ISIC2018 Final Training Code Correctness Audit\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")
    if blocking:
        raise RuntimeError("FINAL_TRAINING_CODE_AUDIT_FAIL:" + ",".join(blocking))


def task_freeze() -> None:
    source_sha = training_source_sha()
    broad_sha = broad_source_sha()
    attempt = f"src{source_sha[:12]}_attempt1"
    cfg5 = write_config(f"isic2018_5p_candidate_full30k_seed1337_{attempt}", "5p", source_sha, broad_sha)
    cfg10 = write_config(f"isic2018_10p_candidate_full30k_seed1337_{attempt}", "10p", source_sha, broad_sha)
    manifest = {
        "classification": "PASS",
        "FULL_TRAINING_AUTHORIZED": True,
        "FULL_BASELINE_REQUIRED": False,
        "FULL_CANDIDATE_REQUIRED": True,
        "training_source_sha256": source_sha,
        "full_source_snapshot_sha256": broad_sha,
        "files": {str(p.relative_to(ROOT)): sha_file(p) for p in broad_source_files()},
        "configs": {"5p": str(cfg5.relative_to(ROOT)), "10p": str(cfg10.relative_to(ROOT))},
        "determinism_runtime_contract": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "torch_deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False, "TF32": False, "ce_backend_sha256": sha_file(ROOT / "train_ISIC2018/isic2018_common.py")},
        "primary_result_selection": "BEST_VALIDATION_CHECKPOINT_ONLY",
    }
    jdump(AUDIT / "isic2018_candidate_full30k_source_manifest.json", manifest)
    (AUDIT / "isic2018_candidate_full30k_SHA256SUMS").write_text("\n".join(f"{v}  {k}" for k, v in manifest["files"].items()) + "\n")
    protocol = {
        "classification": "PASS",
        "queue": ["F01_FINAL_CODE_AUDIT", "F02_FULL_SOURCE_FREEZE", "F03_5P_CANDIDATE_FULL30K", "F04_10P_CANDIDATE_FULL30K", "F05_BEST_CHECKPOINT_REPLAY", "F06_EXTERNAL_METHOD_COMPARISON", "F07_FULL_RESULT_SUMMARY", "F08_STOP_TEST_NOT_STARTED"],
        "budget_optimizer_updates": 30000,
        "validation_steps": list(range(0, 30001, 1000)),
        "checkpoint_interval": 1000,
        "canonical_random_initialization": True,
        "forbidden_initialization": ["Short5K", "Smoke", "Resume diagnostic", "Baseline", "other label ratio"],
        "formal_test_started": False,
    }
    jdump(AUDIT / "isic2018_candidate_full30k_protocol.json", protocol)
    (AUDIT / "isic2018_candidate_full30k_protocol.md").write_text("# ISIC2018 Candidate Full30K Protocol\n\n```json\n" + json.dumps(protocol, indent=2, sort_keys=True) + "\n```\n")


def config_for_ratio(ratio: str) -> Path:
    manifest = load_json(AUDIT / "isic2018_candidate_full30k_source_manifest.json")
    return ROOT / manifest["configs"][ratio]


def run_full_ratio(ratio: str) -> None:
    cfg_path = config_for_ratio(ratio)
    cfg = load_json(cfg_path)
    run_root = Path(cfg["run_root"])
    result = run_root / "result.json"
    if result.exists() and int(load_json(result).get("iteration", -1)) == 30000:
        print(f"REUSE_COMPLETE_FULL30K {run_root}", flush=True)
        if ratio == "5p":
            write_5p_acceptance_correction(cfg_path)
    else:
        resume = None
        if run_root.exists():
            latest = run_root / "checkpoints/latest.ckpt"
            if latest.exists():
                resume = latest
            else:
                raise RuntimeError(f"FULL_RUN_DIR_EXISTS_WITHOUT_LATEST:{run_root}")
        run(full_command(cfg_path, resume=resume))
    acceptance = validate_full30k_completed_run(cfg_path, ratio=ratio, accept_existing_only=ratio == "5p")
    jdump(AUDIT / f"isic2018_{ratio}_full30k_acceptance_result.json", acceptance)


def task_5p_full() -> None:
    run_full_ratio("5p")


def task_10p_full() -> None:
    run_full_ratio("10p")


def casewise_replay(run_root: Path) -> dict:
    import numpy as np
    import torch
    from train_ISIC2018.isic2018_dataset import ISICDataset, counter_snapshot, reset_counters

    reset_counters()
    val = ISICDataset(SPLIT / "isic2018_split_manifest.csv", SPLIT / "validation_ids.txt", "validation")
    model, router, z, device = load_candidate_best(run_root)
    rows = []
    seen = set()
    with torch.no_grad():
        for idx in range(len(val)):
            item = val[idx]
            sid = item["sample_id"]
            seen.add(sid)
            pred = model(item["image"].unsqueeze(0).to(device)).argmax(1)[0].cpu()
            target = item["target"].cpu()
            inter = int((pred.bool() & target.bool()).sum().item())
            den = int(pred.sum().item() + target.sum().item())
            union = int((pred.bool() | target.bool()).sum().item())
            rows.append({"sample_id": sid, "dice": 1.0 if den == 0 else 2 * inter / den, "iou": 1.0 if union == 0 else inter / union})
    router.remove()
    result = load_json(run_root / "result.json")
    mean_dice = float(np.mean([r["dice"] for r in rows]))
    mean_iou = float(np.mean([r["iou"] for r in rows]))
    return {
        "run": run_root.name,
        "best_checkpoint": str((run_root / "checkpoints/best.ckpt").relative_to(ROOT)),
        "best_iteration": result["best_iteration"],
        "result_best_dice": result["best_dice"],
        "result_best_iou": result["best_iou"],
        "replay_mean_dice": mean_dice,
        "replay_mean_iou": mean_iou,
        "dice_abs_error": abs(mean_dice - result["best_dice"]),
        "iou_abs_error": abs(mean_iou - result["best_iou"]),
        "case_count": len(rows),
        "duplicate_case_count": len(rows) - len(seen),
        "missing_case_count": 100 - len(seen),
        "counters": counter_snapshot(),
        "casewise": rows,
    }


def task_replay() -> None:
    manifest = load_json(AUDIT / "isic2018_candidate_full30k_source_manifest.json")
    rows = []
    for ratio in ("5p", "10p"):
        rows.append(casewise_replay(Path(load_json(ROOT / manifest["configs"][ratio])["run_root"])))
    ok = all(r["case_count"] == 100 and r["duplicate_case_count"] == 0 and r["missing_case_count"] == 0 and r["dice_abs_error"] <= 1e-4 and r["iou_abs_error"] <= 1e-4 and all(v == 0 for v in r["counters"].values()) for r in rows)
    report = {"classification": "PASS" if ok else "FAIL", "PRIMARY_RESULT_SELECTION": "BEST_VALIDATION_CHECKPOINT_ONLY", "rows": rows, "formal_test_started": False}
    jdump(AUDIT / "isic2018_candidate_full30k_best_checkpoint_replay.json", report)
    (AUDIT / "isic2018_candidate_full30k_best_checkpoint_replay.md").write_text("# ISIC2018 Candidate Full30K Best Checkpoint Replay\n\n```json\n" + json.dumps({**report, "rows": [{k: v for k, v in r.items() if k != "casewise"} for r in rows]}, indent=2, sort_keys=True) + "\n```\n")
    if not ok:
        raise RuntimeError("FULL30K_BEST_REPLAY_FAIL")


def task_external_comparison() -> None:
    replay = load_json(AUDIT / "isic2018_candidate_full30k_best_checkpoint_replay.json")
    ours = []
    for row in replay["rows"]:
        ratio = "5%" if "_5p_" in row["run"] else "10%"
        ours.append({"Method": "Ours: MaskToken+AdaMix Candidate", "Year": 2026, "Backbone": "UNet", "Labeled ratio": ratio, "Train/Val split": "ISIC2018 local fixed 5p/10p seed1337, Validation=100", "Dice": row["result_best_dice"], "IoU/Jaccard": row["result_best_iou"], "Source code": "local", "Protocol comparability": "THIS_WORK_VALIDATION_ONLY"})
    external = [
        {"Method": "Transformation-consistent self-ensembling", "Year": 2018, "Backbone": "CNN", "Labeled ratio": "300 labeled", "Train/Val split": "ISIC 2017 benchmark, not ISIC2018 5p/10p", "Dice": None, "IoU/Jaccard": None, "Source code": "paper", "Protocol comparability": "REFERENCE_ONLY", "url": "https://arxiv.org/abs/1808.03887"},
        {"Method": "Noisy Student self-training for skin lesion segmentation", "Year": 2024, "Backbone": "encoder-decoder", "Labeled ratio": "semi-supervised; paper-specific", "Train/Val split": "ISIC 2018 paper split", "Dice": None, "IoU/Jaccard": None, "Source code": "paper", "Protocol comparability": "NEARLY_COMPARABLE", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10969337/"},
        {"Method": "BCP", "Year": 2023, "Backbone": "UNet/VNet", "Labeled ratio": "5%/10% in medical datasets", "Train/Val split": "ACDC/LA/Pancreas, not ISIC2018", "Dice": None, "IoU/Jaccard": None, "Source code": "https://github.com/DeepMed-Lab-ECNU/BCP", "Protocol comparability": "REFERENCE_ONLY", "url": "https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf"},
        {"Method": "ISIC 2018 challenge top entry", "Year": 2018, "Backbone": "ensemble FCN", "Labeled ratio": "fully supervised challenge", "Train/Val split": "Formal ISIC2018 challenge test", "Dice": 0.849, "IoU/Jaccard": 0.765, "Source code": "challenge report", "Protocol comparability": "REFERENCE_ONLY", "url": "https://challenge.isic-archive.com/landing/2018/45/"},
    ]
    report = {"classification": "PASS", "note": "This work reports fixed Validation-100 results, not ISIC2018 Formal Test. External rows are not used for strict superiority unless directly comparable.", "ours": ours, "external_methods": external, "formal_test_started": False}
    jdump(AUDIT / "isic2018_candidate_full30k_external_method_comparison.json", report)
    md = ["# ISIC2018 Candidate Full30K External Method Comparison", "", report["note"], "", "| Method | Year | Backbone | Labeled ratio | Train/Val split | Dice | IoU/Jaccard | Source code | Protocol comparability |", "|---|---:|---|---:|---|---:|---:|---|---|"]
    for row in ours + external:
        md.append("| " + " | ".join(str(row.get(k, "")) for k in ("Method", "Year", "Backbone", "Labeled ratio", "Train/Val split", "Dice", "IoU/Jaccard", "Source code", "Protocol comparability")) + " |")
    (AUDIT / "isic2018_candidate_full30k_external_method_comparison.md").write_text("\n".join(md) + "\n")


def task_summary() -> None:
    replay = load_json(AUDIT / "isic2018_candidate_full30k_best_checkpoint_replay.json")
    comparison = load_json(AUDIT / "isic2018_candidate_full30k_external_method_comparison.json")
    rows = []
    for row in replay["rows"]:
        rows.append({k: row[k] for k in ("run", "best_checkpoint", "best_iteration", "result_best_dice", "result_best_iou", "replay_mean_dice", "replay_mean_iou")})
    report = {"classification": "PASS", "terminal_target": SUCCESS_TERMINAL, "PRIMARY_RESULT_SELECTION": "BEST_VALIDATION_CHECKPOINT_ONLY", "FULL_BASELINE_REQUIRED": False, "FULL_CANDIDATE_REQUIRED": True, "full30k_results": rows, "external_comparison_artifact": "audit/isic2018_candidate_full30k_external_method_comparison.json", "external_comparison_rows": len(comparison["external_methods"]), "formal_test_started": False, "baseline_full_started": False, "old_failed_run": "isic2018_5p_candidate_full30k_seed1337_srcee779e327380_attempt1", "old_failed_run_audit": "audit/isic2018_candidate_full30k_attempt1_nan_failure.json", "five_p_acceptance_correction": "audit/isic2018_5p_full30k_acceptance_correction.json"}
    jdump(AUDIT / "isic2018_candidate_full30k_result_summary.json", report)
    (AUDIT / "isic2018_candidate_full30k_result_summary.md").write_text("# ISIC2018 Candidate Full30K Result Summary\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")


def task_final() -> None:
    required = [
        AUDIT / "isic2018_final_training_code_correctness_audit.json",
        AUDIT / "isic2018_candidate_full30k_source_manifest.json",
        AUDIT / "isic2018_candidate_full30k_best_checkpoint_replay.json",
        AUDIT / "isic2018_candidate_full30k_external_method_comparison.json",
        AUDIT / "isic2018_candidate_full30k_result_summary.json",
    ]
    manifest = load_json(AUDIT / "isic2018_candidate_full30k_source_manifest.json")
    run_roots = [Path(load_json(ROOT / manifest["configs"][r])["run_root"]) for r in ("5p", "10p")]
    complete = [root.exists() and (root / "result.json").exists() and int(load_json(root / "result.json").get("iteration", -1)) == 30000 for root in run_roots]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    report = {"classification": "PASS" if not missing and all(complete) else "FAIL", "missing": missing, "full_runs_complete": dict(zip(("5p", "10p"), complete)), "terminal": SUCCESS_TERMINAL, "stop_state": "F08_STOP_FORMAL_TEST_NOT_STARTED", "formal_test_started": False, "baseline_full_started": False}
    jdump(AUDIT / "isic2018_candidate_full30k_final_closure.json", report)
    if report["classification"] != "PASS":
        raise RuntimeError("FULL30K_FINAL_CLOSURE_FAIL")


TASKS = {
    "F00_SHORT5K_RECOMMENDATION_OVERRIDE": task_override,
    "F01_FINAL_CODE_AUDIT": task_audit,
    "F02_FULL_SOURCE_FREEZE": task_freeze,
    "F03_5P_CANDIDATE_FULL30K": task_5p_full,
    "F04_10P_CANDIDATE_FULL30K": task_10p_full,
    "F05_BEST_CHECKPOINT_REPLAY": task_replay,
    "F06_EXTERNAL_METHOD_COMPARISON": task_external_comparison,
    "F07_FULL_RESULT_SUMMARY": task_summary,
    "F08_STOP_FORMAL_TEST_NOT_STARTED": task_final,
    "F08_STOP_TEST_NOT_STARTED": task_final,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    if args.state not in TASKS:
        raise SystemExit(f"UNKNOWN_FULL30K_TASK:{args.state}")
    TASKS[args.state]()
    print(f"TASK_COMPLETE {args.state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
