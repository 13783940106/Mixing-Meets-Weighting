#!/usr/bin/env python3
"""Silent dynamic supervisor and post-run controller for the BSCE V1 full run."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/bsce_v1_full_20260718_1450"
LOG = AUDIT / "console.log"
STATE = AUDIT / "monitor_state.json"
TIMELINE = AUDIT / "dynamic_supervision_timeline.jsonl"
CURSOR = ROOT / "experiments/context_checkpoint.json"
PROJECT = ROOT / "experiments/project_state.json"
LOCK = ROOT / "experiments/.gpu_job.lock"
RUN = ROOT / "train_promise12/beta_FFT/PROMISE12_bcp_fft_test_seed_pb2d_original_2d_combo_mask_token_adamix_bsce_v1_full_20260718_1450_7/self_train"
LAUNCHER_PID = int(sys.argv[1])
INTERVAL_SECONDS = 15
BASE = 0.887704
TARGET = 0.897704

VAL_RE = re.compile(r"iteration (\d+) : model1_mean_dice : ([^\s]+)")
PRE_RE = re.compile(r"iteration (\d+): loss: ([^,\s]+), mix_dice")
SELF_RE = re.compile(r"iteration (\d+): loss: ([^,\s]+), model1_loss")
ENDPOINT_RE = re.compile(
    r"iteration (\d+) : use_bsce_v1 : True, bsce_v1_seam_inconsistency : ([^,\s]+), "
    r"bsce_v1_nonseam_inconsistency : ([^,\s]+), bsce_v1_counterfactual_composition_error : ([^,\s]+), "
    r"bsce_v1_seam_ratio : ([^,\s]+), bsce_v1_logit_gradient_norm : ([^,\s]+), "
    r"bsce_v1_finite : (True|False), total_finite : (True|False)"
)
ERROR_PATTERNS = {
    "traceback": r"Traceback \(most recent call last\)",
    "oom": r"out of memory|CUDA out of memory",
    "cuda_error": r"CUDA error|CUDNN_STATUS|NCCL error",
    "filesystem_error": r"No space left on device|Read-only file system|Input/output error",
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def finite_float(raw: str) -> float:
    try:
        value = float(raw.rstrip(","))
    except ValueError:
        return float("nan")
    return value


def process_rows() -> list[dict]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,args="], capture_output=True, text=True, check=False
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            rows.append({"pid": int(fields[0]), "ppid": int(fields[1]), "stat": fields[2], "args": fields[3]})
    return rows


def gpu_rows() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(",", 2)]
        if len(fields) == 3 and fields[0].isdigit():
            try:
                memory = int(fields[2])
            except ValueError:
                memory = None
            rows.append({"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": memory})
    return rows


def lock_held() -> bool:
    result = subprocess.run(["flock", "-n", str(LOCK), "true"], capture_output=True, check=False)
    return result.returncode != 0


def checkpoint_rows() -> list[dict]:
    rows = []
    if RUN.exists():
        for path in sorted(RUN.glob("*.pth")):
            stat = path.stat()
            rows.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def parse_snapshot(status: str) -> dict:
    text = LOG.read_text(errors="replace") if LOG.exists() else ""
    pre = [(int(i), finite_float(v)) for i, v in PRE_RE.findall(text)]
    self_loss = [(int(i), finite_float(v)) for i, v in SELF_RE.findall(text)]
    values = sorted({int(i): finite_float(v) for i, v in VAL_RE.findall(text)}.items())
    endpoint_by_iter = {}
    for match in ENDPOINT_RE.finditer(text):
        endpoint_by_iter[int(match.group(1))] = {
            "iteration": int(match.group(1)),
            "seam_inconsistency": finite_float(match.group(2)),
            "nonseam_inconsistency": finite_float(match.group(3)),
            "counterfactual_composition_error": finite_float(match.group(4)),
            "seam_ratio": finite_float(match.group(5)),
            "logit_gradient_norm": finite_float(match.group(6)),
            "bsce_finite": match.group(7) == "True",
            "total_finite": match.group(8) == "True",
        }
    endpoints = [endpoint_by_iter[k] for k in sorted(endpoint_by_iter)]
    best = max(values, key=lambda x: x[1]) if values else None
    latest = values[-1] if values else None
    last_values = [v for _, v in values[-3:]]
    strictly_decreasing_3 = len(last_values) == 3 and last_values[0] > last_values[1] > last_values[2]
    collapse_risk = bool(best and latest and best[1] - latest[1] >= 0.05 and strictly_decreasing_3)
    losses = [v for _, v in pre[-1:] + self_loss[-1:]]
    endpoint_numeric = []
    if endpoints:
        endpoint_numeric = [v for k, v in endpoints[-1].items() if k != "iteration" and isinstance(v, float)]
    nonfinite = any(not math.isfinite(v) for v in losses + [v for _, v in values] + endpoint_numeric)
    explicit_errors = {key: bool(re.search(pattern, text, re.I)) for key, pattern in ERROR_PATTERNS.items()}
    explicit_errors["nonfinite"] = nonfinite or bool(re.search(r"(^|[^A-Za-z])(NaN|Inf|-Inf)([^A-Za-z]|$)", text))
    rows = process_rows()
    trainers = [
        r for r in rows
        if re.match(r"^(?:python|/\S*/python) train_promise12_bcp_fft\.py ", r["args"])
    ]
    experiment_ids = set()
    for row in trainers:
        match = re.search(r"(?:^| )--exp ([^ ]+)", row["args"])
        experiment_ids.add(match.group(1) if match else f"UNKNOWN_PID_{row['pid']}")
    controllers = [
        r for r in rows
        if re.match(r"^(?:python|/\S*/python) scripts/bsce_v1_full_postcontroller\.py ", r["args"])
    ]
    gpus = gpu_rows()
    checkpoints = checkpoint_rows()
    checkpoint_names = {x["name"] for x in checkpoints}
    generic_best_present = "unet_best_model.pth" in checkpoint_names
    expected_best = None
    if best:
        matches = sorted(RUN.glob(f"model1_iter_{best[0]}_dice_*.pth"))
        expected_best = matches[-1].name if matches else None
    stage = "SELF_TRAIN" if self_loss else ("PRETRAIN" if pre else "INITIALIZING")
    conclusion = "NO_VALIDATION_YET"
    if values:
        if best == latest:
            conclusion = "CURRENT_BEST_REFRESHED"
        elif collapse_risk:
            conclusion = "POTENTIAL_SUSTAINED_DEGRADATION_REVIEW_ONLY"
        else:
            conclusion = "NORMAL_VALIDATION_OSCILLATION"
    return {
        "status": status,
        "timestamp": now(),
        "stage": stage,
        "launcher_pid": LAUNCHER_PID,
        "launcher_alive": alive(LAUNCHER_PID),
        "controller_pid": os.getpid(),
        "controller_count": len(controllers),
        "training_processes": trainers,
        "training_process_count": len(trainers),
        "gpu_processes": gpus,
        "gpu_process_count": len(gpus),
        "gpu_lock_held": lock_held(),
        "concurrent_experiment_ids": sorted(experiment_ids),
        "concurrent_experiment_detected": len(experiment_ids) > 1,
        "pretrain_latest_iter": max([i for i, _ in pre] or [0]),
        "self_train_latest_iter": max([i for i, _ in self_loss] or [0]),
        "latest_loss": (self_loss or pre or [None])[-1],
        "validation_count": len(values),
        "latest_val": list(latest) if latest else None,
        "best_val": list(best) if best else None,
        "delta_vs_fixed_base": best[1] - BASE if best else None,
        "delta_vs_keep_threshold": best[1] - TARGET if best else None,
        "validation_assessment": conclusion,
        "strictly_decreasing_last3": strictly_decreasing_3,
        "potential_collapse": collapse_risk,
        "error_scan": explicit_errors,
        "bad_pattern": any(explicit_errors.values()),
        "mechanism_endpoint_count": len(endpoints),
        "latest_mechanism_endpoint": endpoints[-1] if endpoints else None,
        "mechanism_all_finite": bool(endpoints) and all(
            x["bsce_finite"] and x["total_finite"] and all(
                math.isfinite(v) for k, v in x.items() if k != "iteration" and isinstance(v, float)
            ) for x in endpoints
        ),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "generic_best_present": generic_best_present,
        "explicit_current_best_checkpoint": expected_best,
    }


def append_timeline(data: dict, event: str) -> None:
    row = dict(data)
    row["event"] = event
    with TIMELINE.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sync_durable_cursor(data: dict) -> None:
    if CURSOR.exists():
        cursor = json.loads(CURSOR.read_text())
        cursor.update({
            "updated_at": data["timestamp"],
            "current_stage": "V13_4_BSCE_V1_DIRECT_30K_FULL_RUNNING",
            "active_training": "pb2d_original_2d_combo_mask_token_adamix_bsce_v1_full_20260718_1450",
            "active_stage": data["stage"],
            "active_latest_iter": data["self_train_latest_iter"] or data["pretrain_latest_iter"],
            "active_latest_val": data["latest_val"],
            "active_best_val": data["best_val"],
            "active_validation_assessment": data["validation_assessment"],
            "active_mechanism_finite": data["mechanism_all_finite"],
            "next_legal_action": "SILENT_DYNAMIC_SUPERVISION_TO_NATURAL_COMPLETION_THEN_AUTOMATIC_AUDIT",
        })
        cursor["gpu_job"] = {
            "launcher_pid": LAUNCHER_PID, "controller_pid": os.getpid(), "lock": "experiments/.gpu_job.lock",
            "log": "audit/bsce_v1_full_20260718_1450/console.log", "state": "audit/bsce_v1_full_20260718_1450/monitor_state.json",
        }
        atomic_json(CURSOR, cursor)
    if PROJECT.exists():
        project = json.loads(PROJECT.read_text())
        project.update({
            "active_training_latest_iter": data["self_train_latest_iter"] or data["pretrain_latest_iter"],
            "active_training_latest_val_dsc": data["latest_val"][1] if data["latest_val"] else None,
            "active_training_latest_val_iter": data["latest_val"][0] if data["latest_val"] else None,
            "active_training_best_val_dsc": data["best_val"][1] if data["best_val"] else None,
            "active_training_best_val_iter": data["best_val"][0] if data["best_val"] else None,
            "active_training_controller_pid": os.getpid(),
            "active_training_gpu_pid": data["gpu_processes"][0]["pid"] if len(data["gpu_processes"]) == 1 else None,
            "active_training_status": f"RUNNING_{data['stage']}_SILENT_DYNAMIC_SUPERVISION",
            "active_training_validation_assessment": data["validation_assessment"],
            "active_training_mechanism_finite": data["mechanism_all_finite"],
            "active_training_monitor_updated_at": data["timestamp"],
        })
        atomic_json(PROJECT, project)


last_val_count = -1
last_stage = None
last_checkpoint_signature = None
last_heartbeat = 0.0
while alive(LAUNCHER_PID):
    data = parse_snapshot("RUNNING")
    atomic_json(STATE, data)
    signature = [(x["name"], x["bytes"], x["mtime_ns"]) for x in data["checkpoints"]]
    event = None
    if data["validation_count"] != last_val_count:
        event = "VALIDATION_POINT" if last_val_count >= 0 else "SUPERVISOR_START_OR_RECOVERY"
    elif data["stage"] != last_stage:
        event = "STAGE_TRANSITION"
    elif signature != last_checkpoint_signature:
        event = "CHECKPOINT_CHANGE"
    elif time.time() - last_heartbeat >= 300:
        event = "HEARTBEAT"
    if event:
        append_timeline(data, event)
        sync_durable_cursor(data)
        last_heartbeat = time.time()
    last_val_count = data["validation_count"]
    last_stage = data["stage"]
    last_checkpoint_signature = signature
    time.sleep(INTERVAL_SECONDS)

exit_data = parse_snapshot("TRAIN_PROCESS_EXITED_AWAITING_POSTAUDIT")
atomic_json(STATE, exit_data)
append_timeline(exit_data, "TRAIN_PROCESS_EXITED")
sync_durable_cursor(exit_data)
time.sleep(5)
result = subprocess.run(
    ["conda", "run", "--no-capture-output", "-n", "beta_fft", "python", "audit/run_bsce_v1_full_postaudit.py"],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
(AUDIT / "postaudit_console.log").write_text(result.stdout)
final_status = "POSTAUDIT_COMPLETE" if result.returncode == 0 else "POSTAUDIT_FAILED"
final_data = parse_snapshot(final_status)
atomic_json(STATE, final_data)
append_timeline(final_data, final_status)
atomic_json(AUDIT / "final_runtime_consistency.json", {
    "timestamp": now(),
    "postaudit_returncode": result.returncode,
    "launcher_pid": LAUNCHER_PID,
    "launcher_alive": alive(LAUNCHER_PID),
    "controller_pid_during_finalization": os.getpid(),
    "controller_exit_expected_after_write": True,
    "gpu_processes_after_postaudit": gpu_rows(),
    "gpu_lock_held_after_launcher_exit": lock_held(),
    "concurrent_experiment_detected": final_data["concurrent_experiment_detected"],
    "monitor_status": final_status,
})
raise SystemExit(result.returncode)
