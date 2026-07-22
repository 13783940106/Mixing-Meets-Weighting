#!/usr/bin/env python3
"""Detached dynamic controller for one unique V15.1 run root."""

from __future__ import annotations

import argparse, json, os, re, signal, subprocess, time
from datetime import datetime
from pathlib import Path

ITERATION = re.compile(r"iteration[ =](\d+)", re.I)
VALIDATION = re.compile(r"VALIDATION iteration=(\d+).*?mDice=([0-9.]+)")
ERRORS = re.compile(r"NaN|\bInf\b|OOM|out of memory|Traceback|CUDA error|non-finite|MODE_LIFECYCLE_REGRESSION", re.I)
MODES = re.compile(r"model1_mode=(TRAIN|EVAL).*?model2_mode=(TRAIN|EVAL).*?teacher_mode=(TRAIN|EVAL)")
POST_MODES = re.compile(r"post_validation_model1_mode=(TRAIN|EVAL).*?post_validation_model2_mode=(TRAIN|EVAL).*?post_validation_teacher_mode=(TRAIN|EVAL)")


def alive(pid):
    try: os.kill(pid, 0); return True
    except (ProcessLookupError, PermissionError): return False


def command(pid):
    try: return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError: return ""


def gpu_rows():
    query = "pid,used_gpu_memory"
    result = subprocess.run(["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def training_has_gpu_device(pid):
    fd_root = Path(f"/proc/{pid}/fd")
    try:
        return any(os.readlink(path).startswith("/dev/nvidia") for path in fd_root.iterdir())
    except (OSError, PermissionError):
        return False


def process_matches(token, monitor=False):
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
    marker = "v15_1_mscmr_monitor.py" if monitor else ("v15_1_mscmr_pretrain.py", "v15_1_mscmr_beta_fft_baseline.py")
    rows = []
    for line in result.stdout.splitlines():
        if token not in line: continue
        if (marker in line) if isinstance(marker, str) else any(item in line for item in marker):
            rows.append(line.strip())
    return rows


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run_root", required=True); p.add_argument("--training_pid", type=int, required=True)
    p.add_argument("--max_iterations", type=int, required=True); p.add_argument("--validation_interval", type=int, required=True)
    p.add_argument("--heartbeat_seconds", type=int, default=30); p.add_argument("--pretrain_val", type=float)
    p.add_argument("--collapse_pp", type=float, default=15.0); args = p.parse_args()
    root = Path(args.run_root).resolve(); state_path = root / "monitor_state.json"; timeline_path = root / "monitor_timeline.jsonl"
    pid_path = root / "controller.pid"; pid_path.write_text(f"{os.getpid()}\n")
    previous_state = {}
    if state_path.exists():
        try: previous_state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError): previous_state = {}
    previous_validation = previous_state.get("last_validation")
    last_validation_iteration = int(previous_validation[0]) if previous_validation else -1
    last_heartbeat = 0.; started = datetime.now().astimezone().isoformat()
    with timeline_path.open("a", buffering=1) as timeline:
        while True:
            now = time.time(); text = (root / "console.log").read_text(errors="replace") if (root / "console.log").exists() else ""
            iterations = [int(v) for v in ITERATION.findall(text)]; current = max(iterations, default=0)
            validations = [(int(i), float(v)) for i, v in VALIDATION.findall(text)]
            best = max(validations, key=lambda pair: pair[1]) if validations else None
            training_alive = alive(args.training_pid); training_cmd = command(args.training_pid)
            training_matches = process_matches(str(root), monitor=False); controller_matches = process_matches(str(root), monitor=True)
            gpu = gpu_rows(); gpu_device_bound = training_has_gpu_device(args.training_pid)
            errors = sorted(set(match.group(0) for match in ERRORS.finditer(text)))
            modes = MODES.findall(text); post_modes = POST_MODES.findall(text)
            latest_modes = list(modes[-1]) if modes else None; latest_post_modes = list(post_modes[-1]) if post_modes else None
            mode_lifecycle_ok = latest_modes in (None, ["TRAIN", "TRAIN", "EVAL"]) and latest_post_modes in (None, ["TRAIN", "TRAIN", "EVAL"])
            collapse_threshold = args.pretrain_val - args.collapse_pp / 100.0 if args.pretrain_val is not None else None
            last_three = validations[-3:]
            recovering = len(last_three) == 3 and last_three[0][1] < last_three[1][1] < last_three[2][1]
            collapse_sequence = last_three if collapse_threshold is not None and len(last_three) == 3 and not recovering and all(value < collapse_threshold for _, value in last_three) else []
            checkpoints = sorted(str(path.name) for path in (root / "checkpoints").glob("*.ckpt")) if (root / "checkpoints").exists() else []
            state = {"run_root": str(root), "controller_pid": os.getpid(), "controller_command": command(os.getpid()),
                "training_pid": args.training_pid, "training_command": training_cmd, "training_alive": training_alive,
                "controller_alive": True, "detached_controller_exit_does_not_signal_training": True,
                "started_at": started, "heartbeat_at": datetime.now().astimezone().isoformat(), "heartbeat_seconds": args.heartbeat_seconds,
                "last_iteration": current, "max_iterations": args.max_iterations, "validation_interval": args.validation_interval,
                "last_validation": list(validations[-1]) if validations else None, "best_validation": list(best) if best else None,
                "next_validation_iteration": ((current // args.validation_interval) + 1) * args.validation_interval,
                "checkpoint_files": checkpoints, "gpu_process_rows": gpu, "training_gpu_device_fd_bound": gpu_device_bound,
                "gpu_training_pid_match": gpu_device_bound,
                "training_process_matches": training_matches, "controller_process_matches": controller_matches,
                "duplicate_training": len(training_matches) > 1, "duplicate_controller": len(controller_matches) > 1,
                "model_modes": latest_modes, "post_validation_modes": latest_post_modes, "mode_lifecycle_ok": mode_lifecycle_ok,
                "pretrain_val": args.pretrain_val, "collapse_threshold": collapse_threshold,
                "collapse_sequence": [list(item) for item in collapse_sequence],
                "errors": errors, "natural_completion_marker": "NATURAL_COMPLETION" in text}
            atomic_json(state_path, state)
            for val_iteration, val in validations:
                if val_iteration > last_validation_iteration:
                    event = {"time": state["heartbeat_at"], "event": "VALIDATION", "iteration": val_iteration, "mDice": val,
                             "best": list(best), "finite": not errors, "training_alive": training_alive,
                             "model_modes": latest_post_modes or latest_modes, "mode_lifecycle_ok": mode_lifecycle_ok}
                    timeline.write(json.dumps(event) + "\n"); print("【RUNNING_STATUS】", json.dumps(event), flush=True)
                    last_validation_iteration = val_iteration
            if now - last_heartbeat >= args.heartbeat_seconds:
                timeline.write(json.dumps({"time": state["heartbeat_at"], "event": "HEARTBEAT", "iteration": current,
                    "training_alive": training_alive, "gpu_match": gpu_device_bound, "model_modes": latest_modes,
                    "mode_lifecycle_ok": mode_lifecycle_ok, "errors": errors}) + "\n")
                last_heartbeat = now
            stop_reason = None
            if not mode_lifecycle_ok:
                stop_reason = "MODE_LIFECYCLE_REGRESSION"
            elif collapse_sequence:
                stop_reason = "SELF_TRAINING_DESTROYS_PRETRAIN"
            if stop_reason and training_alive:
                state["authorized_gate_stop"] = stop_reason; atomic_json(state_path, state)
                timeline.write(json.dumps({"time": state["heartbeat_at"], "event": stop_reason,
                    "sequence": [list(item) for item in collapse_sequence]}) + "\n")
                os.kill(args.training_pid, signal.SIGTERM)
            if not training_alive:
                state["controller_alive"] = False; state["exit_decision"] = "NATURAL_COMPLETION" if state["natural_completion_marker"] else "TRAINING_PROCESS_LOST"
                atomic_json(state_path, state); timeline.write(json.dumps({"time": state["heartbeat_at"], "event": state["exit_decision"]}) + "\n"); break
            time.sleep(5)
    return 0


if __name__ == "__main__": raise SystemExit(main())
