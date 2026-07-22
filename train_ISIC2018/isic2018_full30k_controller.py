#!/usr/bin/env python3
"""Persistent real-task controller for ISIC2018 Candidate Full-30K."""
from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "runs/ISIC2018/control"
PY = sys.executable
SUCCESS_TERMINAL = "ISIC2018_5P_10P_CANDIDATE_FULL30K_COMPLETE"


@dataclass(frozen=True)
class TaskSpec:
    state: str
    next_state: str
    requires_gpu: bool
    artifacts: tuple[str, ...]

    def command(self) -> list[str]:
        return [PY, "-m", "train_ISIC2018.isic2018_full30k_tasks", "--state", self.state]


TASKS = [
    TaskSpec("F00_SHORT5K_RECOMMENDATION_OVERRIDE", "F01_FINAL_CODE_AUDIT", False, ("audit/isic2018_short5k_full_recommendation_override.json",)),
    TaskSpec("F01_FINAL_CODE_AUDIT", "F02_FULL_SOURCE_FREEZE", True, ("audit/isic2018_final_training_code_correctness_audit.json",)),
    TaskSpec("F02_FULL_SOURCE_FREEZE", "F03_5P_CANDIDATE_FULL30K", False, ("audit/isic2018_candidate_full30k_source_manifest.json", "audit/isic2018_candidate_full30k_protocol.json")),
    TaskSpec("F03_5P_CANDIDATE_FULL30K", "F04_10P_CANDIDATE_FULL30K", True, ("audit/isic2018_candidate_full30k_source_manifest.json",)),
    TaskSpec("F04_10P_CANDIDATE_FULL30K", "F05_BEST_CHECKPOINT_REPLAY", True, ("audit/isic2018_candidate_full30k_source_manifest.json",)),
    TaskSpec("F05_BEST_CHECKPOINT_REPLAY", "F06_EXTERNAL_METHOD_COMPARISON", True, ("audit/isic2018_candidate_full30k_best_checkpoint_replay.json",)),
    TaskSpec("F06_EXTERNAL_METHOD_COMPARISON", "F07_FULL_RESULT_SUMMARY", False, ("audit/isic2018_candidate_full30k_external_method_comparison.json",)),
    TaskSpec("F07_FULL_RESULT_SUMMARY", "F08_STOP_FORMAL_TEST_NOT_STARTED", False, ("audit/isic2018_candidate_full30k_result_summary.json",)),
    TaskSpec("F08_STOP_FORMAL_TEST_NOT_STARTED", SUCCESS_TERMINAL, False, ("audit/isic2018_candidate_full30k_final_closure.json",)),
]
TASK_BY_STATE = {t.state: t for t in TASKS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def gpu_pid() -> str | None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    for line in result.stdout.splitlines():
        if "python" in line.lower():
            return line.split(",", 1)[0].strip()
    return None


def stale_lock(lock: Path) -> bool:
    data = lock.read_text(errors="replace").strip() if lock.exists() else ""
    try:
        return not alive(int(data))
    except Exception:
        return True


def acquire_lock() -> Path:
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = CONTROL / "full30k_lock"
    if lock.exists() and stale_lock(lock):
        (CONTROL / f"full30k_lock.stale.{int(time.time())}").write_text(lock.read_text(errors="replace"))
        lock.unlink()
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    return lock


def base_state(current: str, last_completed: str | None = None) -> dict:
    spec = TASK_BY_STATE.get(current)
    return {
        "execution_mode": "PERSISTENT_SERVER_CONTROLLER",
        "tmux_session": "isic2018-prefull",
        "phase": "CANDIDATE_FULL30K",
        "controller_pid": os.getpid(),
        "controller_command": "python -m train_ISIC2018.isic2018_full30k_controller --run-queue",
        "child_pid": None,
        "child_command": None,
        "gpu_pid": gpu_pid(),
        "current_state": current,
        "last_completed_state": last_completed,
        "next_automatic_state": None if spec is None else spec.next_state,
        "terminal_allowed": False,
        "terminal_reason": "INTERMEDIATE_STATE",
        "early_stop_guard_pass": False,
        "heartbeat_timestamp": utc_now(),
        "FULL_TRAINING_AUTHORIZED": True,
        "FULL_BASELINE_REQUIRED": False,
        "FULL_CANDIDATE_REQUIRED": True,
        "FORMAL_TEST_STARTED": False,
        "queue": [t.__dict__ for t in TASKS],
    }


def validate_artifacts(spec: TaskSpec) -> list[str]:
    missing = [p for p in spec.artifacts if not (ROOT / p).exists()]
    if missing:
        raise RuntimeError("MISSING_REQUIRED_ARTIFACTS:" + ",".join(missing))
    return list(spec.artifacts)


def run_task(spec: TaskSpec, last_completed: str | None) -> str:
    command = spec.command()
    log_path = CONTROL / f"{spec.state}.stdout.log"
    state = base_state(spec.state, last_completed)
    state.update({"child_command": " ".join(command), "active_run": spec.state, "current_artifact": ",".join(spec.artifacts)})
    write_json(CONTROL / "prefull_state.json", state)
    write_json(CONTROL / "prefull_heartbeat.json", state)
    write_json(CONTROL / "prefull_active_run.json", {"active_run": spec.state, "stdout_path": str(log_path), "child_command": command})
    with log_path.open("a", buffering=1) as out, (CONTROL / "prefull_controller.log").open("a", buffering=1) as controller_log:
        controller_log.write(f"{utc_now()} START {spec.state} {' '.join(command)}\n")
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        selector = selectors.DefaultSelector()
        if child.stdout:
            selector.register(child.stdout, selectors.EVENT_READ)
        state["child_pid"] = child.pid
        write_json(CONTROL / "prefull_active_run.json", {"active_run": spec.state, "child_pid": child.pid, "stdout_path": str(log_path), "child_command": command})
        last_tick = 0.0
        while True:
            for key, _ in selector.select(timeout=0.5):
                line = key.fileobj.readline()
                if line:
                    out.write(line)
                    controller_log.write(f"{spec.state}: {line}")
            code = child.poll()
            now = time.time()
            if now - last_tick >= 5:
                state.update({"child_pid": child.pid, "gpu_pid": gpu_pid(), "heartbeat_timestamp": utc_now()})
                write_json(CONTROL / "prefull_state.json", state)
                write_json(CONTROL / "prefull_heartbeat.json", state)
                last_tick = now
            if code is not None:
                if child.stdout:
                    try:
                        selector.unregister(child.stdout)
                    except Exception:
                        pass
                    rest = child.stdout.read()
                    if rest:
                        out.write(rest)
                        controller_log.write(f"{spec.state}: {rest}")
                break
        if code != 0:
            state.update({"child_pid": child.pid, "exit_code": code, "last_failure_class": "CHILD_EXIT_CODE", "heartbeat_timestamp": utc_now()})
            write_json(CONTROL / "prefull_state.json", state)
            controller_log.write(f"{utc_now()} FAIL {spec.state} exit={code}\n")
            raise RuntimeError(f"{spec.state}_FAILED_EXIT_{code}")
        artifacts = validate_artifacts(spec)
        state.update({"child_pid": None, "gpu_pid": gpu_pid(), "exit_code": 0, "last_completed_state": spec.state, "current_state": spec.next_state, "next_automatic_state": TASK_BY_STATE.get(spec.next_state).next_state if spec.next_state in TASK_BY_STATE else None, "validated_artifacts": artifacts, "heartbeat_timestamp": utc_now()})
        write_json(CONTROL / "prefull_state.json", state)
        write_json(CONTROL / "prefull_heartbeat.json", state)
        controller_log.write(f"{utc_now()} PASS {spec.state} artifacts={artifacts}\n")
    return spec.next_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-queue", action="store_true")
    parser.add_argument("--start-state", default="F00_SHORT5K_RECOMMENDATION_OVERRIDE")
    args = parser.parse_args()
    if not args.run_queue:
        raise SystemExit("use --run-queue")
    lock = acquire_lock()
    current = args.start_state
    last_completed = read_json(CONTROL / "prefull_state.json").get("last_completed_state")
    try:
        while current in TASK_BY_STATE:
            current = run_task(TASK_BY_STATE[current], last_completed)
            last_completed = read_json(CONTROL / "prefull_state.json").get("last_completed_state")
        final_state = base_state(SUCCESS_TERMINAL, last_completed)
        final_state.update({"terminal_allowed": True, "terminal_reason": "FINAL_SUCCESS", "early_stop_guard_pass": True, "next_automatic_state": None})
        write_json(CONTROL / "prefull_state.json", final_state)
        write_json(CONTROL / "prefull_heartbeat.json", final_state)
        write_json(CONTROL / "prefull_active_run.json", {"active_run": None, "terminal": SUCCESS_TERMINAL})
        return 0
    except BaseException as exc:
        state = read_json(CONTROL / "prefull_state.json")
        state.update({"last_failure_class": type(exc).__name__, "last_failure": repr(exc), "terminal_allowed": False, "heartbeat_timestamp": utc_now()})
        write_json(CONTROL / "prefull_state.json", state)
        with (CONTROL / "prefull_controller.log").open("a", buffering=1) as log:
            log.write(f"{utc_now()} FULL30K_CONTROLLER_FAIL {repr(exc)}\n")
        return 2
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
