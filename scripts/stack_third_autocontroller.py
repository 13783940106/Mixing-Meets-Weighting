#!/usr/bin/env python3
"""State-machine controller for PB2D stack-third full experiments.

The controller is intentionally conservative: it can monitor an active full,
audit completion, classify the result, update persistent state, and execute the
next queued candidate through explicit gates. It does not invent a patch when no
candidate manifest exists; that condition is recorded as REAL_BLOCKER instead of
being represented as fake automation.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_STACK_BEST = 0.887704
ACCEPT_THRESHOLD = 0.897704
MAX_FAILURES = 3

CONTROL_DIR_REL = Path("audit/stack_third_controller")
STATE_REL = Path("experiments/project_state.json")
HANDOFF_REL = Path("experiments/handoff_summary.md")
RESUME_REL = Path("experiments/resume_prompt.md")
DECISION_LOG_REL = Path("experiments/innovation_decision_log.tsv")

VAL_RE = re.compile(r"iteration\s+(\d+)\s+:\s+model1_mean_dice\s+:\s+([0-9.]+)")
TEST_RE = re.compile(r"iteration\s+(\d+)\s+:\s+model1_mean_test_dice\s+:\s+([0-9.]+)")
ITER_RE = re.compile(r"iteration\s+(\d+):\s+loss:")
LCC_RE = re.compile(
    r"iteration\s+(\d+)\s+:\s+stack_early_local_contrast_consistency_v1_enabled\s+:\s+True,.*?"
    r"stack_early_local_contrast_consistency_v1_ramp\s+:\s+([0-9.eE+-]+),.*?"
    r"stack_early_local_contrast_consistency_v1_finite\s+:\s+(True|False),\s+total_finite\s+:\s+(True|False)"
)
EPHC_RE = re.compile(
    r"iteration\s+(\d+)\s+:\s+stack_early_probability_histogram_consistency_v1_enabled\s+:\s+True,.*?"
    r"stack_early_probability_histogram_consistency_v1_ramp\s+:\s+([0-9.eE+-]+),.*?"
    r"stack_early_probability_histogram_consistency_v1_finite\s+:\s+(True|False),\s+total_finite\s+:\s+(True|False)"
)
ERROR_RE = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|out of memory|CUDA error|"
    r"NaN|nan|Inf|inf|finite\s*:\s*False|total_finite\s*:\s*False"
)
FINITE_RE = re.compile(r"total_finite\s+:\s+(True|False)")


STATES = [
    "RECOVER_REAL_STATE",
    "MONITOR_ACTIVE_FULL",
    "ACTIVE_FULL_COMPLETED",
    "ACTIVE_FULL_INTERRUPTED",
    "PARSE_FULL_METRICS",
    "WRITE_COMPLETION_REPORT",
    "CLASSIFY_DECISION",
    "UPDATE_PERSISTENT_STATE",
    "DISCOVER_NEXT_CANDIDATE",
    "WRITE_CANDIDATE_CARD",
    "COMPATIBILITY_GATE",
    "PATCH_CANDIDATE",
    "STATIC_GATE",
    "SMOKE_GATE",
    "FULL_LAUNCH_GATE",
    "LAUNCH_DETACHED_FULL",
    "VERIFY_NEW_FULL",
    "SUCCESS",
    "REAL_BLOCKER",
]


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_text(path):
    return path.read_text(errors="replace") if path.exists() else ""


def read_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    with path.open() as handle:
        return json.load(handle)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def append_line(path, line):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line.rstrip("\n") + "\n")


def tail(text, limit=4000):
    return text[-limit:] if len(text) > limit else text


def boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    return read_text(path).strip() if path.exists() else "unknown"


def pid_alive(pid):
    try:
        return int(pid) > 0 and Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def proc_cmdline(pid):
    path = Path(f"/proc/{pid}/cmdline")
    return read_text(path).replace("\x00", " ") if path.exists() else ""


def iso_from_timestamp(ts):
    if not ts:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().isoformat(timespec="seconds")


def run_command(command, cwd, env=None, timeout=900):
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=merged_env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr


class StackThirdController:
    def __init__(self, root=ROOT, interval=300, simulate=False):
        self.root = Path(root)
        self.interval = interval
        self.simulate = simulate
        self.control_dir = self.root / CONTROL_DIR_REL
        self.state_path = self.root / STATE_REL
        self.control_state_path = self.control_dir / "controller_state.json"
        self.action_log_path = self.control_dir / "action_log.jsonl"
        self.lock_path = self.control_dir / "controller.lock"
        self.queue_path = self.control_dir / "candidate_queue.json"
        self.blocker_path = self.control_dir / "real_blocker_report.md"
        self.self_pid = os.getpid()
        self.start_time = now()
        self.current_state = "RECOVER_REAL_STATE"
        self.current_action = "boot"
        self.last_successful_action = None
        self.last_exception = None
        self.consecutive_failures = 0
        self.next_action = "RECOVER_REAL_STATE"
        self.metrics = {}
        self.decision = None
        self.active_candidate = None
        self.new_pid = None
        self.control_dir.mkdir(parents=True, exist_ok=True)

    def next_seq(self):
        if not self.action_log_path.exists():
            return 1
        last = 0
        for line in self.action_log_path.read_text(errors="replace").splitlines():
            try:
                last = max(last, int(json.loads(line).get("seq", 0)))
            except json.JSONDecodeError:
                continue
        return last + 1

    def log_action(self, state, action, command=None, return_code=0,
                   stdout="", stderr="", artifact=None, next_state=None):
        record = {
            "seq": self.next_seq(),
            "timestamp": now(),
            "state": state,
            "action": action,
            "command": command,
            "return_code": return_code,
            "stdout_tail": tail(stdout or "", 2000),
            "stderr_tail": tail(stderr or "", 2000),
            "artifact": str(artifact) if artifact else None,
            "next_state": next_state,
        }
        append_line(self.action_log_path, json.dumps(record, sort_keys=True))
        if return_code == 0:
            self.last_successful_action = action
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        return record

    def heartbeat(self):
        project = read_json(self.state_path, {})
        previous = read_json(self.control_state_path, {})
        previous_heartbeat = previous.get("heartbeat_at")
        age = 0
        if previous_heartbeat:
            try:
                parsed = datetime.fromisoformat(previous_heartbeat)
                age = max(0, int((datetime.now(timezone.utc).astimezone() - parsed).total_seconds()))
            except ValueError:
                age = None
        snapshot = {
            "controller_pid": self.self_pid,
            "controller_start_time": self.start_time,
            "controller_boot_id": boot_id(),
            "heartbeat_at": now(),
            "heartbeat_age_seconds": age,
            "current_state": self.current_state,
            "current_action": self.current_action,
            "active_training_pid": project.get("active_training_pid"),
            "active_experiment": project.get("active_training_experiment"),
            "latest_iter": project.get("active_training_latest_iter", project.get("active_training_latest_val_iter")),
            "best_val": project.get("active_training_best_val_dsc"),
            "last_successful_action": self.last_successful_action,
            "last_exception": self.last_exception,
            "consecutive_failures": self.consecutive_failures,
            "next_action": self.next_action,
        }
        atomic_json(self.control_state_path, snapshot)
        return snapshot

    def parse_log(self, log_path):
        text = read_text(log_path)
        if "Start self_training" in text:
            pretrain_text, self_text = text.rsplit("Start self_training", 1)
        elif "model1_loss:" in text:
            split_at = text.find("model1_loss:")
            line_start = text.rfind("\n", 0, split_at)
            pretrain_text = text[:max(0, line_start)]
            self_text = text[max(0, line_start):]
        else:
            pretrain_text, self_text = text, ""
        active_text = self_text if self_text else text
        vals = [(int(i), float(v)) for i, v in VAL_RE.findall(active_text)]
        tests = {int(i): float(v) for i, v in TEST_RE.findall(active_text)}
        pretrain_iters = [int(i) for i in ITER_RE.findall(pretrain_text)]
        iters = [int(i) for i in ITER_RE.findall(active_text)]
        finite_flags = FINITE_RE.findall(active_text)
        lcc = [(int(i), float(ramp), finite == "True", total == "True") for i, ramp, finite, total in LCC_RE.findall(active_text)]
        ephc = [(int(i), float(ramp), finite == "True", total == "True") for i, ramp, finite, total in EPHC_RE.findall(active_text)]
        candidate_logs = lcc + ephc
        candidate_logs.sort(key=lambda item: item[0])
        best_iter, best_val = (None, None)
        if vals:
            best_iter, best_val = max(vals, key=lambda item: item[1])
        latest_iter, latest_val = vals[-1] if vals else (None, None)
        stat = log_path.stat() if log_path.exists() else None
        previous_live = read_json(log_path.parent / "live_status.json", {})
        size = stat.st_size if stat else 0
        mtime = stat.st_mtime if stat else None
        previous_size = previous_live.get("log_size_bytes")
        previous_mtime = previous_live.get("last_log_mtime_epoch")
        log_growing = bool(
            previous_size is None
            or size > previous_size
            or (previous_mtime is not None and mtime is not None and mtime > previous_mtime)
        )
        latest_candidate = candidate_logs[-1] if candidate_logs else (None, None, None, None)
        final_val = vals[-1][1] if vals else None
        final_test = tests.get(vals[-1][0]) if vals else None
        return {
            "log_path": str(log_path.relative_to(self.root)) if log_path.exists() else str(log_path),
            "log_size_bytes": size,
            "last_log_mtime_epoch": mtime,
            "last_log_mtime": iso_from_timestamp(mtime),
            "log_growing": log_growing,
            "pretrain_latest_iter": max(pretrain_iters) if pretrain_iters else None,
            "self_train_started": bool(self_text),
            "latest_iter": max(iters) if iters else latest_iter,
            "latest_val_iter": latest_iter,
            "latest_val": latest_val,
            "latest_test": tests.get(latest_iter),
            "best_iter": best_iter,
            "best_val": best_val,
            "test_at_best": tests.get(best_iter),
            "final_val": final_val,
            "final_test": final_test,
            "val_points": len(vals),
            "finite_points": sum(flag == "True" for flag in finite_flags),
            "nonfinite_points": sum(flag == "False" for flag in finite_flags),
            "nonfinite_hits": sum(flag == "False" for flag in finite_flags),
            "candidate_enabled": bool(candidate_logs),
            "candidate_latest_iter": latest_candidate[0],
            "candidate_latest_ramp": latest_candidate[1],
            "candidate_latest_finite": latest_candidate[2],
            "candidate_latest_total_finite": latest_candidate[3],
            "candidate_finite_points": sum(finite and total for _, _, finite, total in candidate_logs),
            "candidate_nonfinite_points": sum((not finite) or (not total) for _, _, finite, total in candidate_logs),
            "error_hits": len(ERROR_RE.findall(text)),
        }

    def checkpoint_status(self, checkpoint_dir):
        files = {entry.name for entry in checkpoint_dir.glob("*")} if checkpoint_dir.exists() else set()
        best_files = [name for name in files if name.startswith("model1_iter_") and "_dice_" in name]
        return {
            "checkpoint_dir": str(checkpoint_dir.relative_to(self.root)) if checkpoint_dir.exists() else str(checkpoint_dir),
            "best_checkpoint_seen": "unet_best_model.pth" in files and bool(best_files),
            "best_checkpoint_files": sorted(best_files)[-5:],
            "final_checkpoint_seen": "unet_final_model.pth" in files and "model1_iter_30000_final.pth" in files,
            "final_checkpoint_files": sorted(name for name in files if name.endswith("_final.pth") or name == "unet_final_model.pth"),
        }

    def training_identity_ok(self, pid, experiment):
        if self.simulate and pid:
            return True
        if not pid_alive(pid):
            return False
        cmd = proc_cmdline(pid)
        if "train_promise12_bcp_fft.py" in cmd and (not experiment or experiment in cmd):
            return True
        if experiment and ("run_pb2d_original_2d_combo" in cmd or "bash" in cmd or "conda" in cmd):
            try:
                table = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
            except subprocess.CalledProcessError:
                return False
            return "train_promise12_bcp_fft.py" in table and experiment in table
        return False

    def find_training_python_pid(self, experiment):
        if self.simulate:
            return None
        if not experiment:
            return None
        try:
            table = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        except subprocess.CalledProcessError:
            return None
        fallback = None
        for line in table.splitlines():
            if "train_promise12_bcp_fft.py" not in line or experiment not in line:
                continue
            try:
                pid_text, args = line.strip().split(None, 1)
                parsed_pid = int(pid_text)
            except (ValueError, IndexError):
                continue
            if args.startswith("python train_promise12_bcp_fft.py") or args.startswith("python3 train_promise12_bcp_fft.py"):
                return parsed_pid
            if fallback is None and "conda run" not in args:
                fallback = parsed_pid
        return fallback

    def recover_real_state(self):
        self.current_action = "recover_real_state"
        project = read_json(self.state_path, {})
        pid = project.get("active_training_pid")
        experiment = project.get("active_training_experiment")
        log_path = self.root / project.get("active_training_log", "")
        ckpt_dir = self.root / project.get("active_training_expected_checkpoint_dir", "")
        metrics = self.parse_log(log_path)
        ckpt = self.checkpoint_status(ckpt_dir)
        self.metrics = {**metrics, **ckpt}

        if self.training_identity_ok(pid, experiment):
            next_state = "MONITOR_ACTIVE_FULL"
            action = "active full verified"
        elif experiment and ckpt["final_checkpoint_seen"]:
            next_state = "ACTIVE_FULL_COMPLETED"
            action = "completed full detected"
        elif experiment and project.get("active_training_status", "").startswith("FULL_COMPLETE"):
            next_state = "DISCOVER_NEXT_CANDIDATE"
            action = "previous full already classified"
        elif experiment and (log_path.exists() or ckpt_dir.exists()) and not ckpt["final_checkpoint_seen"]:
            next_state = "ACTIVE_FULL_INTERRUPTED"
            action = "incomplete full detected"
        else:
            next_state = "DISCOVER_NEXT_CANDIDATE"
            action = "no active full"
        self.log_action("RECOVER_REAL_STATE", action, artifact=self.control_state_path, next_state=next_state)
        return next_state

    def monitor_active_full(self):
        self.current_action = "monitor_active_full"
        project = read_json(self.state_path, {})
        pid = project.get("active_training_pid")
        experiment = project.get("active_training_experiment")
        log_path = self.root / project.get("active_training_log", "")
        ckpt_dir = self.root / project.get("active_training_expected_checkpoint_dir", "")
        self.metrics = {**self.parse_log(log_path), **self.checkpoint_status(ckpt_dir)}
        train_pid = self.find_training_python_pid(experiment) or pid
        process_alive = self.training_identity_ok(pid, experiment) or self.training_identity_ok(train_pid, experiment)
        if process_alive:
            next_state = "MONITOR_ACTIVE_FULL"
            action = "training still running"
        elif self.metrics.get("final_checkpoint_seen"):
            next_state = "ACTIVE_FULL_COMPLETED"
            action = "training completed"
        else:
            next_state = "ACTIVE_FULL_INTERRUPTED"
            action = "training missing incomplete"
        live = log_path.parent / "live_status.json"
        live_payload = {
            **self.metrics,
            "experiment": experiment,
            "training_pid": train_pid,
            "launcher_pid": pid if train_pid != pid else None,
            "process_alive": process_alive,
            "state": next_state,
            "heartbeat_at": now(),
            "next_action": next_state,
            "updated_at": now(),
        }
        atomic_json(live, live_payload)
        project.update({
            "active_training_pid": train_pid,
            "active_training_launcher_pid": pid if train_pid != pid else project.get("active_training_launcher_pid"),
            "active_training_status": next_state,
            "active_training_latest_iter": self.metrics.get("latest_iter"),
            "active_training_latest_val_iter": self.metrics.get("latest_val_iter"),
            "active_training_latest_val_dsc": self.metrics.get("latest_val"),
            "active_training_best_val_iter": self.metrics.get("best_iter"),
            "active_training_best_val_dsc": self.metrics.get("best_val"),
            "stack_third_controller_updated_at": now(),
            "next_required_action": "MONITOR_ACTIVE_FULL" if next_state == "MONITOR_ACTIVE_FULL" else next_state,
        })
        atomic_json(self.state_path, project)
        stdout = json.dumps({
            "experiment": experiment,
            "training_pid": train_pid,
            "latest_iter": self.metrics.get("latest_iter"),
            "best_iter": self.metrics.get("best_iter"),
            "best_val": self.metrics.get("best_val"),
            "log_growing": self.metrics.get("log_growing"),
            "error_hits": self.metrics.get("error_hits"),
            "nonfinite_hits": self.metrics.get("nonfinite_hits"),
        }, sort_keys=True)
        self.log_action("MONITOR_ACTIVE_FULL", action, stdout=stdout, artifact=live, next_state=next_state)
        return next_state

    def audit_completed_full(self):
        self.current_action = "audit_completed_full"
        project = read_json(self.state_path, {})
        log_path = self.root / project.get("active_training_log", "")
        ckpt_dir = self.root / project.get("active_training_expected_checkpoint_dir", "")
        self.metrics = {**self.parse_log(log_path), **self.checkpoint_status(ckpt_dir)}
        ok = (
            self.metrics.get("final_checkpoint_seen")
            and self.metrics.get("best_checkpoint_seen")
            and self.metrics.get("best_val") is not None
            and self.metrics.get("error_hits", 0) == 0
            and self.metrics.get("nonfinite_points", 0) == 0
        )
        next_state = "PARSE_FULL_METRICS" if ok else "ACTIVE_FULL_INTERRUPTED"
        artifact = self.control_dir / "latest_completed_full_audit.json"
        atomic_json(artifact, self.metrics)
        self.log_action("ACTIVE_FULL_COMPLETED", "audit_completed_full", artifact=artifact, next_state=next_state)
        return next_state

    def audit_interrupted_full(self):
        self.current_action = "audit_interrupted_full"
        project = read_json(self.state_path, {})
        artifact = self.control_dir / "active_full_interrupted_or_incomplete_audit.md"
        text = (
            "# Active Full Interrupted Or Incomplete Audit\n\n"
            f"- time: `{now()}`\n"
            f"- experiment: `{project.get('active_training_experiment')}`\n"
            f"- pid: `{project.get('active_training_pid')}`\n"
            f"- log: `{project.get('active_training_log')}`\n"
            f"- checkpoint_dir: `{project.get('active_training_expected_checkpoint_dir')}`\n"
            f"- latest_iter: `{self.metrics.get('latest_iter')}`\n"
            f"- best_val: `{self.metrics.get('best_val')}`\n"
            f"- final_checkpoint_seen: `{self.metrics.get('final_checkpoint_seen')}`\n"
            "\nDecision: `STACK_THIRD_FAIL_OR_INCOMPLETE`.\n"
        )
        artifact.write_text(text)
        self.log_action("ACTIVE_FULL_INTERRUPTED", "write interrupted audit", artifact=artifact, next_state="UPDATE_PERSISTENT_STATE")
        self.decision = "STACK_THIRD_FAIL_OR_INCOMPLETE"
        return "UPDATE_PERSISTENT_STATE"

    def parse_full_metrics(self):
        self.current_action = "parse_full_metrics"
        artifact = self.control_dir / "latest_full_metrics_summary.json"
        atomic_json(artifact, self.metrics)
        self.log_action("PARSE_FULL_METRICS", "write metrics summary", artifact=artifact, next_state="WRITE_COMPLETION_REPORT")
        return "WRITE_COMPLETION_REPORT"

    def write_completion_report(self):
        self.current_action = "write_completion_report"
        project = read_json(self.state_path, {})
        artifact = self.control_dir / "latest_full_completion_report.md"
        report = (
            "# Stack Third Full Completion Report\n\n"
            f"- time: `{now()}`\n"
            f"- experiment: `{project.get('active_training_experiment')}`\n"
            f"- log: `{project.get('active_training_log')}`\n"
            f"- checkpoint_dir: `{project.get('active_training_expected_checkpoint_dir')}`\n"
            f"- best val DSC: `{self.metrics.get('best_val')}` @ `{self.metrics.get('best_iter')}`\n"
            f"- test at best: `{self.metrics.get('test_at_best')}` (non-selection reference)\n"
            f"- final val/test: `{self.metrics.get('final_val')}` / `{self.metrics.get('final_test')}`\n"
            f"- best checkpoint seen: `{self.metrics.get('best_checkpoint_seen')}`\n"
            f"- final checkpoint seen: `{self.metrics.get('final_checkpoint_seen')}`\n"
            f"- error hits: `{self.metrics.get('error_hits')}`\n"
            f"- nonfinite points: `{self.metrics.get('nonfinite_points')}`\n"
            "\nSelection policy: validation-only best. Test is recorded only as reference.\n"
        )
        artifact.write_text(report)
        self.log_action("WRITE_COMPLETION_REPORT", "write completion report", artifact=artifact, next_state="CLASSIFY_DECISION")
        return "CLASSIFY_DECISION"

    def classify_decision(self):
        self.current_action = "classify_decision"
        best = self.metrics.get("best_val")
        if (
            best is None
            or self.metrics.get("error_hits", 0) > 0
            or self.metrics.get("nonfinite_points", 0) > 0
            or not self.metrics.get("best_checkpoint_seen")
            or not self.metrics.get("final_checkpoint_seen")
        ):
            self.decision = "STACK_THIRD_FAIL"
        elif best >= ACCEPT_THRESHOLD:
            self.decision = "STACK_THIRD_ACCEPT_PLUS_ONE"
        elif best > BASE_STACK_BEST:
            self.decision = "STACK_THIRD_WEAK_POSITIVE"
        else:
            self.decision = "STACK_THIRD_NO_GAIN"
        artifact = self.control_dir / "latest_full_decision.md"
        artifact.write_text(
            "# Stack Third Decision\n\n"
            f"- decision: `{self.decision}`\n"
            f"- best val DSC: `{best}`\n"
            f"- base stack best: `{BASE_STACK_BEST}`\n"
            f"- accept threshold: `{ACCEPT_THRESHOLD}`\n"
            "- selection: validation-only\n"
        )
        next_state = "SUCCESS" if self.decision == "STACK_THIRD_ACCEPT_PLUS_ONE" else "UPDATE_PERSISTENT_STATE"
        self.log_action("CLASSIFY_DECISION", "classify decision", artifact=artifact, next_state=next_state)
        return next_state

    def update_registry_and_state(self):
        self.current_action = "update_registry_and_state"
        project = read_json(self.state_path, {})
        if self.metrics:
            project.update({
                "active_training_pid": None,
                "active_training": None,
                "active_training_best_val_dsc": self.metrics.get("best_val"),
                "active_training_best_val_iter": self.metrics.get("best_iter"),
                "active_training_latest_val_dsc": self.metrics.get("latest_val"),
                "active_training_latest_val_iter": self.metrics.get("latest_val_iter"),
                "active_training_status": f"FULL_COMPLETE_{self.decision}" if self.decision else project.get("active_training_status"),
                "next_required_action": "CONTINUE_EARLY_ACTIVE_STACK_THIRD_MODULE_SEARCH_NEXT_ROUND",
                "stack_third_controller_updated_at": now(),
            })
        if self.decision == "STACK_THIRD_FAIL_OR_INCOMPLETE":
            project["active_training_status"] = "STACK_THIRD_FAIL_OR_INCOMPLETE"
        atomic_json(self.state_path, project)
        self.update_handoff_files(project)
        self.log_action("UPDATE_PERSISTENT_STATE", "update project_state handoff resume", artifact=self.state_path, next_state="DISCOVER_NEXT_CANDIDATE")
        return "DISCOVER_NEXT_CANDIDATE"

    def update_handoff_files(self, project):
        summary = self.root / HANDOFF_REL
        resume = self.root / RESUME_REL
        block = (
            "# Current PB2D Stack Third Controller State\n\n"
            f"Updated: {now()}\n\n"
            "- Fixed base stack: `Mask Token + AdaMix`, best val `0.887704`.\n"
            "- Third-module accept threshold: `0.897704`.\n"
            "- LGDC has completed and is archived as `STACK_THIRD_NO_GAIN`, best val `0.874780 @27400`.\n"
            "- No active training process is currently verified.\n"
            "- Next required action: `CONTINUE_EARLY_ACTIVE_STACK_THIRD_MODULE_SEARCH_NEXT_ROUND`.\n"
            "- Controller must use a real candidate queue/manifest before launching another full.\n"
        )
        summary.write_text(block)
        resume.write_text(block + "\nDo not rerun completed candidates or start a full without passing controller gates.\n")

    def load_queue(self):
        data = read_json(self.queue_path, {"candidates": []})
        if isinstance(data, list):
            return {"candidates": data}
        data.setdefault("candidates", [])
        return data

    def save_queue(self, queue):
        atomic_json(self.queue_path, queue)

    def discover_next_candidate(self):
        self.current_action = "discover_next_candidate"
        queue = self.load_queue()
        pending = [item for item in queue["candidates"] if item.get("status", "pending") == "pending"]
        artifact = self.control_dir / "latest_candidate_discovery.md"
        if not pending:
            text = (
                "# Candidate Discovery\n\n"
                f"- time: `{now()}`\n"
                "- result: `NO_PENDING_REAL_CANDIDATE_MANIFEST`\n"
                "- next_state: `REAL_BLOCKER`\n\n"
                "The controller can execute queued candidates, but no untried early-active "
                "third-module candidate with a real launcher/patch plan is currently present in "
                "`audit/stack_third_controller/candidate_queue.json`.\n"
            )
            artifact.write_text(text)
            self.log_action("DISCOVER_NEXT_CANDIDATE", "no pending candidate", artifact=artifact, next_state="REAL_BLOCKER")
            return "REAL_BLOCKER"
        self.active_candidate = pending[0]
        self.active_candidate["status"] = "selected"
        queue["candidates"] = [self.active_candidate if item is pending[0] else item for item in queue["candidates"]]
        self.save_queue(queue)
        artifact.write_text(
            "# Candidate Discovery\n\n"
            f"- selected: `{self.active_candidate.get('candidate_id')}`\n"
            f"- launcher: `{self.active_candidate.get('launcher')}`\n"
            f"- start_iter: `{self.active_candidate.get('start_iter')}`\n"
            f"- ramp_iters: `{self.active_candidate.get('ramp_iters')}`\n"
        )
        self.log_action("DISCOVER_NEXT_CANDIDATE", "selected pending candidate", artifact=artifact, next_state="WRITE_CANDIDATE_CARD")
        return "WRITE_CANDIDATE_CARD"

    def build_candidate_card(self):
        self.current_action = "build_candidate_card"
        candidate = self.active_candidate or {}
        card_path = self.root / candidate.get("candidate_card", f"audit/{candidate.get('candidate_id', 'unknown')}_candidate_card.md")
        if not card_path.exists():
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_text(
                "# Stack Third Candidate Card\n\n"
                f"- candidate: `{candidate.get('candidate_id')}`\n"
                f"- early-active: start_iter `{candidate.get('start_iter')}`, ramp_iters `{candidate.get('ramp_iters')}`\n"
                "- fixed base: `Mask Token + AdaMix`\n"
                "- selection: validation-only\n"
            )
        self.log_action("WRITE_CANDIDATE_CARD", "candidate card ready", artifact=card_path, next_state="COMPATIBILITY_GATE")
        return "COMPATIBILITY_GATE"

    def run_compatibility_audit(self):
        self.current_action = "run_compatibility_audit"
        candidate = self.active_candidate or {}
        start_iter = int(candidate.get("start_iter", 999999))
        ramp_iters = int(candidate.get("ramp_iters", 999999))
        launcher = self.root / candidate.get("launcher", "")
        forbidden = any(token in json.dumps(candidate).lower() for token in ["stagea", "matched-control", "arm a", "arm b", "late-only"])
        passed = start_iter <= 1000 and ramp_iters <= 3000 and launcher.exists() and not forbidden
        artifact = self.root / candidate.get("compatibility_audit", f"audit/{candidate.get('candidate_id', 'unknown')}_compatibility_audit.md")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            "# Compatibility Audit\n\n"
            f"- candidate: `{candidate.get('candidate_id')}`\n"
            f"- start_iter <= 1000: `{start_iter <= 1000}` ({start_iter})\n"
            f"- ramp_iters <= 3000: `{ramp_iters <= 3000}` ({ramp_iters})\n"
            f"- launcher exists: `{launcher.exists()}`\n"
            f"- forbidden protocol tokens: `{forbidden}`\n"
            f"- decision: `{'PASS' if passed else 'FAIL'}`\n"
        )
        next_state = "PATCH_CANDIDATE" if passed else "REAL_BLOCKER"
        self.log_action("COMPATIBILITY_GATE", "compatibility audit", return_code=0 if passed else 2, artifact=artifact, next_state=next_state)
        return next_state

    def apply_minimal_patch(self):
        self.current_action = "apply_minimal_patch"
        candidate = self.active_candidate or {}
        command = candidate.get("patch_command")
        if command:
            rc, out, err = run_command(command, self.root, timeout=int(candidate.get("patch_timeout", 900)))
        else:
            rc, out, err = 0, "PATCH_NOT_REQUIRED_FOR_PREIMPLEMENTED_CANDIDATE", ""
        artifact = self.root / candidate.get("source_patch_audit", f"audit/{candidate.get('candidate_id', 'unknown')}_source_patch_audit.md")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"# Source Patch Audit\n\n- rc: `{rc}`\n- stdout tail:\n\n```text\n{tail(out)}\n```\n")
        next_state = "STATIC_GATE" if rc == 0 else "REAL_BLOCKER"
        self.log_action("PATCH_CANDIDATE", "apply minimal patch", command=command, return_code=rc, stdout=out, stderr=err, artifact=artifact, next_state=next_state)
        return next_state

    def run_static_gate(self):
        self.current_action = "run_static_gate"
        candidate = self.active_candidate or {}
        launcher = candidate.get("launcher")
        commands = [
            "/opt/conda/bin/python -m py_compile train_promise12/train_promise12_bcp_fft.py",
            f"bash -n {launcher}",
        ]
        outputs = []
        rc_final = 0
        for command in commands:
            rc, out, err = run_command(command, self.root, timeout=300)
            outputs.append((command, rc, out, err))
            rc_final = rc_final or rc
        artifact = self.control_dir / "latest_static_gate.md"
        artifact.write_text("\n".join(
            [f"# Static Gate\n"] +
            [f"## `{cmd}`\n\n- rc: `{rc}`\n\n```text\n{tail(out + err)}\n```" for cmd, rc, out, err in outputs]
        ))
        next_state = "SMOKE_GATE" if rc_final == 0 else "REAL_BLOCKER"
        self.log_action("STATIC_GATE", "run static checks", command=" && ".join(commands), return_code=rc_final, artifact=artifact, next_state=next_state)
        return next_state

    def run_smoke(self):
        self.current_action = "run_smoke"
        candidate = self.active_candidate or {}
        command = candidate.get("smoke_command")
        if self.simulate:
            rc, out, err = 0, "SIMULATED_SMOKE_PASS", ""
        elif command:
            rc, out, err = run_command(command, self.root, timeout=int(candidate.get("smoke_timeout", 3600)))
        else:
            rc, out, err = 2, "", "missing smoke_command"
        artifact = self.root / candidate.get("smoke_report", f"audit/{candidate.get('candidate_id', 'unknown')}_smoke_report.md")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"# Smoke Report\n\n- rc: `{rc}`\n\n```text\n{tail(out + err)}\n```\n")
        next_state = "FULL_LAUNCH_GATE" if rc == 0 else "REAL_BLOCKER"
        self.log_action("SMOKE_GATE", "run smoke", command=command, return_code=rc, stdout=out, stderr=err, artifact=artifact, next_state=next_state)
        return next_state

    def write_smoke_report(self):
        return "FULL_LAUNCH_GATE"

    def run_full_launch_audit(self):
        self.current_action = "run_full_launch_audit"
        candidate = self.active_candidate or {}
        active = self.find_active_training()
        passed = active is None
        artifact = self.root / candidate.get("full_launch_audit", f"audit/{candidate.get('candidate_id', 'unknown')}_full_launch_audit.md")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            "# Full Launch Audit\n\n"
            f"- candidate: `{candidate.get('candidate_id')}`\n"
            f"- no concurrent training: `{passed}`\n"
            f"- start_iter <= 1000: `{int(candidate.get('start_iter', 999999)) <= 1000}`\n"
            f"- ramp_iters <= 3000: `{int(candidate.get('ramp_iters', 999999)) <= 3000}`\n"
            "- smoke checks finite/shape/checkpoint only; full selection is validation-only.\n"
        )
        next_state = "LAUNCH_DETACHED_FULL" if passed else "MONITOR_ACTIVE_FULL"
        self.log_action("FULL_LAUNCH_GATE", "full launch audit", return_code=0 if passed else 2, artifact=artifact, next_state=next_state)
        return next_state

    def launch_detached_full(self):
        self.current_action = "launch_detached_full"
        candidate = self.active_candidate or {}
        launcher = candidate.get("launcher")
        log_path = self.root / candidate.get("full_log", f"audit/{candidate.get('candidate_id', 'unknown')}_full/console.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = candidate.get("env", {})
        if self.simulate:
            self.new_pid = self.self_pid
            rc, out, err = 0, f"SIMULATED_PID={self.new_pid}", ""
        else:
            env_prefix = " ".join(f"{k}={json.dumps(str(v))}" for k, v in env.items())
            command = f"{env_prefix} setsid bash {launcher} >> {log_path} 2>&1 < /dev/null & echo $!"
            rc, out, err = run_command(command, self.root, timeout=60)
            try:
                self.new_pid = int(out.strip().splitlines()[-1])
            except (ValueError, IndexError):
                self.new_pid = None
        project = read_json(self.state_path, {})
        if rc == 0:
            project.update({
                "current_candidate": candidate.get("candidate_id"),
                "active_training": candidate.get("candidate_id"),
                "active_training_pid": self.new_pid,
                "active_training_experiment": candidate.get("experiment"),
                "active_training_log": str(log_path.relative_to(self.root)),
                "active_training_expected_checkpoint_dir": candidate.get("checkpoint_dir"),
                "active_training_status": "FULL_LAUNCHED",
                "next_required_action": "MONITOR_ACTIVE_FULL",
                "stack_third_controller_updated_at": now(),
            })
            atomic_json(self.state_path, project)
        next_state = "VERIFY_NEW_FULL" if rc == 0 else "REAL_BLOCKER"
        self.log_action("LAUNCH_DETACHED_FULL", "launch detached full", command=launcher, return_code=rc, stdout=out, stderr=err, artifact=log_path, next_state=next_state)
        return next_state

    def monitor_launched_full(self):
        return self.verify_new_full()

    def verify_new_full(self):
        self.current_action = "verify_new_full"
        project = read_json(self.state_path, {})
        pid = project.get("active_training_pid")
        experiment = project.get("active_training_experiment")
        ok = self.simulate or self.training_identity_ok(pid, experiment)
        next_state = "MONITOR_ACTIVE_FULL" if ok else "REAL_BLOCKER"
        self.log_action("VERIFY_NEW_FULL", "verify launched full", return_code=0 if ok else 2, artifact=self.state_path, next_state=next_state)
        return next_state

    def find_active_training(self):
        if self.simulate:
            return None
        try:
            out = subprocess.check_output(["pgrep", "-af", "train_promise12_bcp_fft.py"], text=True)
            return out.strip() or None
        except subprocess.CalledProcessError:
            return None

    def real_blocker(self):
        self.current_action = "real_blocker"
        text = (
            "# Stack Third Controller Real Blocker\n\n"
            f"- time: `{now()}`\n"
            f"- current_state: `{self.current_state}`\n"
            f"- current_action: `{self.current_action}`\n"
            f"- last_exception: `{self.last_exception}`\n"
            "- reason: controller has no pending real candidate manifest to execute, or a gate failed.\n"
            "- required input: add a genuine early-active candidate entry to "
            "`audit/stack_third_controller/candidate_queue.json` with launcher, smoke command, "
            "start_iter <= 1000, ramp_iters <= 3000, and checkpoint/log paths.\n"
        )
        self.blocker_path.write_text(text)
        self.log_action("REAL_BLOCKER", "write blocker report", artifact=self.blocker_path, next_state="REAL_BLOCKER")
        return "REAL_BLOCKER"

    def success(self):
        self.current_action = "success"
        self.log_action("SUCCESS", "accept threshold reached", artifact=self.control_dir / "latest_full_decision.md", next_state="SUCCESS")
        return "SUCCESS"

    def dispatch(self, state):
        handlers = {
            "RECOVER_REAL_STATE": self.recover_real_state,
            "MONITOR_ACTIVE_FULL": self.monitor_active_full,
            "ACTIVE_FULL_COMPLETED": self.audit_completed_full,
            "ACTIVE_FULL_INTERRUPTED": self.audit_interrupted_full,
            "PARSE_FULL_METRICS": self.parse_full_metrics,
            "WRITE_COMPLETION_REPORT": self.write_completion_report,
            "CLASSIFY_DECISION": self.classify_decision,
            "UPDATE_PERSISTENT_STATE": self.update_registry_and_state,
            "DISCOVER_NEXT_CANDIDATE": self.discover_next_candidate,
            "WRITE_CANDIDATE_CARD": self.build_candidate_card,
            "COMPATIBILITY_GATE": self.run_compatibility_audit,
            "PATCH_CANDIDATE": self.apply_minimal_patch,
            "STATIC_GATE": self.run_static_gate,
            "SMOKE_GATE": self.run_smoke,
            "FULL_LAUNCH_GATE": self.run_full_launch_audit,
            "LAUNCH_DETACHED_FULL": self.launch_detached_full,
            "VERIFY_NEW_FULL": self.verify_new_full,
            "SUCCESS": self.success,
            "REAL_BLOCKER": self.real_blocker,
        }
        return handlers[state]()

    def step_until_wait(self):
        state = self.current_state
        visited = 0
        while visited < 30:
            visited += 1
            self.current_state = state
            self.next_action = state
            self.heartbeat()
            try:
                next_state = self.dispatch(state)
            except Exception:
                self.last_exception = traceback.format_exc()
                self.consecutive_failures += 1
                self.log_action(state, "handler exception", return_code=1, stderr=self.last_exception, next_state="REAL_BLOCKER")
                next_state = "REAL_BLOCKER" if self.consecutive_failures >= MAX_FAILURES else state
            self.current_state = next_state
            self.next_action = next_state
            self.heartbeat()
            if next_state == "REAL_BLOCKER" and state != "REAL_BLOCKER":
                self.current_state = "REAL_BLOCKER"
                self.real_blocker()
                return "REAL_BLOCKER"
            if next_state == "SUCCESS" and state != "SUCCESS":
                self.current_state = "SUCCESS"
                self.success()
                return "SUCCESS"
            if next_state == "MONITOR_ACTIVE_FULL":
                if state == "RECOVER_REAL_STATE":
                    state = next_state
                    continue
                return next_state
            if next_state in {"SUCCESS", "REAL_BLOCKER"}:
                return next_state
            state = next_state
        self.last_exception = "state transition limit exceeded"
        self.current_state = "REAL_BLOCKER"
        self.real_blocker()
        return "REAL_BLOCKER"

    def run_loop(self, once=False):
        self.control_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.write(json.dumps({"pid": self.self_pid, "start_time": self.start_time, "boot_id": boot_id()}))
            lock.flush()
            while True:
                final_state = self.step_until_wait()
                print(json.dumps(self.heartbeat(), sort_keys=True), flush=True)
                if once:
                    return 0 if final_state != "REAL_BLOCKER" else 2
                project = read_json(self.state_path, {})
                latest_iter = project.get("active_training_latest_iter") or 0
                sleep_seconds = self.interval
                if final_state == "MONITOR_ACTIVE_FULL" and latest_iter >= 29000:
                    sleep_seconds = min(sleep_seconds, 60)
                time.sleep(max(30, sleep_seconds))


def make_fake_log(path, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        (200, 0.760000, 0.700000),
        (1000, min(best, 0.870000), 0.800000),
        (30000, best, 0.810000),
    ]
    path.write_text("\n".join(
        f"iteration {i} : model1_mean_dice : {v:.6f}\niteration {i} : model1_mean_test_dice : {t:.6f}\niteration {i}: loss: 0.1 total_finite : True"
        for i, v, t in values
    ))


def make_fake_root(tmp, scenario, best=None, final=True, active=False, with_candidate=False):
    root = Path(tmp)
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "audit/stack_third_controller").mkdir(parents=True)
    (root / "train_promise12").mkdir()
    (root / "scripts").mkdir()
    (root / "train_promise12/train_promise12_bcp_fft.py").write_text("print('ok')\n")
    (root / "scripts/fake_launcher.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(root / "scripts/fake_launcher.sh", 0o755)
    log = root / "audit/fake_full/console.log"
    ckpt = root / "train_promise12/fake/self_train"
    ckpt.mkdir(parents=True)
    if best is not None:
        make_fake_log(log, best)
    if final:
        (ckpt / "unet_best_model.pth").write_text("x")
        (ckpt / "model1_iter_30000_dice_0.9000.pth").write_text("x")
        (ckpt / "model1_iter_30000_final.pth").write_text("x")
        (ckpt / "unet_final_model.pth").write_text("x")
    pid = os.getpid() if active else None
    atomic_json(root / STATE_REL, {
        "active_training_pid": pid,
        "active_training_experiment": "fake_full" if (active or best is not None) else None,
        "active_training_log": "audit/fake_full/console.log",
        "active_training_expected_checkpoint_dir": "train_promise12/fake/self_train",
        "active_training_status": "RUNNING" if active else "UNKNOWN",
    })
    if with_candidate:
        atomic_json(root / CONTROL_DIR_REL / "candidate_queue.json", {"candidates": [{
            "candidate_id": f"fake_candidate_{scenario}",
            "status": "pending",
            "launcher": "scripts/fake_launcher.sh",
            "experiment": f"fake_candidate_{scenario}_full",
            "start_iter": 0,
            "ramp_iters": 2000,
            "smoke_command": "bash scripts/fake_launcher.sh",
            "full_log": f"audit/fake_candidate_{scenario}_full/console.log",
            "checkpoint_dir": f"train_promise12/fake_candidate_{scenario}/self_train",
        }]})
    return root


def run_self_test(root):
    report = root / CONTROL_DIR_REL / "controller_self_test_report.md"
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        scenarios = [
            ("A_running", {"best": 0.88, "final": False, "active": True, "candidate": False}, "MONITOR_ACTIVE_FULL"),
            ("B_accept", {"best": 0.900000, "final": True, "active": False, "candidate": False}, "SUCCESS"),
            ("C_weak", {"best": 0.890000, "final": True, "active": False, "candidate": True}, "MONITOR_ACTIVE_FULL"),
            ("D_no_gain", {"best": 0.880000, "final": True, "active": False, "candidate": True}, "MONITOR_ACTIVE_FULL"),
            ("E_incomplete", {"best": 0.880000, "final": False, "active": False, "candidate": False}, "REAL_BLOCKER"),
        ]
        all_pass = True
        for name, cfg, expected in scenarios:
            fake_root = make_fake_root(Path(tmpdir) / name, name, cfg["best"], cfg["final"], cfg["active"], cfg["candidate"])
            ctrl = StackThirdController(fake_root, interval=300, simulate=True)
            result = ctrl.step_until_wait()
            passed = result == expected
            all_pass = all_pass and passed
            rows.append((name, expected, result, "PASS" if passed else "FAIL"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Stack Third Controller Self-Test Report\n\n"
        f"- time: `{now()}`\n"
        f"- overall: `{'PASS' if all_pass else 'FAIL'}`\n\n"
        "| scenario | expected | actual | status |\n"
        "|---|---|---|---|\n"
        + "\n".join(f"| `{a}` | `{b}` | `{c}` | `{d}` |" for a, b, c, d in rows)
        + "\n"
    )
    return 0 if all_pass else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test(ROOT)
    controller = StackThirdController(ROOT, args.interval)
    return controller.run_loop(once=args.once)


if __name__ == "__main__":
    sys.exit(main())
