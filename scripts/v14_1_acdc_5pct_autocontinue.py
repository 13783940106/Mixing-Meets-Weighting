#!/usr/bin/env python3
"""Authorized V14.1 handoff: replay 5% parent, preflight, then launch unchanged transfer."""

import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/v14_1_acdc_5pct_autocontinue"
STATE = ROOT / "experiments/acdc_mask_adamix_transfer_state.json"
SOURCE = ROOT / "train_ACDC/v14_acdc_mask_adamix_transfer.py"
COMPONENTS = ROOT / "train_ACDC/v14_mask_adamix_components.py"
FULL_RUN = ROOT / "runs/beta_fft_acdc_5pct_mask_token_adamix_full_seed1337"
FROZEN = {
    SOURCE: "8cac120e030a5f21f0e9a827b60e3a6ecd0dae2d5df81c7bbcaf8016de6040e0",
    COMPONENTS: "75754fa75e6fd6fc01c97682262895998799c7ce3c0bc2dcbe058ef4c63dcaee",
    ROOT / "train_ACDC/Pre_model/label3/unet_best_model.pth": "3486fd3aecf5f30e77e7e15bec0841f56c87db6246a6dc50cc4dc6f2ee2087e0",
    Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/val.list"): "61b935c936d8e12f7889e82d954f421fa25fb95305e92a476096b3242de8777d",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_state(**fields):
    state = json.loads(STATE.read_text())
    state.update({"updated_at": datetime.now().astimezone().isoformat(), **fields})
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def fail(message):
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "REAL_BLOCKER.md").write_text(f"# REAL_BLOCKER\n\n`{message}`\n")
    update_state(stage="V14_1_ACDC_5PCT_BLOCKED", training_authorized=False, next_action=message)
    raise RuntimeError(message)


def run_command(command, log_path):
    with log_path.open("w") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        fail(f"COMMAND_FAILED_{log_path.name}_{result.returncode}")


def probe(name, enabled):
    run = ROOT / f"runs/v14_1_acdc_5pct_{name}_seed1337"
    log_path = AUDIT / f"{name}.console.log"
    if run.exists():
        text = (run / "train.log").read_text() if (run / "train.log").exists() else ""
        if "resume_check_ok" not in text or "global_step=2" not in text:
            fail(f"PARTIAL_PREFLIGHT_RUN_PRESENT_{name}")
    else:
        command = [
            "conda", "run", "-n", "beta_fft", "python", str(SOURCE),
            "--root_path", "/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC",
            "--labelnum", "3", "--seed", "1337", "--gpu", "0", "--max_iterations", "2", "--save_interval", "2",
            "--run_root", str(run), "--pretrained_model", str(ROOT / "train_ACDC/Pre_model/label3"),
            "--mask_token_enabled", str(enabled).lower(), "--adamix_enabled", str(enabled).lower(), "--mechanism_log_interval", "1",
        ]
        run_command(command, log_path)
    text = (run / "train.log").read_text()
    required = ["labelnum=3", "seed=1337", "max_iterations=2", "resume_check_ok", "global_step=2"]
    if not all(item in text for item in required):
        fail(f"PREFLIGHT_IDENTITY_OR_STRICT_LOAD_FAIL_{name}")
    if re.search(r"Traceback|CUDA error|out of memory|\bNaN\b|\bInf\b", text, flags=re.IGNORECASE):
        fail(f"PREFLIGHT_NONFINITE_OR_ENGINEERING_ERROR_{name}")
    if enabled:
        mask_rows = [json.loads(line.split("mask_token_endpoint=", 1)[1]) for line in text.splitlines() if "mask_token_endpoint=" in line]
        mix_rows = [json.loads(line.split("adamix_endpoint=", 1)[1]) for line in text.splitlines() if "adamix_endpoint=" in line]
        mask_ok = bool(mask_rows) and all(row["model1"]["finite"] and row["model1"]["gradient_finite"] for row in mask_rows) and any(row["model1"]["token_gradient_norm"] > 0 for row in mask_rows)
        mix_ok = bool(mix_rows) and all(row["finite"] for row in mix_rows) and any(row["patch_area_ratio"] > 0 and row["mixed_loss_contribution"] > 0 and row["model1_total_gradient_norm"] > 0 for row in mix_rows)
        if not (mask_ok and mix_ok):
            fail("ACDC_5PCT_ON_ENDPOINT_GATE_FAIL")
    return {"run": str(run.relative_to(ROOT)), "strict_load": True, "finite": True, "endpoints": enabled}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "audit/acdc_10pct_mask_token_adamix_metrics.json").exists():
        fail("TEN_PCT_FORMAL_AUDIT_NOT_COMPLETE")
    for path, expected in FROZEN.items():
        actual = sha(path)
        if actual != expected:
            fail(f"FROZEN_SHA_MISMATCH_{path.name}_{actual}")
    if not (ROOT / "audit/acdc_10pct_mask_token_adamix_default_off.md").exists():
        fail("TEN_PCT_DEFAULT_OFF_EVIDENCE_MISSING")

    # Hold the single-GPU lock throughout replay and preflight.  It is released
    # immediately before the formal launcher atomically acquires the same lock.
    lock_handle = (ROOT / "experiments/.gpu_job.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail("GPU_LOCK_BUSY_AFTER_TEN_PCT_COMPLETION")
    update_state(stage="V14_1_ACDC_5PCT_BASELINE_REPLAY", training_authorized=False, next_action="VALIDATION_ONLY_PARENT_REPLAY")
    run_command(["conda", "run", "-n", "beta_fft", "python", str(ROOT / "scripts/v14_1_acdc_5pct_baseline_replay.py")], AUDIT / "baseline_replay.console.log")
    replay = json.loads((ROOT / "audit/v14_1_acdc_5pct_baseline_replay/baseline_replay.json").read_text())
    if replay["status"] != "ACDC_5PCT_BASELINE_REUSE_PASS":
        fail("ACDC_5PCT_BASELINE_REPLAY_FAIL")

    update_state(stage="V14_1_ACDC_5PCT_PREFLIGHT", training_authorized=False, next_action="UNCHANGED_TRANSFER_PREFLIGHT")
    off = probe("defaultoff_probe", False)
    on = probe("on_smoke2", True)
    preflight = {
        "status": "ACDC_5PCT_UNCHANGED_TRANSFER_PREFLIGHT_PASS",
        "baseline": "ACDC_5PCT_BASELINE_REUSE_PASS / NO_RERUN",
        "seed": 1337, "labelnum": 3, "budget": 30000,
        "source_sha256": sha(SOURCE), "components_sha256": sha(COMPONENTS),
        "default_off": {"reused_static_and_dynamic_core": "audit/acdc_10pct_mask_token_adamix_default_off.md", "five_pct_entry_probe": off},
        "on_smoke": on,
    }
    (AUDIT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (AUDIT / "preflight.md").write_text(
        "# V14.1 ACDC 5% unchanged-transfer preflight\n\n"
        "`ACDC_5PCT_UNCHANGED_TRANSFER_PREFLIGHT_PASS`\n\n"
        "Parent replay: `ACDC_5PCT_BASELINE_REUSE_PASS / NO_RERUN`. Seed=1337, labelnum=3. "
        "The exact 10% candidate source/component SHA pair is reused unchanged. OFF entry/strict-load and ON forward/loss/backward/strict-load/finite/endpoints passed.\n"
    )
    if FULL_RUN.exists():
        fail("ACDC_5PCT_FULL_RUN_ROOT_ALREADY_EXISTS")
    fcntl.flock(lock_handle, fcntl.LOCK_UN)
    lock_handle.close()

    for pid_path in (ROOT / "experiments/acdc_5pct_mask_adamix_launcher.pid", ROOT / "experiments/acdc_5pct_mask_adamix_controller.pid"):
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
                fail(f"LIVE_STALE_PID_FILE_{pid_path.name}_{pid}")
            except ProcessLookupError:
                pid_path.unlink()
            except ValueError:
                fail(f"INVALID_PID_FILE_{pid_path.name}")
    launch_log = (ROOT / "audit/acdc_5pct_mask_token_adamix_full_console.log").open("w")
    launcher = subprocess.Popen([str(ROOT / "scripts/v14_1_acdc_5pct_launch.sh")], cwd=ROOT, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    if launcher.poll() is not None:
        fail(f"ACDC_5PCT_LAUNCHER_EXITED_EARLY_{launcher.returncode}")
    controller_log = (ROOT / "audit/acdc_5pct_mask_token_adamix_controller.log").open("w")
    controller = subprocess.Popen(["conda", "run", "-n", "beta_fft", "python", str(ROOT / "scripts/v14_1_acdc_5pct_controller.py")], cwd=ROOT, stdout=controller_log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if (ROOT / "experiments/acdc_5pct_mask_adamix_controller.pid").exists():
            break
        time.sleep(1)
    if not (ROOT / "experiments/acdc_5pct_mask_adamix_controller.pid").exists():
        fail("ACDC_5PCT_CONTROLLER_DID_NOT_START")
    actual_controller = int((ROOT / "experiments/acdc_5pct_mask_adamix_controller.pid").read_text().strip())
    update_state(
        stage="V14_1_ACDC_5PCT_FULL_RUNNING", training_authorized=True,
        five_pct_parent_best_validation_mDice=replay["best_validation_raw"]["dice"], five_pct_parent_best_iteration=replay["best_iteration"],
        active_launcher_pid=launcher.pid, active_controller_pid=actual_controller, active_training_pid=None,
        gpu_concurrency=1, next_action="MONITOR_5PCT_TO_NATURAL_COMPLETION_THEN_AUTO_AUDIT",
    )
    with (ROOT / "experiments/acdc_mask_adamix_transfer_registry.tsv").open("a") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat()}\tFULL\tRUNNING\t{FULL_RUN.relative_to(ROOT)}\tACDC_5P_BETA_FFT_2D_MASK_TOKEN_ADAMIX\t1337\tofficial_labelnum3\tNA\t0\tUNCHANGED_TRANSFER_LAUNCHED\tbaseline reuse pass; preflight pass; exact 10% source SHA pair\n")
    (AUDIT / "launch.json").write_text(json.dumps({"launcher_pid": launcher.pid, "controller_wrapper_pid": controller.pid, "controller_pid": actual_controller, "run": str(FULL_RUN.relative_to(ROOT))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
