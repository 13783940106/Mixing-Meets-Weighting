#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42"
STATE = RUN / "audit/monitor_state.json"
TIMELINE = RUN / "audit/monitor_timeline.tsv"
LAUNCH_PID = ROOT / "experiments/acdc_mask_adamix_launcher.pid"
CONTROLLER_PID = ROOT / "experiments/acdc_mask_adamix_controller.pid"


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


CONTROLLER_PID.write_text(str(os.getpid()) + "\n")
while not LAUNCH_PID.exists():
    time.sleep(1)
launcher = int(LAUNCH_PID.read_text().strip())
TIMELINE.parent.mkdir(parents=True, exist_ok=True)
if not TIMELINE.exists():
    TIMELINE.write_text("timestamp\tlauncher_alive\titeration\tbest_dsc\tbest_iter\terror\tgpu\n")

last_iteration = 0
best_dsc, best_iter = 0.0, None
error = ""
while True:
    text = (RUN / "train.log").read_text() if (RUN / "train.log").exists() else ""
    iterations = [int(value) for value in re.findall(r"iteration (\d+): loss", text)]
    if iterations: last_iteration = max(iterations)
    for match in re.finditer(r"best_checkpoint_updated .* global_step=(\d+) metric=([0-9.]+)", text):
        iteration, score = int(match.group(1)), float(match.group(2))
        if score >= best_dsc: best_dsc, best_iter = score, iteration
    failures = re.findall(r"(Traceback|CUDA error|out of memory|NaN|Inf)", text, flags=re.IGNORECASE)
    error = failures[-1] if failures else ""
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    except Exception:
        gpu = "UNKNOWN"
    payload = {"updated_at": datetime.now().astimezone().isoformat(), "launcher_pid": launcher, "controller_pid": os.getpid(), "launcher_alive": alive(launcher), "iteration": last_iteration, "target_iteration": 30000, "best_validation_mDice": best_dsc, "best_iteration": best_iter, "error": error, "gpu": gpu}
    STATE.write_text(json.dumps(payload, indent=2) + "\n")
    with TIMELINE.open("a") as handle:
        handle.write(f"{payload['updated_at']}\t{int(payload['launcher_alive'])}\t{last_iteration}\t{best_dsc}\t{best_iter}\t{error}\t{gpu}\n")
    if not alive(launcher): break
    time.sleep(30)

text = (RUN / "train.log").read_text() if (RUN / "train.log").exists() else ""
normal = last_iteration >= 30000 and "resume_check_ok" in text and not error
if normal:
    command = ["conda", "run", "-n", "beta_fft", "python", str(ROOT / "scripts/v14_acdc_mask_adamix_post_audit.py")]
    result = subprocess.run(command, cwd=ROOT, stdout=(RUN / "audit/post_audit.log").open("w"), stderr=subprocess.STDOUT)
    if result.returncode != 0:
        (RUN / "audit/POST_AUDIT_FAILED").write_text(str(result.returncode) + "\n")
else:
    (RUN / "audit/TRAINING_NOT_NATURALLY_COMPLETE").write_text(json.dumps({"iteration": last_iteration, "error": error}) + "\n")
CONTROLLER_PID.unlink(missing_ok=True)
