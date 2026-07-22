#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONTROL_DIR="audit/stack_third_controller"
PID_FILE="${CONTROL_DIR}/controller.pid"
STATE_FILE="${CONTROL_DIR}/controller_state.json"
WATCHDOG_PID_FILE="${CONTROL_DIR}/watchdog.pid"
WATCHDOG_STATE_FILE="${CONTROL_DIR}/watchdog_state.json"
WATCHDOG_LOG="${CONTROL_DIR}/watchdog.log"
mkdir -p "$CONTROL_DIR"

if [[ "${STACK_THIRD_WATCHDOG_CHILD:-0}" != "1" ]]; then
  if [[ -s "$WATCHDOG_PID_FILE" ]]; then
    OLD_PID="$(cat "$WATCHDOG_PID_FILE" || true)"
    if [[ -n "$OLD_PID" ]] && ps -p "$OLD_PID" -o cmd= 2>/dev/null | rg -q "stack_third_controller_watchdog.sh"; then
      echo "$OLD_PID"
      exit 0
    fi
  fi
  STACK_THIRD_WATCHDOG_CHILD=1 setsid bash "$0" >> "$WATCHDOG_LOG" 2>&1 < /dev/null &
  PID="$!"
  echo "$PID" > "$WATCHDOG_PID_FILE"
  echo "$PID"
  exit 0
fi

while true; do
  NOW="$(date +%s)"
  NEED_RESTART=0
  PID=""
  if [[ -s "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" || true)"
  fi
  if [[ -z "$PID" ]] || ! ps -p "$PID" -o cmd= 2>/dev/null | rg -q "stack_third_autocontroller.py"; then
    NEED_RESTART=1
  fi
  if [[ -s "$STATE_FILE" ]]; then
    HEARTBEAT="$(/opt/conda/bin/python - "$STATE_FILE" <<'PY'
import json, sys
from datetime import datetime
p=sys.argv[1]
try:
    data=json.load(open(p))
    print(data.get("heartbeat_at",""))
except Exception:
    print("")
PY
)"
    if [[ -n "$HEARTBEAT" ]]; then
      HB_EPOCH="$(date -d "$HEARTBEAT" +%s 2>/dev/null || echo 0)"
      AGE=$(( NOW - HB_EPOCH ))
      if (( AGE > 900 )); then
        NEED_RESTART=1
      fi
    fi
  fi
  if (( NEED_RESTART == 1 )); then
    echo "$(date --iso-8601=seconds) restarting controller"
    bash scripts/run_stack_third_autocontroller.sh >/dev/null || true
  fi
  /opt/conda/bin/python - "$WATCHDOG_STATE_FILE" "$PID" "$NEED_RESTART" "$$" <<'PY'
import json, os, sys
from datetime import datetime

path, controller_pid, need_restart, watchdog_pid = sys.argv[1:5]
payload = {
    "watchdog_pid": int(watchdog_pid),
    "writer_pid": os.getpid(),
    "heartbeat_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "controller_pid_observed": controller_pid or None,
    "restart_requested": bool(int(need_restart)),
}
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
  sleep 600
done
