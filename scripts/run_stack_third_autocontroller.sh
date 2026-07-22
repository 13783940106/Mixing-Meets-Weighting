#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p audit/stack_third_controller

PID_FILE="audit/stack_third_controller/controller.pid"
LOG_FILE="audit/stack_third_controller/controller.log"

if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && ps -p "$OLD_PID" -o cmd= 2>/dev/null | rg -q "stack_third_autocontroller.py"; then
    echo "$OLD_PID"
    exit 0
  fi
fi

setsid /opt/conda/bin/python scripts/stack_third_autocontroller.py --interval "${STACK_THIRD_CONTROLLER_INTERVAL:-300}" \
  >> "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"
echo "$PID"
