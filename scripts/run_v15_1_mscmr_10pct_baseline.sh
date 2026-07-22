#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${MSCMR_RUN_ROOT:-$PROJECT_ROOT/runs/mscmr_10pct_beta_fft_2d_baseline_bayeseg_fixed_split_seed1337}"
PYTHON_BIN="/home/renguanming/.conda/envs/beta_fft/bin/python"
if [[ -e "$RUN_ROOT/training.pid" ]] && kill -0 "$(cat "$RUN_ROOT/training.pid")" 2>/dev/null; then echo "active training already exists" >&2; exit 2; fi
mkdir -p "$RUN_ROOT"; cd "$PROJECT_ROOT"
setsid env PYTHONUNBUFFERED=1 "$PYTHON_BIN" train_MSCMRSEG/v15_1_mscmr_beta_fft_baseline.py \
  --data_root /home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw \
  --labeled_manifest audit/mscmr_10pct_labeled_manifest.tsv --unlabeled_manifest audit/mscmr_10pct_unlabeled_manifest.tsv \
  --validation_manifest audit/mscmr_split_manifest.tsv --validation_role validation --label_ratio 10pct \
  --split bayeseg_fixed_split --seed 1337 --gpu 0 --max_iterations 30000 --batch_size 8 --labeled_batch_size 4 \
  --image_size 256 256 --base_lr 0.01 --u_weight 0.5 --ema_decay 0.99 --consistency 0.1 --consistency_rampup 30000 \
  --fft_radius 30 --validation_interval 200 --checkpoint_interval 1000 \
  --pretrain_checkpoint runs/mscmr_10pct_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt \
  --run_root "$RUN_ROOT" >>"$RUN_ROOT/launcher_stdout.log" 2>&1 < /dev/null &
sleep 1
TRAINING_PID="$(pgrep -f "v15_1_mscmr_beta_fft_baseline.py.*--run_root $RUN_ROOT" | head -n 1)"; test -n "$TRAINING_PID"
echo "$TRAINING_PID" > "$RUN_ROOT/training.pid"
setsid env PYTHONUNBUFFERED=1 "$PYTHON_BIN" train_MSCMRSEG/v15_1_mscmr_monitor.py \
  --run_root "$RUN_ROOT" --training_pid "$TRAINING_PID" --max_iterations 30000 --validation_interval 200 --heartbeat_seconds 30 \
  >>"$RUN_ROOT/controller_console.log" 2>&1 < /dev/null &
sleep 1
CONTROLLER_PID="$(pgrep -f "v15_1_mscmr_monitor.py.*--run_root $RUN_ROOT" | head -n 1)"; test -n "$CONTROLLER_PID"
echo "$CONTROLLER_PID" > "$RUN_ROOT/controller.pid.launcher"
printf 'RUN_ROOT=%s\nTRAINING_PID=%s\nCONTROLLER_PID=%s\n' "$RUN_ROOT" "$TRAINING_PID" "$CONTROLLER_PID"
