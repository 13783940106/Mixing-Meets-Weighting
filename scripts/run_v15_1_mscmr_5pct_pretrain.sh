#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/renguanming/jupyterlab/rgm/beta-FFT-main"
RUN="$ROOT/runs/mscmr_5pct_pretrain_bayeseg_fixed_split_seed1337"
PY="/home/renguanming/.conda/envs/beta_fft/bin/python"
mkdir -p "$RUN"
tmux new-session -d -s mscmr_run2_5pct_pretrain "$PY -u $ROOT/train_MSCMRSEG/v15_1_mscmr_pretrain.py --data_root /home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw --labeled_manifest $ROOT/audit/mscmr_5pct_labeled_manifest.tsv --validation_manifest $ROOT/audit/mscmr_split_manifest.tsv --validation_role validation --label_ratio 5pct --split bayeseg_fixed_split --seed 1337 --gpu 0 --max_iterations 10000 --batch_size 4 --image_size 256 256 --base_lr 0.01 --validation_interval 200 --checkpoint_interval 1000 --run_root $RUN --num_workers 0 2>&1 | tee -a $RUN/launcher_stdout.log"
sleep 2
PID="$(pgrep -f "v15_1_mscmr_pretrain.py.*--run_root $RUN" | head -n1)"; test -n "$PID"; echo "$PID" > "$RUN/training.pid"
tmux new-session -d -s mscmr_run2_5pct_pretrain_controller "$PY -u $ROOT/train_MSCMRSEG/v15_1_mscmr_monitor.py --run_root $RUN --training_pid $PID --max_iterations 10000 --validation_interval 200 --heartbeat_seconds 30 2>&1 | tee -a $RUN/controller_console.log"
sleep 2
CPID="$(pgrep -f "v15_1_mscmr_monitor.py.*--run_root $RUN" | head -n1)"; test -n "$CPID"; echo "$CPID" > "$RUN/controller.pid"
printf 'RUN=%s\nTRAINING_PID=%s\nCONTROLLER_PID=%s\n' "$RUN" "$PID" "$CPID"
