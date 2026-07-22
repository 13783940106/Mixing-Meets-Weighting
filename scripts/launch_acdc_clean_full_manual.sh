#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/renguanming/jupyterlab/rgm/beta-FFT-main
TRAIN_DIR="$ROOT/train_ACDC"
DATA_ROOT=/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC
RUN_ROOT=/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/beta_fft_acdc_clean_full_seed42
PRETRAINED_DIR=$ROOT/train_ACDC/Pre_model/clean_label12_seed42
PRETRAINED_CKPT=$PRETRAINED_DIR/unet_best_model.pth
LABELED_LIST=$ROOT/data/acdc/splits_clean_beta_fft/labeled_clean_beta_fft.list
UNLABELED_LIST=$ROOT/data/acdc/splits_clean_beta_fft/unlabeled_clean_beta_fft.list
VAL_LIST=$ROOT/data/acdc/splits_clean_beta_fft/val_clean_beta_fft.list
TEST_LIST=$ROOT/data/acdc/splits_clean_beta_fft/test_clean_beta_fft.list

cd "$TRAIN_DIR"
test -d "$DATA_ROOT"
test -f "$LABELED_LIST"
test -f "$UNLABELED_LIST"
test -f "$VAL_LIST"
test -f "$TEST_LIST"
test -e "$PRETRAINED_CKPT" || {
  echo "ERROR: clean supervised pretrained model not found: $PRETRAINED_CKPT"
  exit 1
}
if [ -d "$RUN_ROOT" ] && [ "$(find "$RUN_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: RUN_ROOT exists and is non-empty: $RUN_ROOT"
  exit 1
fi
if pgrep -af "beta_fft_acdc_repro.py.*beta_fft_acdc_clean_full_seed42" >/dev/null; then
  echo "ERROR: matching clean full training process already exists"
  exit 1
fi
mkdir -p "$RUN_ROOT"

setsid conda run --no-capture-output -n beta_fft \
  python beta_fft_acdc_repro.py \
  --root_path "$DATA_ROOT" \
  --labelnum 12 \
  --seed 42 \
  --gpu 0 \
  --max_iterations 30000 \
  --save_interval 200 \
  --split_name clean \
  --labeled_list "$LABELED_LIST" \
  --unlabeled_list "$UNLABELED_LIST" \
  --val_list "$VAL_LIST" \
  --test_list "$TEST_LIST" \
  --run_root "$RUN_ROOT" \
  --pretrained_model "$PRETRAINED_DIR" \
  > "$RUN_ROOT/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUN_ROOT/launcher.pid"
echo "launched pid=$PID"
echo "log=$RUN_ROOT/train.log"
