#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/renguanming/jupyterlab/rgm/beta-FFT-main
TRAIN_DIR=$ROOT/train_ACDC
DATA_ROOT=/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC
RUN_ROOT=$ROOT/runs/beta_fft_acdc_clean_supervised_pretrain_seed42
PRE_DIR=$TRAIN_DIR/Pre_model/clean_label12_seed42
TARGET_WEIGHT=$PRE_DIR/unet_best_model.pth
LABELED_LIST=$ROOT/data/acdc/splits_clean_beta_fft/labeled_clean_beta_fft.list
VAL_LIST=$ROOT/data/acdc/splits_clean_beta_fft/val_clean_beta_fft.list
ENTRY=$TRAIN_DIR/pretrain_acdc_clean_supervised.py

cd "$TRAIN_DIR"
test -d "$DATA_ROOT"
test -f "$LABELED_LIST"
test -f "$VAL_LIST"
test -f "$ENTRY"
conda env list | grep -qE "^beta_fft[[:space:]]"
if pgrep -af "pretrain_acdc_clean_supervised.py.*beta_fft_acdc_clean_supervised_pretrain_seed42" >/dev/null; then
  echo "ERROR: matching clean supervised pretraining process already exists"
  exit 1
fi
if [ -d "$RUN_ROOT" ] && [ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: run directory is not empty: $RUN_ROOT"
  exit 1
fi
if [ -e "$TARGET_WEIGHT" ]; then
  echo "ERROR: target pretrained weight already exists: $TARGET_WEIGHT"
  exit 1
fi
mkdir -p "$RUN_ROOT" "$PRE_DIR"

setsid conda run --no-capture-output -n beta_fft \
  python "$ENTRY" \
  --root_path "$DATA_ROOT" \
  --labeled_list "$LABELED_LIST" \
  --val_list "$VAL_LIST" \
  --labelnum 12 \
  --seed 42 \
  --gpu 0 \
  --output_dir "$PRE_DIR" \
  --run_root "$RUN_ROOT" \
  --model unet \
  --pre_iterations 10000 \
  --batch_size 24 \
  --labeled_bs 12 \
  --base_lr 0.01 \
  --num_classes 4 \
  > "$RUN_ROOT/train.log" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$RUN_ROOT/launcher.pid"
echo "launched pid=$PID"
echo "log=$RUN_ROOT/train.log"
echo "target_weight=$TARGET_WEIGHT"
