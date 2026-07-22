#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${PROJECT_ROOT}/train_promise12"
DATA_ROOT="${PB2D_DATA_ROOT:-${PROJECT_ROOT}/data_staging/promise12/20260624_141558/data_promise12}"
EXP="${PB2D_EXP:-pb2d_original_2d_adamix_bcp_lite_v1}"
GPU="${PB2D_GPU:-0}"

EXTRA_ARGS=()
if [[ "${PB2D_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --pretrain_max_iterations_override "${PB2D_PRETRAIN_ITERS:-2}"
    --self_train_max_iterations_override "${PB2D_SELFTRAIN_ITERS:-20}"
    --adamix_bcp_lite_v1_log_interval "${PB2D_ADAMIX_BCP_LOG_INTERVAL:-5}"
  )
fi

if [[ "${PB2D_5K:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --pretrain_max_iterations_override "${PB2D_PRETRAIN_ITERS:-500}"
    --self_train_max_iterations_override 5000
  )
fi

cd "${TRAIN_DIR}"

conda run --no-capture-output -n "${PB2D_CONDA_ENV:-beta_fft}" python train_promise12_bcp_fft.py \
  --root_path "${DATA_ROOT}" \
  --model_1 unet \
  --model_2 swin_unet \
  --max_iterations 30000 \
  --batch_size 16 \
  --labeled_bs 8 \
  --labeled_num 7 \
  --base_lr 0.01 \
  --seed 1337 \
  --gpu "${GPU}" \
  --deterministic 1 \
  --use_adamix_bcp_lite_v1 \
  --adamix_bcp_lite_v1_min_side "${PB2D_ADAMIX_BCP_MIN_SIDE:-0.18}" \
  --adamix_bcp_lite_v1_max_side "${PB2D_ADAMIX_BCP_MAX_SIDE:-0.38}" \
  --adamix_bcp_lite_v1_log_interval "${PB2D_ADAMIX_BCP_LOG_INTERVAL:-200}" \
  --exp "${EXP}" \
  "${EXTRA_ARGS[@]}"
