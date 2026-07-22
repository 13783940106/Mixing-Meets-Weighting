#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${PROJECT_ROOT}/train_promise12"
DATA_ROOT="${PB2D_DATA_ROOT:-${PROJECT_ROOT}/data_staging/promise12/20260624_141558/data_promise12}"
EXP="${PB2D_EXP:-pb2d_original_2d_combo_adamix_bcp_ema_bcp_anchor_lite_v1}"
GPU="${PB2D_GPU:-0}"

EXTRA_ARGS=()
if [[ "${PB2D_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --pretrain_max_iterations_override "${PB2D_PRETRAIN_ITERS:-2}"
    --self_train_max_iterations_override "${PB2D_SELFTRAIN_ITERS:-20}"
    --adamix_bcp_lite_v1_log_interval "${PB2D_ADAMIX_BCP_LOG_INTERVAL:-5}"
    --ema_bcp_anchor_lite_v1_log_interval "${PB2D_EMA_BCP_ANCHOR_LOG_INTERVAL:-5}"
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
  --use_ema_bcp_anchor_lite_v1 \
  --ema_bcp_anchor_lite_v1_weight "${PB2D_EMA_BCP_ANCHOR_WEIGHT:-0.02}" \
  --ema_bcp_anchor_lite_v1_start_iter "${PB2D_EMA_BCP_ANCHOR_START_ITER:-1000}" \
  --ema_bcp_anchor_lite_v1_ramp_iters "${PB2D_EMA_BCP_ANCHOR_RAMP_ITERS:-2000}" \
  --ema_bcp_anchor_lite_v1_conf_floor "${PB2D_EMA_BCP_ANCHOR_CONF_FLOOR:-0.25}" \
  --ema_bcp_anchor_lite_v1_log_interval "${PB2D_EMA_BCP_ANCHOR_LOG_INTERVAL:-200}" \
  --exp "${EXP}" \
  "${EXTRA_ARGS[@]}"
