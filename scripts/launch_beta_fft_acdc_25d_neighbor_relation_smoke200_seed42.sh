#!/bin/bash
# beta-FFT ACDC neighbor relation smoke200 only.
# NOT FULL. NOT DETACHED. NOT RESUME.
set -euo pipefail
PROJECT_ROOT="/home/renguanming/jupyterlab/rgm/beta-FFT-main"
RUN_ROOT="${PROJECT_ROOT}/runs/beta_fft_acdc_25d_neighbor_relation_smoke200_seed42"
EXP_NAME="beta_fft_acdc_25d_neighbor_relation_smoke200_seed42"
CONSOLE_LOG="${PROJECT_ROOT}/audit/beta_fft_neighbor_relation_smoke200_console.log"
if pgrep -f "train_ACDC/beta_fft_acdc_repro.py.*${EXP_NAME}" >/dev/null 2>&1; then
  echo "ERROR: existing smoke200 process found for ${EXP_NAME}"
  pgrep -af "train_ACDC/beta_fft_acdc_repro.py.*${EXP_NAME}" || true
  exit 1
fi
if [ -d "${RUN_ROOT}" ] && [ -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: run_root exists and is non-empty: ${RUN_ROOT}"
  exit 1
fi
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/logs" "${RUN_ROOT}/audit"
cd "${PROJECT_ROOT}"
source /opt/conda/etc/profile.d/conda.sh
conda activate beta_fft
python train_ACDC/beta_fft_acdc_repro.py \
  --root_path /home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC \
  --exp "${EXP_NAME}" \
  --model unet \
  --max_iterations 200 \
  --batch_size 24 \
  --labeled_bs 12 \
  --labelnum 7 \
  --base_lr 0.01 \
  --seed 42 \
  --gpu 0 \
  --consistency 0.1 \
  --consistency_rampup 200.0 \
  --magnitude 6.0 \
  --s_param 6 \
  --patch_size 64 \
  --h_size 4 \
  --w_size 4 \
  --top_num 4 \
  --data_manifest_name official \
  --pretrained_model train_ACDC/Pre_model/label7 \
  --input_mode 25d_neighbor_relation \
  --context_slices 3 \
  --relation_type center_anchored_neighbor \
  --relation_stage x0 \
  --relation_dim 8 \
  --save_interval 200 \
  --run_root "${RUN_ROOT}" \
  2>&1 | tee "${CONSOLE_LOG}"
