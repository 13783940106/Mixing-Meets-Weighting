#!/bin/bash
# beta-FFT ACDC 2.5D 3-channel Full Training Launch Script
# generated_at: 2026-06-18
# updated_for_sharedaug: 2026-06-18
# status: MANUAL_LAUNCH_ONLY - starts detached via setsid after safety checks

set -euo pipefail

PROJECT_ROOT="/home/renguanming/jupyterlab/rgm/beta-FFT-main"
RUN_ROOT="${PROJECT_ROOT}/runs/beta_fft_acdc_25d_3ch_official_full_seed42"
PID_FILE="${RUN_ROOT}/launcher.pid"
RESUME_CKPT="${1:-}"
EXP_NAME="beta_fft_acdc_25d_3ch_official_full_seed42"

# ============== SAFETY CHECKS ==============
if pgrep -f "train_ACDC/beta_fft_acdc_repro.py.*${EXP_NAME}" >/dev/null 2>&1; then
    echo "ERROR: Existing ${EXP_NAME} training process found. Refusing to launch duplicate."
    pgrep -af "train_ACDC/beta_fft_acdc_repro.py.*${EXP_NAME}" || true
    exit 1
fi

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "ERROR: Existing training process running with PID ${OLD_PID}"
        exit 1
    fi
fi

if [ -z "${RESUME_CKPT}" ]; then
    if [ -d "${RUN_ROOT}" ] && [ -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        echo "ERROR: ${RUN_ROOT} already exists and is non-empty. Refusing fresh full launch."
        exit 1
    fi
else
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "ERROR: resume checkpoint does not exist: ${RESUME_CKPT}"
        exit 1
    fi
fi

# ============== PATH CHECKS ==============
REQUIRED_PATHS=(
    "${PROJECT_ROOT}/train_ACDC/beta_fft_acdc_repro.py"
    "${PROJECT_ROOT}/train_ACDC/Pre_model/label7/unet_best_model.pth"
    "/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/data"
    "${PROJECT_ROOT}/audit/beta_fft_25d_3ch_shared_augmentation_audit.md"
    "${PROJECT_ROOT}/audit/beta_fft_25d_3ch_sharedaug_smoke200_report.md"
)
for p in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: Required path not found: $p"
        exit 1
    fi
done

# ============== SETUP ==============
mkdir -p "${RUN_ROOT}/"{checkpoints,logs,audit}

RESUME_ARGS=""
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_ARGS="--resume_ckpt ${RESUME_CKPT}"
fi

cat <<EOF
============================================
beta-FFT ACDC 2.5D 3-channel Full Training
run_root=${RUN_ROOT}
input_mode=25d_3ch
seed=42
max_iterations=30000
save_interval=200
resume_ckpt=${RESUME_CKPT:-None}
started at: $(date)
============================================
EOF

# ============== DETACHED LAUNCH ==============
export PROJECT_ROOT RUN_ROOT PID_FILE RESUME_ARGS EXP_NAME
setsid bash -lc '
set -euo pipefail
echo $$ > "${PID_FILE}"
cd "${PROJECT_ROOT}"
conda run --no-capture-output -n beta_fft python train_ACDC/beta_fft_acdc_repro.py \
  --root_path /home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC \
  --exp "${EXP_NAME}" \
  --model unet \
  --max_iterations 30000 \
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
  --input_mode 25d_3ch \
  --context_slices 3 \
  --save_interval 200 \
  --run_root "${RUN_ROOT}" \
  ${RESUME_ARGS} \
  2>&1 | tee "${RUN_ROOT}/train_console.log"
exit_code=${PIPESTATUS[0]}
rm -f "${PID_FILE}"
exit ${exit_code}
' >/dev/null 2>&1 &

LAUNCHER_PID=$!
sleep 2
if [ -f "${PID_FILE}" ]; then
    TRAIN_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
else
    TRAIN_PID="${LAUNCHER_PID}"
fi

echo "Detached launcher started. launcher_pid=${LAUNCHER_PID} train_pid=${TRAIN_PID}"
echo "train_console=${RUN_ROOT}/train_console.log"
echo "train_log=${RUN_ROOT}/train.log"
