#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
exec 9>experiments/.gpu_job.lock
flock -n 9
echo "$$" > experiments/acdc_5pct_mask_adamix_launcher.pid
exec conda run -n beta_fft python train_ACDC/v14_acdc_mask_adamix_transfer.py \
  --root_path /home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC \
  --labelnum 3 --seed 1337 --gpu 0 --max_iterations 30000 --save_interval 200 \
  --run_root /home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/beta_fft_acdc_5pct_mask_token_adamix_full_seed1337 \
  --pretrained_model /home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label3 \
  --mask_token_enabled true --adamix_enabled true --mechanism_log_interval 200
