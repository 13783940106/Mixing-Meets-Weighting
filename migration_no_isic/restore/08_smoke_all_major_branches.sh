#!/usr/bin/env bash
set -euo pipefail
echo "Read-only smoke guide: build models/load selected checkpoints/one forward only. Do NOT start full training or Formal Test."
python restore/02_verify_gpu_environment.py
python restore/07_verify_registries.py
python restore/06_verify_all_checkpoints.py || echo "Checkpoint full verification may require paths after extraction; inspect output."
