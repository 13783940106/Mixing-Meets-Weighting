#!/usr/bin/env bash
set -euo pipefail
echo "Create a fresh conda env on 5090D; do not copy old env directory."
conda env create -f ../environment/environment_full.yml || echo "If old PyTorch lacks 5090D support, install a current CUDA/PyTorch build manually, then use pip_freeze/conda_list as reference."
