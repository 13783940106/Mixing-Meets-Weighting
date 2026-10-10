# Recorded experiment environment

The June 24, 2026 project bootstrap record (`audit/project_bootstrap_20260624_141325/environment_snapshot.txt`) records Python 3.10.20, PyTorch 2.4.1+cu121, CUDA 12.1 and an NVIDIA A100-SXM4-40GB. Its numeric library versions agree with `requirements-core.txt` where recorded. The October 7 `tcsmix_control_revision_20261006/formal_R1_20261007/reports/environment_freeze.txt` additionally records torchvision 0.19.1+cu121 and the remaining core versions pinned there. These are saved environment records, not a new installation or a claim that every imported historical file was archived.

The existing `environment_beta_fft.yml` remains available as an older migration export. Its CUDA dependencies describe the original A100 environment. The base-environment migration export is not the training environment.

For the original hardware/software profile:

```bash
conda create -n beta_fft_paper python=3.10.20 pip
conda activate beta_fft_paper
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt
```

No package installation, GPU initialization or training/evaluation was performed for this publication update. Only source and metadata were checked.
