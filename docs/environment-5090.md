# Separate RTX 5090 / 5090D installation

Use a fresh environment, rather than copying a Conda environment directory or the A100 CUDA 12.1 wheel set. PyTorch introduced Blackwell support with its [2.7 release and CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/). The following pinned alternative is listed in the [official installation matrix](https://pytorch.org/get-started/previous-versions/#v271):

```bash
conda create -n beta_fft_5090 python=3.10.20 pip
conda activate beta_fft_5090
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-core.txt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

The final environment variable retains legacy loading of this project's trusted checkpoints containing optimizer and NumPy RNG state on PyTorch 2.6+. See [PyTorch serialization notes](https://docs.pytorch.org/docs/2.7/notes/serialization.html#environment-variables). Use a host NVIDIA driver compatible with CUDA 12.8 and this GPU; the wheel includes the CUDA runtime. Checkpoint paths are supplied separately, as shown in the README.

This is an installation recipe, not the original experiment environment. It has not been installed or validated on a 5090 during this code-only update; numerical parity and execution on this hardware have not been tested.
