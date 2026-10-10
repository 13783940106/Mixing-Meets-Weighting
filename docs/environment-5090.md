# RTX 5090 / 5090D installation recipe

Create a separate Conda environment with CUDA 12.8 PyTorch wheels. Blackwell support is described in the [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/); the pinned versions below appear in the [official installation matrix](https://pytorch.org/get-started/previous-versions/#v271).

```bash
conda create -n beta_fft_5090 python=3.10.20 pip
conda activate beta_fft_5090
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-core.txt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Use the final environment variable when loading this project's trusted full-state checkpoints containing optimizer and NumPy RNG state with PyTorch 2.6+. See the [serialization notes](https://docs.pytorch.org/docs/2.7/notes/serialization.html#environment-variables). The host NVIDIA driver must support CUDA 12.8 and the GPU; the PyTorch wheel supplies the CUDA runtime.

Set dataset, initialization and output paths as shown in the [README](../README.md). The original experiment environment is documented [separately](environment-original.md).

Validation status: installation recipe; project execution and numerical comparison on RTX 5090 are pending.
