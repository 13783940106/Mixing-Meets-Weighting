# Original experiment environment

| Component | Version / hardware |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| CUDA runtime | 12.1 |
| GPU | NVIDIA A100-SXM4-40GB |
| Project dependencies | `requirements-core.txt` |

Create the environment:

```bash
conda create -n beta_fft_paper python=3.10.20 pip
conda activate beta_fft_paper
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt
```

`environment_beta_fft.yml` provides the earlier migration export. Use the [RTX 5090/5090D recipe](environment-5090.md) for the separate Blackwell environment.
