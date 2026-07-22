# Code and no-ISIC package restore guide

- Git repository: source code, configs, scripts, lightweight experiment metadata, environment files and restore helpers.
- No-ISIC `tar.zst` packages: PROMISE12/PB2D, ACDC, MSCMR and other non-ISIC experiment runs, logs and checkpoints.
- Raw medical datasets are not packed; mount them on the 5090D server and create symlinks according to `DATASET_PATH_MAP_NO_ISIC.tsv`.
- ISIC2018 experiment assets are not migrated in this package. The full package remains on the source server at `/home/renguanming/jupyterlab/rgm/migration_packages/beta_fft_all_20260722_230917`.

## 5090D outline

```bash
git clone <PRIVATE_REPOSITORY_URL>
cd beta-FFT-main-code
# Extract the no-ISIC project/checkpoint tar.zst packages into the target parent directory.
# Then run migration_no_isic/restore scripts and relink datasets.
```
