# β-FFT code snapshot for private migration

This is a code-only staging copy prepared for a private Git repository. It contains source code, model definitions, training/evaluation/controller scripts, configuration templates, lightweight protocol and audit metadata, environment exports and 5090D restore helpers.

It intentionally does not contain `runs/`, checkpoints, raw medical datasets, large logs, migration tarballs, or secrets. ISIC2018 experiment assets are excluded from the no-ISIC transfer package; small ISIC source files are retained as project code.

## 5090D code restore

```bash
git clone <PRIVATE_REPOSITORY_URL>
cd beta-FFT-main-code
```

Then transfer and extract the no-ISIC package set from:

```text
beta_fft_no_isic_20260722_235239
```

Extract the project package and checkpoint package into the target parent directory, recreate the conda environment from `migration_no_isic/environment/`, and relink raw datasets according to `migration_no_isic/DATASET_PATH_MAP_NO_ISIC.tsv`.

## Relationship to tar.zst packages

- Git: source, configs, scripts, lightweight metadata, restore helpers.
- tar.zst package: non-ISIC runs, logs, audits and checkpoint/model files for PROMISE12/PB2D, ACDC, MSCMR and other non-ISIC branches.
- Raw datasets: not included; mount/relink on the 5090D server.
