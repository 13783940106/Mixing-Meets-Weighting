#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[2]
needed=['migration_full_beta_fft/ALL_EXPERIMENTS_INVENTORY.tsv','migration_full_beta_fft/CHECKPOINT_INVENTORY.tsv','migration_full_beta_fft/DATASET_PATH_MAP.tsv','migration_full_beta_fft/EXTERNAL_DEPENDENCY_PATHS.tsv']
missing=[p for p in needed if not (root/p).exists()]
print({'missing':missing})
raise SystemExit(1 if missing else 0)
