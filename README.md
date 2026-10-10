# Mixing Meets Weighting

Code for **Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation**, built on β-FFT. The five paper configurations cover ACDC 5%/10%, PROMISE12 20%, and COVID-19 5%/10%. Earlier code and migration material already in this repository are retained.

| Paper name | Existing implementation name | Source |
| --- | --- | --- |
| TRC | Mask Token / `SpatialMaskTokenRouter` | `train_ACDC/v14_mask_adamix_components.py`, `train_promise12/networks/unet.py` |
| TCSMix | AdaMix / `generate_adamix_bcp_lite_v1_mask` | ACDC component file and PROMISE/COVID trainer |
| CTW | CRW / `compute_v21_3_innovation_x` | `train_ACDC/v21_3_acdc10_third_innovation_components.py`, ID `V21_3_X03_CONTOUR_RELIABILITY_WEIGHT` |

**Main Full / paper R1 is original X03 / Full_main (ACDC10% Dice 90.35%).** Its mixing control is the teacher's all-pixel mean maximum class probability, averaged over the two teacher batches. `R1_v2_B` / `formal_R1_20261007` (89.98%) changes that signal and is a separate experiment. Implementation names and the main algorithm are preserved.

## Installation

Use Python 3.10.20 and a CUDA GPU. The recorded A100 profile uses PyTorch 2.4.1+cu121:

```bash
git clone https://github.com/13783940106/beta-FFT-main-code.git
cd beta-FFT-main-code
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt
```

See the [recorded environment](docs/environment-original.md) and separate [RTX 5090/5090D recipe](docs/environment-5090.md). The latter is not an execution-validated environment.

## Data and initialization

Obtain original data from [CODE_AND_DATA_AVAILABILITY.md](CODE_AND_DATA_AVAILABILITY.md). Defaults are `data/`, `weights/`, `outputs/`; override with `DATA_ROOT`, `WEIGHTS_ROOT`, `OUTPUT_ROOT`, or launcher arguments. Raw PROMISE12 MHD/RAW pairs share a flat directory; raw COVID files go under `images/` and `infection_masks/`. ACDC preprocessing searches the provider's patient directories recursively.

```bash
export DATA_ROOT="$PWD/data"
export WEIGHTS_ROOT="$PWD/weights"
python tools/prepare_data.py --dataset acdc --raw-root data_raw/ACDC --data-root "$DATA_ROOT"
python tools/prepare_data.py --dataset promise12 --raw-root data_raw/PROMISE12 --data-root "$DATA_ROOT"
python train_COVID19/covid19_official20_v2.py --raw-root data_raw/COVID19 --data-root "$DATA_ROOT"
```

Processed layouts are `data/ACDC/data/slices/*.h5` plus volume HDF5 under `data/ACDC/data/`, `data/PROMISE12/npy_image/*.npy`, and `data/COVID19_OFFICIAL20/derived/slices_256/`. Frozen lists/geometry are in [splits/](splits/). Existing processed data can be placed at these paths directly; retain array/list order and the ACDC/PROMISE `.list` files. All preprocessing commands reject an existing output directory.

ACDC needs `weights/label3/unet_best_model.pth` and `weights/label7/unet_best_model.pth`, containing `net` and `opt`. Both students and the teacher inherit model initialization; students inherit optimizer state. These weights are not uploaded. Their actual generation commands/update counts were not saved in confirmed records; the parsed `pre_iterations=10000` field does not establish their generation budget. PROMISE/COVID perform the recorded 10,000-update pretraining stage before 30,000 SSL updates.

PROMISE retains the actual trainer's **first 202 training array rows** as labeled. Its nominal seven-case registry is supplied separately; see [split details](splits/promise12/README.md). Historical checkpoint reproduction requires the original staged array ordering. Newly prepared arrays have not been checked for byte identity with that staging.

## Five Full training commands

```bash
python tools/train_paper.py --config configs/paper/acdc5.json
python tools/train_paper.py --config configs/paper/acdc10.json
python tools/train_paper.py --config configs/paper/promise20.json
python tools/train_paper.py --config configs/paper/covid5.json
python tools/train_paper.py --config configs/paper/covid10.json
```

The launcher delegates to `train_ACDC/v21_3_acdc10_third_innovation_transfer.py` or `train_promise12/train_promise12_bcp_fft.py`. The latter is the shared archived Final3 entry; existing branches/default-off controls are retained. `--print-command` prints argv without importing the trainer. `--gpu`, `--data-root`, `--weights-root`, `--output-root` configure paths/device. Existing output directories are rejected.

| Setting | Training seed | Actual labeled allocation | SSL updates per student | Validation cadence |
| --- | --- | --- | --- | --- |
| ACDC5 | 1337 | first 68 slices / 3 patients | 30000 | every 200 |
| ACDC10 | 42 | first 136 slices / 7 patients | 30000 | every 200 |
| PROMISE20 | 1337 | first 202 training array rows | 30000 | every 200 |
| COVID5 | 1337 | 120 of 2415 training slices | 30000 | step 1, then every 500 |
| COVID10 | 1337 | 241 of 2415 training slices | 30000 | step 1, then every 500 |

COVID split/subset seed is independently 42. Selection uses student-1 validation Dice, updating only on strict improvement (earliest-step tie); test scores do not select checkpoints. SGD uses LR 0.01, momentum 0.9, weight decay 0.0001. ACDC SSL keeps constant LR and inherited optimizer state; PROMISE/COVID SSL use the existing polynomial schedule (exponent 0.9). CTW weight is 0.05.

For the ACDC10 eight-combination experiment, `--arm ABC` maps A/B/C to TRC/TCSMix/CTW. `000`, `100`, `010`, `001`, `110`, `101`, `011`, `111` cover every combination, with other settings fixed:

```bash
python tools/train_paper.py --config configs/paper/acdc10.json --arm 101
```

## Checkpoints and evaluation

ACDC saves a full-state dictionary in `checkpoints/best.ckpt`; evaluation uses `model_1`. PROMISE/COVID save the selected student-1 state dictionary in `self_train/unet_best_model.pth`. Use an existing validation-selected checkpoint or the corresponding output of a new run. Historical SHA256 and validation steps are recorded in the configs and [paper results](docs/paper_results.tsv). No weights are distributed here.

```bash
python tools/evaluate_paper.py --config configs/paper/acdc5.json --checkpoint outputs/acdc5/checkpoints/best.ckpt
python tools/evaluate_paper.py --config configs/paper/acdc10.json --checkpoint outputs/acdc10/checkpoints/best.ckpt
python tools/evaluate_paper.py --config configs/paper/promise20.json --checkpoint outputs/promise20/self_train/unet_best_model.pth
python tools/evaluate_paper.py --config configs/paper/covid5.json --checkpoint outputs/covid5/self_train/unet_best_model.pth
python tools/evaluate_paper.py --config configs/paper/covid10.json --checkpoint outputs/covid10/self_train/unet_best_model.pth
```

Default split is `test`; add `--split val` to validate a supplied checkpoint. Evaluation accepts `--data-root`, `--output-dir`, `--gpu` and arbitrary checkpoint paths. It never selects or replaces checkpoints. See [metric conventions](docs/metrics.md).

| Main Full | Dice (%) | IoU (%) | HD95 | ASD | Distance unit |
| --- | --- | --- | --- | --- | --- |
| ACDC5 | 88.77 | 80.51 | 2.30 | 0.74 | grid |
| ACDC10 / paper R1 | 90.35 | 82.93 | 1.42 | 0.43 | grid |
| PROMISE12 20% | 83.86 | 72.37 | 2.99 | 1.41 | grid |
| COVID-19 5% | 67.31 | 51.45 | 22.79 | 4.06 | mm |
| COVID-19 10% | 67.94 | 52.38 | 11.88 | 3.03 | mm |

These are existing paper/receipt values. COVID main Full HD95 remains author-confirmed 22.79/11.88 mm; other spacing-only versions do not replace it. This update performs static syntax, dependency, path and Git-diff checks only. Training, model loading, inference, evaluation and checkpoint reselection were not performed; packaged commands have not been execution-validated.

[Source provenance](docs/source_provenance.json) · [Upstream attribution/licenses](docs/THIRD_PARTY_NOTICES.md) · [Data/code availability](CODE_AND_DATA_AVAILABILITY.md)
