# Mixing Meets Weighting

PyTorch code for **Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation**. The method extends β-FFT with TRC, TCSMix and CTW and provides five paper configurations: ACDC 5%/10%, PROMISE12 20% and COVID-19 5%/10%.

| Component | Operation | Implementation |
| --- | --- | --- |
| TRC | Retrieve learned reference vectors at the decoder–head interface | `SpatialMaskTokenRouter` in the ACDC components and PROMISE/COVID U-Net |
| TCSMix | Set the exchanged rectangle scale from the teacher's mean maximum class probability over all pixels | `generate_adamix_bcp_lite_v1_mask` |
| CTW | Weight student consistency by spatial changes in confidence | `compute_v21_3_innovation_x`, ID `V21_3_X03_CONTOUR_RELIABILITY_WEIGHT` |

The ACDC10 paper configuration is `Full_main` (original X03). The corresponding Dice is **90.35%**. Source names `Mask Token`, `AdaMix` and `CRW` refer to TRC, TCSMix and CTW, respectively.

## Installation

The original environment uses Python 3.10.20, PyTorch 2.4.1 and CUDA 12.1:

```bash
git clone https://github.com/13783940106/Mixing-Meets-Weighting.git
cd Mixing-Meets-Weighting
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt
```

See the [original environment](docs/environment-original.md) and [RTX 5090/5090D installation recipe](docs/environment-5090.md).

## Data preparation

Download the datasets from the providers listed in [Code and data availability](CODE_AND_DATA_AVAILABILITY.md). Use `data/`, `weights/` and `outputs/` as the default directories, or configure `DATA_ROOT`, `WEIGHTS_ROOT` and `OUTPUT_ROOT`.

```bash
export DATA_ROOT="$PWD/data"
export WEIGHTS_ROOT="$PWD/weights"
python tools/prepare_data.py --dataset acdc --raw-root data_raw/ACDC --data-root "$DATA_ROOT"
python tools/prepare_data.py --dataset promise12 --raw-root data_raw/PROMISE12 --data-root "$DATA_ROOT"
python train_COVID19/covid19_official20_v2.py --raw-root data_raw/COVID19 --data-root "$DATA_ROOT"
```

Place PROMISE12 MHD/RAW pairs in a flat directory and COVID files under `images/` and `infection_masks/`. ACDC preprocessing searches the patient directories recursively. Use a fresh output directory for preprocessing.

| Dataset | Prepared files |
| --- | --- |
| ACDC | `data/ACDC/data/slices/*.h5`, volume HDF5 files in `data/ACDC/data/`, and split lists |
| PROMISE12 | `data/PROMISE12/npy_image/*.npy` and split lists |
| COVID-19 | `data/COVID19_OFFICIAL20/derived/slices_256/` and manifests |

Split lists and geometry metadata are in [splits/](splits/). Keep the prepared array and list order when using existing data. PROMISE12 training labels the first 202 training array rows; its case registry and array allocation are described in [split details](splits/promise12/README.md).

ACDC training requires `weights/label3/unet_best_model.pth` for 5% and `weights/label7/unet_best_model.pth` for 10%, each containing `net` and `opt`. Both students and the teacher load the model initialization; the students also load the optimizer state. Obtain these initialization files separately. The PROMISE/COVID entry performs 10,000 pretraining updates followed by 30,000 semi-supervised updates. The ACDC entry starts from the corresponding pretrained initialization and performs 30,000 semi-supervised updates.

## Training

```bash
python tools/train_paper.py --config configs/paper/acdc5.json
python tools/train_paper.py --config configs/paper/acdc10.json
python tools/train_paper.py --config configs/paper/promise20.json
python tools/train_paper.py --config configs/paper/covid5.json
python tools/train_paper.py --config configs/paper/covid10.json
```

Configure the device and directories with `--gpu`, `--data-root`, `--weights-root` and `--output-root`. Use `--print-command` to inspect the generated command. Each run uses a fresh output directory.

| Setting | Seed | Labeled allocation | SSL updates per student | Validation cadence |
| --- | --- | --- | --- | --- |
| ACDC5 | 1337 | first 68 slices / 3 patients | 30000 | every 200 updates |
| ACDC10 | 42 | first 136 slices / 7 patients | 30000 | every 200 updates |
| PROMISE20 | 1337 | first 202 training array rows | 30000 | every 200 updates |
| COVID5 | 1337 | 120 of 2415 training slices | 30000 | update 1, then every 500 |
| COVID10 | 1337 | 241 of 2415 training slices | 30000 | update 1, then every 500 |

COVID uses split/subset seed 42. The students are independent U-Nets; the teacher follows student 1 by EMA. In the shared PROMISE/COVID entry, all three networks are constructed from `model_1`; `model_2` is a retained, unused field.

SGD uses initial LR 0.01, momentum 0.9 and weight decay 0.0001. The SSL students inherit the selected pretraining optimizer state. ACDC uses a constant learning rate; PROMISE/COVID decay student 1's rate polynomially with exponent 0.9 and keep student 2's rate fixed. TRC uses four 16-dimensional reference vectors; TCSMix uses scale endpoints 0.18/0.38; CTW uses coefficient 0.05. Student-1 validation Dice selects the checkpoint, retaining the earliest update on a tie.

For the ACDC10 eight-combination experiment, `--arm ABC` controls TRC/TCSMix/CTW. For example, `101` enables TRC and CTW:

```bash
python tools/train_paper.py --config configs/paper/acdc10.json --arm 101
```

### PROMISE12 augmentation modes

The paper config explicitly selects `legacy_overwrite`: color jitter uses the original image while the strong-view label retains the cutout region as background. The alternative `cutout_then_jitter` mode applies jitter to the cutout image and keeps it paired with the cutout label. Use its separate configuration for new training:

```bash
python tools/train_paper.py --config configs/followup/promise20_cutout_consistent.json
```

The alternative writes to `outputs/promise20_cutout_consistent/`. Its experiment status is **not run**, with no reported score. Direct loader/trainer calls default to `cutout_then_jitter`; the paper command supplies its mode explicitly. See the [version notes](docs/publication_fixes_20261010.md).

## Evaluation

Evaluate a supplied student-1 checkpoint:

```bash
python tools/evaluate_paper.py --config configs/paper/acdc5.json --checkpoint outputs/acdc5/checkpoints/best.ckpt
python tools/evaluate_paper.py --config configs/paper/acdc10.json --checkpoint outputs/acdc10/checkpoints/best.ckpt
python tools/evaluate_paper.py --config configs/paper/promise20.json --checkpoint outputs/promise20/self_train/unet_best_model.pth
python tools/evaluate_paper.py --config configs/paper/covid5.json --checkpoint outputs/covid5/self_train/unet_best_model.pth
python tools/evaluate_paper.py --config configs/paper/covid10.json --checkpoint outputs/covid10/self_train/unet_best_model.pth
```

ACDC checkpoints contain a full-state dictionary; evaluation reads `model_1`. PROMISE/COVID checkpoints contain the student-1 state dictionary. The default split is `test`; use `--split val` for validation. Paths and device can be set with `--data-root`, `--output-dir` and `--gpu`. Checkpoint metadata is recorded in the configs and [paper results](docs/paper_results.tsv). See [evaluation metrics](docs/metrics.md) for reconstruction, aggregation and distance conventions.

## Paper results

| Full model | Dice (%) | IoU (%) | HD95 | ASD | Distance unit |
| --- | --- | --- | --- | --- | --- |
| ACDC5 | 88.77 | 80.51 | 2.30 | 0.74 | grid |
| ACDC10 | 90.35 | 82.93 | 1.42 | 0.43 | grid |
| PROMISE12 20% | 83.86 | 72.37 | 2.99 | 1.41 | grid |
| COVID-19 5% | 67.31 | 51.45 | 22.79 | 4.06 | mm |
| COVID-19 10% | 67.94 | 52.38 | 11.88 | 3.03 | mm |

[Code and data availability](CODE_AND_DATA_AVAILABILITY.md) · [Source provenance](docs/source_provenance.json) · [Upstream attribution and licenses](docs/THIRD_PARTY_NOTICES.md)
