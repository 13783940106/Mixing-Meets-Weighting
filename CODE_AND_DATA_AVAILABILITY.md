# Code and data availability

## Code

Source code, experiment configurations, split manifests, preprocessing scripts and evaluation instructions are available at [Mixing-Meets-Weighting](https://github.com/13783940106/Mixing-Meets-Weighting). Start with the [README](README.md) and the configurations in `configs/paper/`.

## Data sources

Obtain the original datasets directly from their providers:

| Dataset | Provider | Preparation entry | Split files |
| --- | --- | --- | --- |
| ACDC | [Official database](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) | `tools/prepare_data.py --dataset acdc` | `splits/acdc/` |
| PROMISE12 | [Official download page](https://promise12.grand-challenge.org/Download/) and [Zenodo archive](https://zenodo.org/records/8014041) | `tools/prepare_data.py --dataset promise12` | `splits/promise12/` |
| COVID-19 | [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/records/3757476) | `train_COVID19/covid19_official20_v2.py` | `splits/covid19/` |

COVID-19 uses the 20 annotated CT volumes in `COVID-19-CT-Seg_20cases.zip` and the infection labels in `Infection_Mask.zip`, including Coronacases and Radiopaedia cases. `COVID19_OFFICIAL20` names this repository's derived data layout and split protocol.

Images, masks, initialization weights and trained checkpoints are obtained separately. The repository provides the preparation scripts and configuration files.

## Preparation and evaluation protocol

**ACDC.** Prepare paired ED/ES volumes, apply whole-volume min/max normalization and export HDF5 volumes and slices. Training resizes slices to 256×256; evaluation restores predictions to the saved volume grid. The split contains 70/10/20 training/validation/test patients. The 5%/10% allocations use the first 68/136 training slices. Paper distances use unit-grid coordinates; physical geometry can be exported to `spacing.json`.

**PROMISE12.** Read flat MHD/RAW pairs, resize slices to 224×224 with nearest-neighbor interpolation, normalize each image slice by min/max and apply SimpleITK CurvatureFlow with time step 0.125 and five iterations. The case registry is Case00–34/35–39/40–49 for training/validation/internal testing. The array sampler labels rows 0–201 and treats the remaining rows as unlabeled. Keep the staged training-array order; see [case registry and array allocation](splits/promise12/README.md). Geometry metadata is in `raw_geometry.tsv`; evaluation uses unit-grid distances on staged ZYX arrays. Augmentation modes are described in the [README](README.md#promise12-augmentation-modes).

**COVID-19.** Canonicalize volumes to RAS, retain all axis-2 slices, clip image intensities at slice percentiles 0.5/99.5, normalize to [0,1] and resize to 256×256 using image interpolation order 1 and mask order 0. The split contains 14/2/4 patients and 2415/490/615 slices. Split/subset seed 42 selects 120/241 labeled slices for 5%/10%; training uses seed 1337. Manifests record sample identifiers and original/canonical geometry. Distance evaluation uses the spacing stored with the derived grid, as specified in [evaluation metrics](docs/metrics.md).

## Terms and citations

Use each dataset according to its provider's terms and citation instructions:

- ACDC: Bernard et al., *IEEE Transactions on Medical Imaging* (2018), DOI [10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).
- PROMISE12: cite the challenge paper listed on the [official details page](https://promise12.grand-challenge.org/Details/) and follow the archive's `LICENSE.TXT`.
- COVID-19: cite the provider's [Zenodo record](https://doi.org/10.5281/zenodo.3757476), distributed under CC BY 4.0.

Code attribution and retained upstream licenses are listed in [third-party notices](docs/THIRD_PARTY_NOTICES.md).

## Manuscript statement

> Code and experiment configurations are available at https://github.com/13783940106/Mixing-Meets-Weighting. Data preparation instructions and links to the original ACDC, PROMISE12, and COVID-19 dataset providers are documented in the repository.
