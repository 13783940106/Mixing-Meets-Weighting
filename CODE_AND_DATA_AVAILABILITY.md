# Code and data availability

## Repository access

The GitHub API reports [13783940106/beta-FFT-main-code](https://github.com/13783940106/beta-FFT-main-code) as **public**, default branch `main`, checked October 10, 2026. This update does not change visibility. Code, configurations, split manifests and preprocessing/evaluation instructions are provided. Raw/processed images, masks, initialization weights and trained checkpoints are not redistributed here.

## Original data and project preprocessing

| Dataset | Original provider / acquisition link | Project preprocessing and splits |
| --- | --- | --- |
| ACDC | [Official ACDC database](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), linking to the provider download portal | `tools/prepare_data.py --dataset acdc`; `splits/acdc/` |
| PROMISE12 | [Official challenge download page](https://promise12.grand-challenge.org/Download/), directing users to [Zenodo DOI 10.5281/zenodo.8014041](https://zenodo.org/records/8014041) | `tools/prepare_data.py --dataset promise12`, using `train_promise12/dataloaders/promise12.py`; `splits/promise12/` |
| COVID-19 | **COVID-19 CT Lung and Infection Segmentation Dataset**, [Zenodo DOI 10.5281/zenodo.3757476](https://zenodo.org/records/3757476), Jun Ma and collaborators; 20 annotated CT volumes | `train_COVID19/covid19_official20_v2.py`; `configs/paper/covid19_raw_patients.json`; `splits/covid19/` |

COVID uses the 20-case record's CT volumes (`COVID-19-CT-Seg_20cases.zip`) and binary infection annotations (`Infection_Mask.zip`), including Coronacases and Radiopaedia cases. This is distinct from the 100-image medicalsegmentation.com dataset and other COVID collections. `COVID19_OFFICIAL20` is this project's derived data/protocol, not the provider's original data or benchmark split.

ACDC uses paired ED/ES NIfTI volumes, whole-volume min/max normalization, HDF5 volume/slice export and training augmentation resize to 256×256. Evaluation restores predictions to the saved grid. Existing lists retain 70/10/20 training/validation/test patients; the first 68/136 training slices define the nominal 5%/10% allocations. Physical geometry can be exported to `spacing.json`; paper distances use the saved grid without spacing. Normalization follows the existing ACDC processing source inherited from SSL4MIS; this release adds paths and volume export for the evaluator.

PROMISE12 reads flat MHD/RAW pairs, resizes each slice to 224×224 with nearest-neighbor interpolation, normalizes each image slice by min/max and applies SimpleITK CurvatureFlow (time step 0.125, five iterations). Cases 00–34/35–39/40–49 form the existing training/validation/internal-test registry; challenge external test data are separate. The actual Final3 sampler labels array rows 0–201; remaining rows are unlabeled. The nominal Case00–06 registry and actual array indices are recorded separately in `splits/promise12/`, without inferring or changing their historical row mapping. Raw XYZ geometry is in `raw_geometry.tsv`; evaluation uses unit-grid distances on staged ZYX arrays. New raw preprocessing has not been verified as byte-identical to historical staging.

COVID preprocessing canonicalizes to RAS, uses all axis-2 slices including negatives, clips image intensities at slice percentiles 0.5/99.5, normalizes to [0,1] and resizes to 256×256 (image interpolation order 1, mask order 0). This project uses 14/2/4 train/validation/test patients and 2415/490/615 slices. Split seed 42 and the existing per-patient allocation select 120/241 labeled slices for nominal 5%/10%. Training seed is 1337. Manifests retain dataset case/sample identifiers and original/canonical geometry; array files are excluded. The historical stored-spacing evaluator convention is preserved along with author-confirmed Full HD95 22.79/11.88 mm; no metrics were recomputed.

## Terms and attribution

Obtain data from providers and comply with their terms and required citations. ACDC requests Bernard et al., *IEEE TMI* (2018), DOI 10.1109/TMI.2018.2837502. PROMISE12's [official details](https://promise12.grand-challenge.org/Details/) request the challenge-paper citation; its Zenodo archive includes `LICENSE.TXT` and identifies its license as other attribution terms (`other-at`). Follow that file rather than assuming a standard code license. COVID's Zenodo metadata lists CC BY 4.0 for this record. Code terms are separate: see [retained upstream notices and licenses](docs/THIRD_PARTY_NOTICES.md). This update assigns no blanket license.

Provider links/metadata were verified from official pages or provider Zenodo APIs on October 10, 2026. PROMISE12's official download page points to record 8014041; local historical archives use earlier filenames, so byte identity with the current provider archive is not claimed.

## Manuscript-ready statement

> The source code, experiment configurations, preprocessing scripts and split manifests are publicly available at https://github.com/13783940106/beta-FFT-main-code. Original ACDC and PROMISE12 data are available from their official challenge providers, and the COVID-19 CT Lung and Infection Segmentation Dataset is available at https://doi.org/10.5281/zenodo.3757476. The repository documents project-specific preprocessing and evaluation conventions. Raw and preprocessed medical images and model weights are not redistributed in the repository and remain subject to the data providers' terms.
