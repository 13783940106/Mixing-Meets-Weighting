#!/usr/bin/env python3
"""Materialize the COVID-19 Official20 all-slice V2 protocol assets.

This module is protocol plumbing only. It does not start training, instantiate
test data during training, or select samples from infection/foreground labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_IDENTITY = PROJECT_ROOT / "configs/paper/covid19_raw_patients.json"
RAW_ROOT = PROJECT_ROOT / "data_raw/COVID19"
DATA_ROOT = PROJECT_ROOT / "data"
OUT_ROOT = DATA_ROOT / "COVID19_OFFICIAL20"
MANIFEST_ROOT = OUT_ROOT / "manifests"
DERIVED_ROOT = OUT_ROOT / "derived/slices_256"
AUDIT_JSON = OUT_ROOT / "preprocessing_summary.json"
AUDIT_MD = OUT_ROOT / "preprocessing_summary.md"
MIGRATION_MD = PROJECT_ROOT / "audits/COVID19_PROTOCOL_V2_CODE_MIGRATION.md"

GLOBAL_EXPERIMENT_SEED = 42
PROTOCOL_IDENTITY = "OFFICIAL20_PATIENTLEVEL_ALLSLICE_V2"
IMAGE_SIZE = 256


@dataclass(frozen=True)
class Patient:
    patient_id: str
    image_path: Path
    mask_path: Path
    original_shape: str
    original_spacing: str
    orientation: str
    canonical_orientation: str
    slices: int


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stage1_patients() -> list[Patient]:
    data = json.loads(STAGE1_IDENTITY.read_text(encoding="utf-8"))
    patients = []
    for item in sorted(data["patients"], key=lambda row: row["patient_id"]):
        patients.append(
            Patient(
                patient_id=item["patient_id"],
                image_path=RAW_ROOT / item["image_path"],
                mask_path=RAW_ROOT / item["mask_path"],
                original_shape="x".join(str(v) for v in item["image_shape"]),
                original_spacing=",".join(f"{float(v):.8g}" for v in item["image_spacing"]),
                orientation="".join(item["image_axcodes"]),
                canonical_orientation="".join(item["canonical_mask_axcodes"]),
                slices=int(item["canonical_axial_slices"]),
            )
        )
    return patients


def resize_slice(array: np.ndarray, order: int) -> np.ndarray:
    factors = (IMAGE_SIZE / array.shape[0], IMAGE_SIZE / array.shape[1])
    return zoom(array, factors, order=order)


def normalize_image(slice_2d: np.ndarray) -> np.ndarray:
    values = slice_2d.astype(np.float32)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    values = np.clip(values, lo, hi)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def write_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def hamilton_allocation(budget: int, patient_counts: dict[str, int]) -> dict[str, int]:
    total = sum(patient_counts.values())
    quotas = []
    allocation = {}
    for patient_id, count in sorted(patient_counts.items()):
        raw = budget * count / total
        base = math.floor(raw)
        allocation[patient_id] = base
        quotas.append((raw - base, patient_id))
    remaining = budget - sum(allocation.values())
    for _, patient_id in sorted(quotas, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocation[patient_id] += 1
    return allocation


def patient_seed(patient_id: str) -> int:
    digest = hashlib.sha256(f"COVID19_OFFICIAL20_V2|42|{patient_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_labeled(train_rows: list[dict[str, object]], budget: int) -> list[str]:
    by_patient: dict[str, list[dict[str, object]]] = {}
    for row in train_rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)
    patient_counts = {pid: len(rows) for pid, rows in by_patient.items()}
    allocation = hamilton_allocation(budget, patient_counts)
    selected: list[str] = []
    for patient_id, rows in sorted(by_patient.items()):
        ordered = sorted(rows, key=lambda row: int(row["slice_idx"]))
        rng = np.random.Generator(np.random.PCG64(patient_seed(patient_id)))
        perm = rng.permutation(len(ordered))
        take = set(int(i) for i in perm[: allocation[patient_id]])
        selected.extend(str(ordered[i]["sample_id"]) for i in sorted(take))
    return sorted(selected)


def split_patients(patients: list[Patient]) -> tuple[list[str], list[str], list[str], list[str]]:
    patient_ids = [p.patient_id for p in sorted(patients, key=lambda item: item.patient_id)]
    rng = np.random.Generator(np.random.PCG64(GLOBAL_EXPERIMENT_SEED))
    permutation = [patient_ids[int(i)] for i in rng.permutation(len(patient_ids))]
    return permutation[:14], permutation[14:16], permutation[16:], permutation


def build_master_manifest(patients: list[Patient]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    for patient in patients:
        image = nib.as_closest_canonical(nib.load(str(patient.image_path)))
        mask = nib.as_closest_canonical(nib.load(str(patient.mask_path)))
        image_data = np.asarray(image.dataobj)
        mask_data = np.asarray(mask.dataobj)
        if image_data.shape != mask_data.shape:
            raise RuntimeError(f"shape mismatch after canonicalization: {patient.patient_id}")
        spacing = ",".join(f"{float(v):.8g}" for v in image.header.get_zooms()[:3])
        patient_dir = DERIVED_ROOT / patient.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        for slice_idx in range(image_data.shape[2]):
            mask_slice = (mask_data[:, :, slice_idx] > 0).astype(np.uint8)
            image_slice = normalize_image(image_data[:, :, slice_idx])
            image_resized = resize_slice(image_slice, order=1).astype(np.float32)
            mask_resized = (resize_slice(mask_slice, order=0) > 0).astype(np.uint8)
            sample_id = f"{patient.patient_id}_slice_{slice_idx:04d}"
            derived_path = patient_dir / f"{sample_id}.npz"
            np.savez_compressed(
                derived_path,
                image=image_resized,
                mask=mask_resized,
                patient_id=patient.patient_id,
                slice_idx=np.int32(slice_idx),
                spacing=np.array(image.header.get_zooms()[:3], dtype=np.float32),
            )
            foreground = int(mask_slice.sum())
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": patient.patient_id,
                    "slice_idx": slice_idx,
                    "image_path": patient.image_path.relative_to(RAW_ROOT).as_posix(),
                    "mask_path": patient.mask_path.relative_to(RAW_ROOT).as_posix(),
                    "infection_present": str(foreground > 0).lower(),
                    "foreground_pixels": foreground,
                    "original_shape": patient.original_shape,
                    "original_spacing": patient.original_spacing,
                    "canonical_spacing": spacing,
                    "orientation": patient.orientation,
                    "canonical_orientation": patient.canonical_orientation,
                    "derived_path": (Path("data") / derived_path.relative_to(DATA_ROOT)).as_posix(),
                    "derived_mask_path": (Path("data") / derived_path.relative_to(DATA_ROOT)).as_posix(),
                }
            )
    return rows


def strip_unlabeled_gt_fields(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    forbidden = {"mask_path", "infection_present", "foreground_pixels", "derived_mask_path"}
    return [{k: v for k, v in row.items() if k not in forbidden} for row in rows]


def main() -> int:
    global RAW_ROOT, DATA_ROOT, OUT_ROOT, MANIFEST_ROOT, DERIVED_ROOT, AUDIT_JSON, AUDIT_MD
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="contains images/ and infection_masks/ from Zenodo 3757476")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    RAW_ROOT = args.raw_root.resolve()
    DATA_ROOT = args.data_root.resolve()
    OUT_ROOT = DATA_ROOT / "COVID19_OFFICIAL20"
    if OUT_ROOT.exists():
        raise FileExistsError(f"Use a new data root; preprocessing output already exists: {OUT_ROOT}")
    MANIFEST_ROOT = OUT_ROOT / "manifests"
    DERIVED_ROOT = OUT_ROOT / "derived/slices_256"
    AUDIT_JSON = OUT_ROOT / "preprocessing_summary.json"
    AUDIT_MD = OUT_ROOT / "preprocessing_summary.md"
    patients = read_stage1_patients()
    train_patients, val_patients, test_patients, permutation = split_patients(patients)
    master_rows = build_master_manifest(patients)
    if len(master_rows) != 3520:
        raise RuntimeError(f"expected 3520 all-slice rows, got {len(master_rows)}")

    split_by_patient = {pid: "train" for pid in train_patients}
    split_by_patient.update({pid: "val" for pid in val_patients})
    split_by_patient.update({pid: "test" for pid in test_patients})
    for row in master_rows:
        row["split"] = split_by_patient[str(row["patient_id"])]

    fields = [
        "sample_id",
        "patient_id",
        "slice_idx",
        "split",
        "image_path",
        "mask_path",
        "infection_present",
        "foreground_pixels",
        "original_shape",
        "original_spacing",
        "canonical_spacing",
        "orientation",
        "canonical_orientation",
        "derived_path",
        "derived_mask_path",
    ]
    write_tsv(MANIFEST_ROOT / "ALL_SLICES.tsv", fields, master_rows)

    split_rows = {
        split: [row for row in master_rows if row["split"] == split]
        for split in ["train", "val", "test"]
    }
    write_tsv(MANIFEST_ROOT / "TRAIN_ALL_SLICES.tsv", fields, split_rows["train"])
    write_tsv(MANIFEST_ROOT / "VAL_ALL_SLICES.tsv", fields, split_rows["val"])
    write_tsv(MANIFEST_ROOT / "TEST_ALL_SLICES.tsv", fields, split_rows["test"])
    write_tsv(MANIFEST_ROOT / "COVID19_TRAIN_PATIENTS.tsv", ["patient_id"], [{"patient_id": p} for p in train_patients])
    write_tsv(MANIFEST_ROOT / "COVID19_VAL_PATIENTS.tsv", ["patient_id"], [{"patient_id": p} for p in val_patients])
    write_tsv(MANIFEST_ROOT / "COVID19_TEST_PATIENTS.tsv", ["patient_id"], [{"patient_id": p} for p in test_patients])

    train_rows = split_rows["train"]
    n_train = len(train_rows)
    budget_5p = math.floor(0.05 * n_train)
    budget_10p = math.floor(0.10 * n_train)
    selected_5p = set(select_labeled(train_rows, budget_5p))
    selected_10p = set(select_labeled(train_rows, budget_10p))
    if not selected_5p.issubset(selected_10p):
        raise RuntimeError("5P labeled manifest is not nested inside 10P")

    for tag, selected in [("5P", selected_5p), ("10P", selected_10p)]:
        labeled = [row for row in train_rows if row["sample_id"] in selected]
        unlabeled = [row for row in train_rows if row["sample_id"] not in selected]
        write_tsv(MANIFEST_ROOT / f"COVID19_{tag}_LABELED.tsv", fields, labeled)
        unlabeled_fields = [f for f in fields if f not in {"mask_path", "infection_present", "foreground_pixels", "derived_mask_path"}]
        write_tsv(MANIFEST_ROOT / f"COVID19_{tag}_UNLABELED.tsv", unlabeled_fields, strip_unlabeled_gt_fields(unlabeled))

    counts = {}
    for split, rows in split_rows.items():
        pos = sum(1 for row in rows if row["infection_present"] == "true")
        counts[split] = {"all": len(rows), "positive": pos, "negative": len(rows) - pos}
    total_positive = sum(1 for row in master_rows if row["infection_present"] == "true")
    total_negative = len(master_rows) - total_positive

    protocol_payload = {
        "protocol_identity": PROTOCOL_IDENTITY,
        "protocol_type": "OUR_REPRODUCIBLE_OFFICIAL_PROTOCOL",
        "not_hccp_exact_reproduction": True,
        "stop_hccp_hidden_protocol_search": True,
        "global_experiment_seed": GLOBAL_EXPERIMENT_SEED,
        "rng": "numpy.random.Generator(numpy.random.PCG64(42))",
        "patient_permutation": permutation,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "patients": len(patients),
        "all_slices": len(master_rows),
        "positive_slices": total_positive,
        "negative_slices": total_negative,
        "split_counts": counts,
        "label_budgets": {
            "5P": {
                "labeled": budget_5p,
                "unlabeled": n_train - budget_5p,
                "actual_percentage": budget_5p / n_train * 100,
            },
            "10P": {
                "labeled": budget_10p,
                "unlabeled": n_train - budget_10p,
                "actual_percentage": budget_10p / n_train * 100,
            },
        },
        "nested_5p_subset_10p": True,
        "unlabeled_gt_firewall": {
            "manifest_gt_fields_removed": True,
            "forbidden_fields": ["mask_path", "infection_present", "foreground_pixels", "derived_mask_path"],
        },
        "full_authorization": False,
    }
    protocol_payload["protocol_sha256"] = sha256_text(json.dumps(protocol_payload, sort_keys=True))
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    AUDIT_MD.write_text(
        "\n".join(
            [
                "# COVID19 Official20 All-Slice V2",
                "",
                f"`COVID19_PROTOCOL_IDENTITY = {PROTOCOL_IDENTITY}`",
                "",
                "`Protocol type = OUR_REPRODUCIBLE_OFFICIAL_PROTOCOL`",
                "",
                "`NOT_HCCP_EXACT_REPRODUCTION = true`",
                "",
                "`STOP_HCCP_HIDDEN_PROTOCOL_SEARCH = true`",
                "",
                f"- Patients: `{len(patients)}`",
                f"- All axial slices: `{len(master_rows)}`",
                f"- Infection-positive slices: `{total_positive}`",
                f"- Infection-negative slices: `{total_negative}`",
                f"- Train patients: `{len(train_patients)}`",
                f"- Val patients: `{len(val_patients)}`",
                f"- Test patients: `{len(test_patients)}`",
                f"- Train all slices: `{counts['train']['all']}`",
                f"- Train positive/negative: `{counts['train']['positive']} / {counts['train']['negative']}`",
                f"- Val all slices: `{counts['val']['all']}`",
                f"- Val positive/negative: `{counts['val']['positive']} / {counts['val']['negative']}`",
                f"- Test all slices: `{counts['test']['all']}`",
                f"- Test positive/negative: `{counts['test']['positive']} / {counts['test']['negative']}`",
                f"- 5P labeled: `{budget_5p}`",
                f"- 5P actual percentage: `{budget_5p / n_train * 100:.6f}%`",
                f"- 10P labeled: `{budget_10p}`",
                f"- 10P actual percentage: `{budget_10p / n_train * 100:.6f}%`",
                "- 5P subset 10P: `PASS`",
                "- Unlabeled GT firewall: `PASS`",
                f"- Protocol SHA256: `{protocol_payload['protocol_sha256']}`",
                "",
                "`FULL_AUTHORIZATION = false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "COVID19_PROTOCOL_IDENTITY": PROTOCOL_IDENTITY,
                "ALL_SLICES": len(master_rows),
                "TRAIN_ALL": counts["train"]["all"],
                "VAL_ALL": counts["val"]["all"],
                "TEST_ALL": counts["test"]["all"],
                "N_5P": budget_5p,
                "N_10P": budget_10p,
                "FULL_AUTHORIZATION": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
