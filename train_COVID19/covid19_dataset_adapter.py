#!/usr/bin/env python3
"""Dataset adapter for COVID-19 Official20 all-slice V2 manifests."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


FORBIDDEN_UNLABELED_FIELDS = {"mask_path", "infection_present", "foreground_pixels", "derived_mask_path"}


class Covid19Official20SliceDataset(Dataset):
    """Thin 2D slice dataset for the authoritative training core.

    Negative slices are valid samples. In unlabeled mode the manifest must not
    expose mask or foreground-derived fields.
    """

    def __init__(self, manifest_tsv: str | Path, labeled: bool) -> None:
        self.manifest_tsv = Path(manifest_tsv)
        self.labeled = labeled
        with self.manifest_tsv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            self.rows = list(reader)
            self.fieldnames = list(reader.fieldnames or [])
        if not self.rows:
            raise ValueError(f"empty COVID19 manifest: {self.manifest_tsv}")
        if not labeled:
            leaked = sorted(FORBIDDEN_UNLABELED_FIELDS.intersection(self.fieldnames))
            if leaked:
                raise ValueError(f"unlabeled manifest exposes GT fields: {leaked}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["derived_path"])
        if not path.is_absolute():
            path = (Path(os.environ["DATA_ROOT"]) / Path(*path.parts[1:])
                    if os.environ.get("DATA_ROOT") and path.parts[0] == "data"
                    else Path(__file__).resolve().parents[1] / path)
        payload = np.load(path)
        sample: dict[str, Any] = {
            "image": payload["image"].astype(np.float32),
            "patient_id": row["patient_id"],
            "slice_idx": int(row["slice_idx"]),
            "sample_id": row["sample_id"],
        }
        if self.labeled:
            mask = payload["mask"].astype(np.uint8)
            sample["mask"] = mask
        return sample


class Covid19Official20TwoStreamDataset(Dataset):
    """Single dataset surface for a two-stream labeled/unlabeled sampler."""

    def __init__(self, labeled_manifest_tsv: str | Path, unlabeled_manifest_tsv: str | Path) -> None:
        self.labeled = Covid19Official20SliceDataset(labeled_manifest_tsv, labeled=True)
        self.unlabeled = Covid19Official20SliceDataset(unlabeled_manifest_tsv, labeled=False)
        self.labeled_count = len(self.labeled)
        self.unlabeled_count = len(self.unlabeled)

    def __len__(self) -> int:
        return self.labeled_count + self.unlabeled_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < self.labeled_count:
            sample = dict(self.labeled[index])
            sample["is_labeled"] = True
            return sample
        sample = dict(self.unlabeled[index - self.labeled_count])
        sample["is_labeled"] = False
        return sample

    @property
    def labeled_indices(self) -> list[int]:
        return list(range(self.labeled_count))

    @property
    def unlabeled_indices(self) -> list[int]:
        return list(range(self.labeled_count, self.labeled_count + self.unlabeled_count))


class Covid19Official20PatientVolumeDataset(Dataset):
    """Validation/test dataset that returns complete patient volumes."""

    def __init__(self, manifest_tsv: str | Path) -> None:
        self.manifest_tsv = Path(manifest_tsv)
        with self.manifest_tsv.open("r", encoding="utf-8", newline="") as handle:
            import csv

            rows = list(csv.DictReader(handle, delimiter="\t"))
        if not rows:
            raise ValueError(f"empty COVID19 volume manifest: {self.manifest_tsv}")
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["patient_id"], []).append(row)
        self.patients = [
            (patient_id, sorted(patient_rows, key=lambda item: int(item["slice_idx"])))
            for patient_id, patient_rows in sorted(grouped.items())
        ]

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient_id, rows = self.patients[index]
        images = []
        masks = []
        slice_indices = []
        spacings = []
        for row in rows:
            path = Path(row["derived_path"])
            if not path.is_absolute():
                path = (Path(os.environ["DATA_ROOT"]) / Path(*path.parts[1:])
                    if os.environ.get("DATA_ROOT") and path.parts[0] == "data"
                    else Path(__file__).resolve().parents[1] / path)
            payload = np.load(path)
            images.append(payload["image"].astype(np.float32))
            masks.append(payload["mask"].astype(np.uint8))
            slice_indices.append(int(row["slice_idx"]))
            if "spacing" in payload.files:
                spacing = tuple(float(item) for item in payload["spacing"].tolist())
            else:
                spacing = tuple(float(item) for item in row["canonical_spacing"].split(","))
            spacings.append(spacing)
        expected = list(range(len(rows)))
        if slice_indices != expected:
            raise ValueError(f"non-contiguous patient volume slices for {patient_id}")
        if len(set(spacings)) != 1:
            raise ValueError(f"inconsistent patient volume spacing for {patient_id}: {sorted(set(spacings))}")
        return {
            "image": torch.from_numpy(np.stack(images, axis=0).astype(np.float32)),
            "mask": torch.from_numpy(np.stack(masks, axis=0).astype(np.uint8)).long(),
            "patient_id": patient_id,
            "slice_idx": torch.tensor(slice_indices, dtype=torch.long),
            "spacing": torch.tensor(spacings[0], dtype=torch.float32),
        }


def covid19_two_stream_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples into the batch keys expected by beta-FFT.

    Unlabeled entries receive zero placeholder masks with ``mask_is_real=False``.
    No unlabeled GT or mask path is read or returned by the unlabeled dataset.
    """

    images = []
    masks = []
    mask_strong = []
    is_labeled = []
    sample_ids = []
    patient_ids = []
    slice_indices = []
    for sample in samples:
        image = torch.from_numpy(np.asarray(sample["image"], dtype=np.float32)).unsqueeze(0)
        if sample.get("is_labeled", False):
            mask = torch.from_numpy(np.asarray(sample["mask"], dtype=np.uint8)).long()
            real = True
        else:
            mask = torch.zeros(image.shape[-2:], dtype=torch.long)
            real = False
        images.append(image)
        masks.append(mask)
        mask_strong.append(mask.clone())
        is_labeled.append(real)
        sample_ids.append(sample["sample_id"])
        patient_ids.append(sample["patient_id"])
        slice_indices.append(int(sample["slice_idx"]))
    image_batch = torch.stack(images, dim=0)
    return {
        "image": image_batch,
        "image_strong": image_batch.clone(),
        "mask": torch.stack(masks, dim=0),
        "mask_strong": torch.stack(mask_strong, dim=0),
        "mask_is_real": torch.tensor(is_labeled, dtype=torch.bool),
        "sample_id": sample_ids,
        "patient_id": patient_ids,
        "slice_idx": torch.tensor(slice_indices, dtype=torch.long),
    }
