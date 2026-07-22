"""Frozen LGE-only, patient-manifest-driven 2D data input for V15.1."""

from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from train_MSCMRSEG.v15_1_mscmr_protocol import raw_label_to_train_label


def read_patient_manifest(path: str | Path, role: str | None = None) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if role is not None:
        rows = [row for row in rows if row.get("role") == role]
    if not rows:
        raise ValueError(f"empty patient manifest: {path}")
    required = {"patient_id", "image_path", "label_path"}
    if not required.issubset(rows[0]):
        raise ValueError(f"manifest lacks {sorted(required)}: {path}")
    return rows


def _normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)
    if not finite.all():
        raise ValueError("non-finite image volume")
    foreground = volume != 0
    values = volume[foreground] if foreground.any() else volume.reshape(-1)
    std = float(values.std())
    if std < 1e-8:
        std = 1.0
    return ((volume - float(values.mean())) / std).astype(np.float32)


def resize_slice(array: np.ndarray, size: Sequence[int], label: bool) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array)).float()[None, None]
    mode = "nearest" if label else "bilinear"
    kwargs = {} if label else {"align_corners": False}
    result = F.interpolate(tensor, size=tuple(size), mode=mode, **kwargs)[0, 0]
    return result.numpy().astype(np.int64 if label else np.float32)


class _VolumeCache:
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.items: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, image_path: Path, label_path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
        key = str(image_path)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        image = _normalize(np.asanyarray(nib.load(str(image_path)).dataobj))
        label = None if label_path is None else raw_label_to_train_label(
            np.asanyarray(nib.load(str(label_path)).dataobj))
        if image.ndim != 3 or (label is not None and image.shape != label.shape):
            raise ValueError(f"invalid pair shape: {image_path} {image.shape} / {None if label is None else label.shape}")
        self.items[key] = (image, label)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return image, label


class MSCMRSliceDataset(Dataset):
    def __init__(self, data_root: str | Path, manifest: str | Path, image_size=(256, 256),
                 augment: bool = False, include_label: bool = True, role: str | None = None):
        self.data_root = Path(data_root)
        self.rows = read_patient_manifest(manifest, role=role)
        self.image_size = tuple(image_size)
        self.augment = augment
        self.include_label = include_label
        self.cache = _VolumeCache()
        self.samples = []
        for row_index, row in enumerate(self.rows):
            image_path = self.data_root / row["image_path"]
            image_nii = nib.load(str(image_path))
            label_shape = image_nii.shape
            if include_label:
                label_shape = nib.load(str(self.data_root / row["label_path"])).shape
            if image_nii.shape != label_shape or len(image_nii.shape) != 3:
                raise ValueError(f"manifest pair mismatch: {row['patient_id']}")
            for slice_index in range(image_nii.shape[2]):
                self.samples.append((row_index, slice_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row_index, slice_index = self.samples[index]
        row = self.rows[row_index]
        label_path = self.data_root / row["label_path"] if self.include_label else None
        image, label = self.cache.get(self.data_root / row["image_path"], label_path)
        image_slice = image[:, :, slice_index]
        label_slice = label[:, :, slice_index] if label is not None else np.zeros(image_slice.shape, dtype=np.uint8)
        if self.augment:
            if random.random() < 0.5:
                k = random.randrange(4)
                image_slice = np.rot90(image_slice, k).copy()
                label_slice = np.rot90(label_slice, k).copy()
            if random.random() < 0.5:
                axis = random.randrange(2)
                image_slice = np.flip(image_slice, axis=axis).copy()
                label_slice = np.flip(label_slice, axis=axis).copy()
        image_slice = resize_slice(image_slice, self.image_size, label=False)
        label_slice = resize_slice(label_slice, self.image_size, label=True)
        if label_slice.min() < 0 or label_slice.max() > 3:
            raise ValueError(f"label range violation: {row['patient_id']} slice {slice_index}")
        strong_image, strong_label = image_slice.copy(), label_slice.copy()
        if self.augment:
            gain = random.uniform(0.9, 1.1); bias = random.uniform(-0.1, 0.1)
            strong_image = strong_image * gain + bias
            if random.random() < 0.5:
                h, w = strong_image.shape
                cut_h, cut_w = random.randint(max(1, h // 10), max(2, h // 3)), random.randint(max(1, w // 10), max(2, w // 3))
                top, left = random.randint(0, h - cut_h), random.randint(0, w - cut_w)
                strong_image[top:top + cut_h, left:left + cut_w] = 0
                strong_label[top:top + cut_h, left:left + cut_w] = 0
        sample = {
            "image": torch.from_numpy(image_slice)[None],
            "label": torch.from_numpy(label_slice).long(),
            "image_strong": torch.from_numpy(strong_image.astype(np.float32))[None],
            "label_strong": torch.from_numpy(strong_label).long(),
            "patient_id": row["patient_id"],
            "slice_index": slice_index,
            "is_labeled": self.include_label,
        }
        return sample


class InfiniteBatchSampler(Sampler[List[int]]):
    """Deterministic, resumable random batches over one dataset."""
    def __init__(self, length: int, batch_size: int, seed: int, start_iteration: int = 0):
        if length < 1 or batch_size < 1:
            raise ValueError("empty dataset or batch")
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.iteration = start_iteration

    def __iter__(self):
        while True:
            generator = torch.Generator().manual_seed(self.seed + self.iteration)
            yield torch.randint(self.length, (self.batch_size,), generator=generator).tolist()
            self.iteration += 1

    def __len__(self):
        return 2**31


class PairedInfiniteBatchSampler(Sampler[List[int]]):
    """A batch with an exact labeled prefix and unlabeled suffix."""
    def __init__(self, labeled_length: int, unlabeled_length: int, labeled_bs: int,
                 unlabeled_bs: int, seed: int, start_iteration: int = 0):
        if min(labeled_length, unlabeled_length, labeled_bs, unlabeled_bs) < 1:
            raise ValueError("paired sampler requires non-empty streams")
        self.labeled_length, self.unlabeled_length = labeled_length, unlabeled_length
        self.labeled_bs, self.unlabeled_bs = labeled_bs, unlabeled_bs
        self.seed, self.iteration = seed, start_iteration

    def __iter__(self):
        while True:
            generator = torch.Generator().manual_seed(self.seed + self.iteration)
            labeled = torch.randint(self.labeled_length, (self.labeled_bs,), generator=generator).tolist()
            unlabeled = torch.randint(self.unlabeled_length, (self.unlabeled_bs,), generator=generator).tolist()
            yield labeled + [self.labeled_length + i for i in unlabeled]
            self.iteration += 1

    def __len__(self):
        return 2**31
