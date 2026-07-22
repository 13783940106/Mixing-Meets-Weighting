import csv
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import ndimage
from torch.utils.data import Dataset

from dataloaders.promise12 import color_jitter, cutout_gray, random_rot_flip, random_rotate


def _read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _case_ids_from_split(path):
    return [row["case_id"] for row in _read_tsv(path)]


def _normalize_ct(image, clip_min=-1000.0, clip_max=1000.0):
    image = np.clip(image.astype(np.float32), clip_min, clip_max)
    return (image - clip_min) / (clip_max - clip_min)


def _resize_slice(image, out_size, is_label=False):
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    return cv2.resize(image, (out_size, out_size), interpolation=interp)


class _VolumeCache:
    def __init__(self, max_items=4):
        self.max_items = max_items
        self._items = OrderedDict()

    def get(self, key):
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key, value):
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


class MSDTask07(Dataset):
    """MSD Task07 Pancreas source-compatible 2D adapter.

    Train mode emits resized 2D slices with traceable case/slice ids.
    Val/test modes emit complete original-resolution volumes plus spacing.
    """

    def __init__(
        self,
        data_root,
        mode,
        out_size,
        train_split=None,
        val_split=None,
        test_split=None,
        label_budget=None,
        cache_dir=None,
        clip_min=-1000.0,
        clip_max=1000.0,
    ):
        self.data_root = Path(data_root)
        self.mode = mode
        self.out_size = int(out_size)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache = _VolumeCache(max_items=4)

        self.images_tr = self.data_root / "imagesTr"
        self.labels_tr = self.data_root / "labelsTr"
        self.images_ts = self.data_root / "imagesTs"
        self.dataset_json = self.data_root / "dataset.json"
        if not self.dataset_json.exists():
            raise FileNotFoundError(f"missing dataset.json: {self.dataset_json}")

        self.train_cases = _case_ids_from_split(train_split) if train_split else []
        self.val_cases = _case_ids_from_split(val_split) if val_split else []
        self.test_cases = _case_ids_from_split(test_split) if test_split else []
        self.label_budget_rows = _read_tsv(label_budget) if label_budget else []
        self.labeled_cases = [r["case_id"] for r in self.label_budget_rows if r.get("labeled") == "yes"]
        self.unlabeled_cases = [r["case_id"] for r in self.label_budget_rows if r.get("labeled") != "yes"]

        if mode == "train":
            self.case_ids = self.labeled_cases + self.unlabeled_cases
            if set(self.case_ids) != set(self.train_cases):
                raise ValueError("label budget membership does not match train split")
            self._train_cache_ready = False
            if self.cache_dir is not None:
                self._build_or_load_train_cache()
            else:
                self.slice_refs = []
                self.labeled_slice_count = 0
                for case_id in self.labeled_cases:
                    n = self._shape_for_case(case_id)[2]
                    self.slice_refs.extend((case_id, z, True) for z in range(n))
                    self.labeled_slice_count += n
                for case_id in self.unlabeled_cases:
                    n = self._shape_for_case(case_id)[2]
                    self.slice_refs.extend((case_id, z, False) for z in range(n))
        elif mode == "val":
            self.case_ids = self.val_cases
            self.slice_refs = []
            self.labeled_slice_count = 0
        elif mode == "test":
            self.case_ids = self.test_cases
            self.slice_refs = []
            self.labeled_slice_count = 0
        else:
            raise ValueError(f"unknown mode: {mode}")

    def _paths(self, case_id):
        return self.images_tr / f"{case_id}.nii.gz", self.labels_tr / f"{case_id}.nii.gz"

    def _shape_for_case(self, case_id):
        image_path, _ = self._paths(case_id)
        return nib.load(str(image_path)).shape

    def _load_volume(self, case_id):
        cached = self._cache.get(case_id)
        if cached is not None:
            return cached
        image_path, label_path = self._paths(case_id)
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"missing image/label pair for {case_id}")
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))
        image = np.asanyarray(image_nii.dataobj).astype(np.float32)
        label = np.asanyarray(label_nii.dataobj).astype(np.uint8)
        if image.shape != label.shape:
            raise ValueError(f"image/label shape mismatch for {case_id}: {image.shape} vs {label.shape}")
        spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])
        norm = _normalize_ct(image, self.clip_min, self.clip_max)
        value = (norm, label, spacing)
        self._cache.put(case_id, value)
        return value

    def _build_or_load_train_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.cache_dir / "train_index.json"
        image_path = self.cache_dir / "train_images_f16.dat"
        label_path = self.cache_dir / "train_labels_u8.dat"
        if index_path.exists() and image_path.exists() and label_path.exists():
            meta = json.loads(index_path.read_text())
            self.slice_refs = [(r["case_id"], int(r["slice_id"]), bool(r["is_labeled"])) for r in meta["slice_refs"]]
            self.labeled_slice_count = int(meta["labeled_slice_count"])
            shape = tuple(meta["shape"])
            self._cache_images = np.memmap(image_path, dtype=np.float16, mode="r", shape=shape)
            self._cache_labels = np.memmap(label_path, dtype=np.uint8, mode="r", shape=shape)
            self._train_cache_ready = True
            return

        slice_refs = []
        labeled_slice_count = 0
        case_shapes = []
        labeled_set = set(self.labeled_cases)
        for case_id in self.labeled_cases + self.unlabeled_cases:
            n = self._shape_for_case(case_id)[2]
            is_labeled = case_id in labeled_set
            case_shapes.append((case_id, n, is_labeled))
            if is_labeled:
                labeled_slice_count += n
            for z in range(n):
                slice_refs.append((case_id, z, is_labeled))

        shape = (len(slice_refs), self.out_size, self.out_size)
        images = np.memmap(image_path, dtype=np.float16, mode="w+", shape=shape)
        labels = np.memmap(label_path, dtype=np.uint8, mode="w+", shape=shape)
        offset = 0
        for case_id, n, _ in case_shapes:
            image, label, _ = self._load_volume(case_id)
            for z in range(n):
                images[offset] = _resize_slice(image[:, :, z], self.out_size, is_label=False).astype(np.float16)
                labels[offset] = _resize_slice(label[:, :, z], self.out_size, is_label=True).astype(np.uint8)
                offset += 1
        images.flush()
        labels.flush()
        index_path.write_text(json.dumps({
            "shape": shape,
            "out_size": self.out_size,
            "labeled_slice_count": labeled_slice_count,
            "slice_refs": [
                {"case_id": c, "slice_id": int(z), "is_labeled": bool(is_labeled)}
                for c, z, is_labeled in slice_refs
            ],
        }, indent=2), encoding="utf-8")
        self.slice_refs = slice_refs
        self.labeled_slice_count = labeled_slice_count
        self._cache_images = np.memmap(image_path, dtype=np.float16, mode="r", shape=shape)
        self._cache_labels = np.memmap(label_path, dtype=np.uint8, mode="r", shape=shape)
        self._train_cache_ready = True

    def __len__(self):
        return len(self.slice_refs) if self.mode == "train" else len(self.case_ids)

    def __getitem__(self, idx):
        if self.mode == "train":
            case_id, slice_id, is_labeled = self.slice_refs[idx]
            if getattr(self, "_train_cache_ready", False):
                img = np.asarray(self._cache_images[idx]).astype(np.float32)
                mask = np.asarray(self._cache_labels[idx]).astype(np.uint8)
                spacing = (1.0, 1.0, 1.0)
            else:
                image, label, spacing = self._load_volume(case_id)
                img = _resize_slice(image[:, :, slice_id], self.out_size, is_label=False).astype(np.float32)
                mask = _resize_slice(label[:, :, slice_id], self.out_size, is_label=True).astype(np.uint8)
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)

            image_strong_np, label_strong = cutout_gray(img, mask, p=0.5)
            image_strong = color_jitter(image_strong_np).type("torch.FloatTensor")
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask)
            mask_strong = torch.from_numpy(label_strong.astype(np.uint8))
            return {
                "image": img_tensor,
                "image_strong": image_strong,
                "mask": mask_tensor,
                "mask_strong": mask_strong,
                "case_id": case_id,
                "slice_id": int(slice_id),
                "spacing": torch.tensor(spacing, dtype=torch.float32),
                "is_labeled": bool(is_labeled),
            }

        case_id = self.case_ids[idx]
        image, label, spacing = self._load_volume(case_id)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(label.astype(np.uint8)),
            "case_id": case_id,
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }
