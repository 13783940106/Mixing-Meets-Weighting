"""ISIC 2018 data access with process-wide ground-truth/Test guards."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

COUNTERS = {"unlabeled_gt_open": 0, "unlabeled_gt_decode": 0, "unlabeled_gt_return": 0,
            "test_image_open": 0, "test_mask_open": 0, "test_pixel_access": 0}
def reset_counters():
    for key in COUNTERS: COUNTERS[key] = 0
def counter_snapshot(): return dict(COUNTERS)

class ISICDataset(Dataset):
    def __init__(self, manifest, ids, mode, augment=False):
        self.rows = {r["sample_id"]: r for r in csv.DictReader(Path(manifest).open())}
        self.ids = [x for x in Path(ids).read_text().splitlines() if x]
        self.mode, self.augment = mode, augment
        if mode == "test":
            # The contract permits registration only. Never create a test Dataset in a training process.
            raise RuntimeError("ISIC2018_TEST_DATASET_LOCKED")
    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        row = self.rows[self.ids[index]]
        if row["is_test"] == "True":
            raise RuntimeError("ISIC2018_TEST_PIXEL_ACCESS_LOCKED")
        image = torch.from_numpy(np.array(Image.open(row["image_path"]).convert("RGB"))).permute(2,0,1).float().div_(255)
        original_size = tuple(image.shape[-2:])
        image = F.interpolate(image[None], size=(256,256), mode="bilinear", align_corners=False)[0]
        out = {"image": image, "sample_id": row["sample_id"], "original_size": original_size}
        if self.mode == "unlabeled":
            # Deliberately no mask-path lookup/open/decode/return on this branch.
            return out
        target = torch.from_numpy(np.array(Image.open(row["mask_path"])))
        target = F.interpolate(target[None,None].float(), size=(256,256), mode="nearest")[0,0].gt(0).long()
        out["target"] = target
        return out
