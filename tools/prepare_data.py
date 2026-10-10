#!/usr/bin/env python3
"""Prepare ACDC HDF5 or PROMISE12 arrays using the existing preprocessing."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def acdc(raw_root, output):
    import h5py
    import numpy as np
    import SimpleITK as sitk

    splits = ROOT / "splits/acdc"
    train = set((splits / "train.list").read_text().splitlines())
    cases = train | set((splits / "val.list").read_text().splitlines()) | set((splits / "test.list").read_text().splitlines())
    (output / "data/slices").mkdir(parents=True)
    image_paths = {p.name.removesuffix(".nii.gz"): p for p in raw_root.rglob("*.nii.gz") if not p.name.endswith("_gt.nii.gz")}
    geometry = []
    for case in sorted(cases):
        path = image_paths[case]
        image_itk = sitk.ReadImage(str(path))
        image = sitk.GetArrayFromImage(image_itk)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path.with_name(case + "_gt.nii.gz"))))
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {case}")
        # Same whole-volume min/max normalization as the existing upstream script.
        image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)
        with h5py.File(output / "data" / (case + ".h5"), "w") as handle:
            handle.create_dataset("image", data=image, compression="gzip")
            handle.create_dataset("label", data=mask, compression="gzip")
        if case in train:
            for z in range(image.shape[0]):
                with h5py.File(output / "data/slices" / f"{case}_slice_{z}.h5", "w") as handle:
                    handle.create_dataset("image", data=image[z], compression="gzip")
                    handle.create_dataset("label", data=mask[z], compression="gzip")
        geometry.append({"case": case, "array_axes": "ZYX", "spacing_xyz": image_itk.GetSpacing(),
                         "origin_xyz": image_itk.GetOrigin(), "direction": image_itk.GetDirection()})
    for filename in ("train.list", "train_slices.list", "val.list", "test.list"):
        shutil.copyfile(splits / filename, output / filename)
    (output / "spacing.json").write_text(json.dumps(geometry, indent=2) + "\n")


def promise12(raw_root, output):
    # The existing loader defines resize, slice min/max and CurvatureFlow.
    sys.path.insert(0, str(ROOT / "train_promise12"))
    from dataloaders.promise12 import data_to_array

    output.mkdir(parents=True)
    array_root = output / "npy_image"
    array_root.mkdir()
    data_to_array(str(raw_root), str(array_root), 224, 224)
    for filename in ("train.list", "val.list", "test.list"):
        shutil.copyfile(ROOT / "splits/promise12" / filename, output / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["acdc", "promise12"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")))
    args = parser.parse_args()
    subdir = "ACDC" if args.dataset == "acdc" else "PROMISE12"
    output = args.data_root.resolve() / subdir
    if output.exists():
        raise FileExistsError(f"Use a new data root; preprocessing output already exists: {output}")
    globals()[args.dataset](args.raw_root.resolve(), output)


if __name__ == "__main__":
    main()
