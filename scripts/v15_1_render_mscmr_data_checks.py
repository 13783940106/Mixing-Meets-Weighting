#!/usr/bin/env python3
"""Render fixed MS-CMR image/mask overlay assets without modifying source data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


CASE_RE = re.compile(r"(?:subject|patient)(\d+)", re.IGNORECASE)


def mapped_label(label: np.ndarray) -> tuple[np.ndarray, str]:
    values = set(np.unique(label).tolist())
    if values.issubset({0, 1, 2, 3}):
        return label.astype(np.uint8), "0=BG,1=RV,2=MYO,3=LV"
    if values.issubset({0, 200, 500, 600}):
        output = np.zeros(label.shape, dtype=np.uint8)
        output[label == 600] = 1
        output[label == 200] = 2
        output[label == 500] = 3
        return output, "0=BG,600=RV,200=MYO,500=LV"
    raise ValueError(f"Unknown label values: {sorted(values)}")


def modality(path: Path) -> str:
    text = path.as_posix()
    if "/C0/" in text:
        return "bSSFP"
    if "/T2/" in text:
        return "T2"
    return "LGE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    labels = sorted(root.rglob("labels/*.nii.gz"))
    records = []
    for label_path in labels:
        image_path = label_path.parent.parent / "images" / label_path.name.replace("_manual.nii.gz", ".nii.gz")
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)
        image = np.asanyarray(image_nii.dataobj)
        raw_label = np.asanyarray(label_nii.dataobj)
        label, mapping = mapped_label(raw_label)
        counts = np.count_nonzero(label, axis=(0, 1))
        match = CASE_RE.search(label_path.name)
        records.append({
            "label_path": label_path,
            "image_path": image_path,
            "image": image,
            "label": label,
            "raw_label": raw_label,
            "mapping": mapping,
            "patient_id": f"case{int(match.group(1)):02d}",
            "split": label_path.relative_to(root).parts[0],
            "modality": modality(label_path),
            "foreground_slice": int(np.argmax(counts)),
            "empty_slices": np.where(counts == 0)[0].tolist(),
            "near_empty_slice": int(np.argmin(counts)),
            "affine_match": bool(np.allclose(image_nii.affine, label_nii.affine, atol=1e-5, rtol=0)),
        })

    # Deterministic coverage: train/val/test LGE plus test T2 and bSSFP.
    foreground_targets = [
        ("train", "LGE", "case02"), ("train", "LGE", "case13"),
        ("train", "LGE", "case25"), ("val", "LGE", "case01"),
        ("val", "LGE", "case36"), ("test", "LGE", "case03"),
        ("test", "LGE", "case43"), ("test", "T2", "case10"),
        ("test", "T2", "case33"), ("test", "bSSFP", "case05"),
    ]
    foreground = []
    for target in foreground_targets:
        matched = [record for record in records if (record["split"], record["modality"], record["patient_id"]) == target]
        if len(matched) != 1:
            raise RuntimeError(f"Expected one record for {target}, got {len(matched)}")
        foreground.append((matched[0], matched[0]["foreground_slice"], "foreground"))

    empty = []
    used_patients = set()
    for record in records:
        if len(empty) >= 5:
            break
        if record["patient_id"] in used_patients:
            continue
        slice_index = record["empty_slices"][0] if record["empty_slices"] else record["near_empty_slice"]
        empty.append((record, slice_index, "empty" if record["empty_slices"] else "near_empty"))
        used_patients.add(record["patient_id"])

    rows = []
    colors = ListedColormap(["none", "#ff3355", "#33dd66", "#3388ff"])
    for index, (record, slice_index, category) in enumerate(foreground + empty, start=1):
        image_slice = np.rot90(record["image"][:, :, slice_index])
        label_slice = np.rot90(record["label"][:, :, slice_index])
        raw_values = sorted(np.unique(record["raw_label"][:, :, slice_index]).tolist())
        mapped_values = sorted(np.unique(label_slice).tolist())
        finite = image_slice[np.isfinite(image_slice)]
        low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)

        fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
        axes[0].imshow(image_slice, cmap="gray", vmin=low, vmax=high)
        axes[0].set_title("Image")
        axes[1].imshow(image_slice, cmap="gray", vmin=low, vmax=high)
        overlay = np.ma.masked_where(label_slice == 0, label_slice)
        axes[1].imshow(overlay, cmap=colors, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
        axes[1].set_title("GT overlay")
        for class_index, class_name, axis in zip((1, 2, 3), ("RV", "MYO", "LV"), axes[2:]):
            axis.imshow(label_slice == class_index, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(f"{class_name} (class {class_index})")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(
            f"{record['patient_id']} | {record['split']} | {record['modality']} | slice={slice_index} | "
            f"shape={record['image'].shape} | raw={raw_values} | mapped={mapped_values} | "
            f"affine_match={record['affine_match']}"
        )
        filename = f"{index:02d}_{category}_{record['split']}_{record['modality']}_{record['patient_id']}_slice{slice_index:02d}.png"
        output_path = args.output / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        rows.append({
            "asset": filename,
            "category": category,
            "patient_id": record["patient_id"],
            "split": record["split"],
            "modality": record["modality"],
            "slice_index": slice_index,
            "shape": ",".join(str(x) for x in record["image"].shape),
            "raw_values_in_slice": ",".join(str(x) for x in raw_values),
            "mapped_values_in_slice": ",".join(str(x) for x in mapped_values),
            "mapping": record["mapping"],
            "foreground_voxels": int(np.count_nonzero(label_slice)),
            "affine_match": record["affine_match"],
        })

    with (args.output / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rendered {len(rows)} assets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
