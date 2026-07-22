#!/usr/bin/env python3
"""Build immutable-data manifests and medical-content audits for MS-CMRSeg."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


CASE_RE = re.compile(r"(?:subject|patient)(\d+)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_tuple(values: object) -> str:
    return ",".join(str(value) for value in values)


def identify(relative: Path) -> dict[str, str]:
    parts = relative.parts
    split = parts[0]
    role = "label" if "labels" in parts else "image" if "images" in parts else "other"
    match = CASE_RE.search(relative.name)
    case_number = int(match.group(1)) if match else None
    patient_id = f"case{case_number:02d}" if case_number is not None else "UNRESOLVED"
    if split in {"train", "val"}:
        modality = "LGE"
        modality_evidence = "DE filename under train/val"
    elif len(parts) > 1 and parts[1] == "LGR":
        modality = "LGE"
        modality_evidence = "LGR directory and _LGE filename"
    elif len(parts) > 1 and parts[1] == "C0":
        modality = "bSSFP"
        modality_evidence = "C0 directory and _bSSFP filename"
    elif len(parts) > 1 and parts[1] == "T2":
        modality = "T2"
        modality_evidence = "T2 directory and _T2 filename"
    else:
        modality = "UNRESOLVED"
        modality_evidence = "unrecognized path"
    return {
        "split": split,
        "role": role,
        "patient_id": patient_id,
        "case_number": "" if case_number is None else str(case_number),
        "modality": modality,
        "modality_evidence": modality_evidence,
    }


def expected_image_for_label(label: Path) -> Path:
    image_name = label.name.replace("_manual.nii.gz", ".nii.gz")
    return label.parent.parent / "images" / image_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    all_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    header_rows: list[dict[str, object]] = []
    label_value_rows: list[dict[str, object]] = []
    corrupt_rows: list[dict[str, object]] = []
    nifti_cache: dict[Path, tuple[nib.spatialimages.SpatialImage, np.ndarray]] = {}

    for path in files:
        relative = path.relative_to(root)
        identity = identify(relative)
        common = {
            "relative_path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path),
            "file_type": "NIfTI_GZIP" if path.name.endswith(".nii.gz") else path.suffix.lower().lstrip("."),
            "patient_id": identity["patient_id"],
            "case_number": identity["case_number"],
            "modality": identity["modality"],
            "modality_evidence": identity["modality_evidence"],
            "role": identity["role"],
            "split": identity["split"],
            "volume_or_slice": "volume" if path.name.endswith(".nii.gz") else "unknown",
            "empty_file": path.stat().st_size == 0,
        }
        all_rows.append(common)
        if identity["role"] == "image":
            image_rows.append(common)
        elif identity["role"] == "label":
            label_rows.append(common)

        if not path.name.endswith(".nii.gz"):
            continue
        try:
            nii = nib.load(path)
            data = np.asanyarray(nii.dataobj)
            nifti_cache[path] = (nii, data)
            finite = np.isfinite(data)
            zooms = nii.header.get_zooms()[: data.ndim]
            affine = np.asarray(nii.affine, dtype=float)
            spacing = np.asarray(zooms[:3], dtype=float)
            direction = affine[:3, :3] / np.where(spacing == 0, 1.0, spacing)[None, :]
            header_rows.append({
                "relative_path": relative.as_posix(),
                "patient_id": identity["patient_id"],
                "split": identity["split"],
                "modality": identity["modality"],
                "role": identity["role"],
                "shape": fmt_tuple(data.shape),
                "ndim": data.ndim,
                "dtype": str(data.dtype),
                "spacing": fmt_tuple(float(x) for x in zooms),
                "origin": fmt_tuple(float(x) for x in affine[:3, 3]),
                "direction": fmt_tuple(float(x) for x in direction.reshape(-1)),
                "orientation": "".join(nib.aff2axcodes(affine)),
                "qform_code": int(nii.header["qform_code"]),
                "sform_code": int(nii.header["sform_code"]),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "nan_count": int(np.isnan(data).sum()),
                "inf_count": int(np.isinf(data).sum()),
                "finite": bool(finite.all()),
                "nonzero_voxels": int(np.count_nonzero(data)),
                "nonzero_ratio": float(np.count_nonzero(data) / data.size),
                "slice_count_axis2": data.shape[2] if data.ndim >= 3 else 1,
            })
            if identity["role"] == "label":
                values, counts = np.unique(data, return_counts=True)
                label_value_rows.append({
                    "relative_path": relative.as_posix(),
                    "patient_id": identity["patient_id"],
                    "split": identity["split"],
                    "modality": identity["modality"],
                    "dtype": str(data.dtype),
                    "unique_values": fmt_tuple(values.tolist()),
                    "voxel_counts": fmt_tuple(int(x) for x in counts),
                    "foreground_voxels": int(np.count_nonzero(data)),
                    "empty_foreground": int(np.count_nonzero(data)) == 0,
                    "unknown_value_set": not set(values.tolist()).issubset({0, 1, 2, 3, 200, 500, 600}),
                })
        except Exception as error:  # every failure is recorded and blocks admission
            corrupt_rows.append({"relative_path": relative.as_posix(), "error_type": type(error).__name__, "error": str(error)})

    pair_rows: list[dict[str, object]] = []
    paired_images: set[Path] = set()
    for label in sorted(path for path in files if "/labels/" in path.as_posix()):
        image = expected_image_for_label(label)
        paired_images.add(image)
        identity = identify(label.relative_to(root))
        row: dict[str, object] = {
            "patient_id": identity["patient_id"],
            "split": identity["split"],
            "modality": identity["modality"],
            "image_path": image.relative_to(root).as_posix() if image.is_relative_to(root) else str(image),
            "label_path": label.relative_to(root).as_posix(),
            "image_exists": image.exists(),
            "label_exists": label.exists(),
        }
        if image in nifti_cache and label in nifti_cache:
            image_nii, image_data = nifti_cache[image]
            label_nii, label_data = nifti_cache[label]
            row.update({
                "shape_match": image_data.shape == label_data.shape,
                "affine_match_atol_1e_5": bool(np.allclose(image_nii.affine, label_nii.affine, atol=1e-5, rtol=0)),
                "spacing_match_atol_1e_6": bool(np.allclose(image_nii.header.get_zooms()[:3], label_nii.header.get_zooms()[:3], atol=1e-6, rtol=0)),
                "orientation_match": nib.aff2axcodes(image_nii.affine) == nib.aff2axcodes(label_nii.affine),
                "image_finite": bool(np.isfinite(image_data).all()),
                "label_finite": bool(np.isfinite(label_data).all()),
                "label_unique_values": fmt_tuple(np.unique(label_data).tolist()),
                "label_foreground_voxels": int(np.count_nonzero(label_data)),
                "pair_decision": "PASS" if image_data.shape == label_data.shape and np.allclose(image_nii.affine, label_nii.affine, atol=1e-5, rtol=0) and np.isfinite(image_data).all() and np.isfinite(label_data).all() else "FAIL",
            })
        else:
            row.update({"shape_match": False, "affine_match_atol_1e_5": False, "spacing_match_atol_1e_6": False, "orientation_match": False, "image_finite": False, "label_finite": False, "label_unique_values": "", "label_foreground_voxels": "", "pair_decision": "FAIL"})
        pair_rows.append(row)

    orphan_images = [path for path in files if "/images/" in path.as_posix() and path not in paired_images]

    patient_groups: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in all_rows:
        key = (str(row["patient_id"]), str(row["split"]), str(row["modality"]))
        group = patient_groups.setdefault(key, {
            "patient_id": row["patient_id"], "split": row["split"], "modality": row["modality"],
            "image_paths": [], "label_paths": [],
        })
        group[f"{row['role']}_paths"].append(row["relative_path"])
    patient_rows: list[dict[str, object]] = []
    for group in patient_groups.values():
        images_for_group = group.pop("image_paths")
        labels_for_group = group.pop("label_paths")
        patient_rows.append({
            **group,
            "image_count": len(images_for_group),
            "label_count": len(labels_for_group),
            "image_paths": ",".join(images_for_group),
            "label_paths": ",".join(labels_for_group),
            "pair_complete": len(images_for_group) == 1 and len(labels_for_group) == 1,
        })
    patient_rows.sort(key=lambda row: (row["split"], row["modality"], row["patient_id"]))

    modality_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        modality_counts[(str(row["split"]), str(row["modality"]))][str(row["role"])] += 1
        modality_counts[(str(row["split"]), str(row["modality"]))]["patients"] = len({str(item["patient_id"]) for item in all_rows if item["split"] == row["split"] and item["modality"] == row["modality"]})
    modality_rows = [
        {"split": key[0], "modality": key[1], "patient_count": counts["patients"], "image_count": counts["image"], "label_count": counts["label"]}
        for key, counts in sorted(modality_counts.items())
    ]

    split_sets: dict[str, set[str]] = defaultdict(set)
    lge_split_sets: dict[str, set[str]] = defaultdict(set)
    for row in patient_rows:
        split_sets[str(row["split"])].add(str(row["patient_id"]))
        if row["modality"] == "LGE":
            lge_split_sets[str(row["split"])].add(str(row["patient_id"]))
    split_rows: list[dict[str, object]] = []
    for scope, sets in (("all_modalities", split_sets), ("LGE_only", lge_split_sets)):
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(sets[left] & sets[right])
            split_rows.append({"scope": scope, "left_split": left, "right_split": right, "overlap_count": len(overlap), "overlap_patient_ids": ",".join(overlap), "decision": "PASS" if not overlap else "FAIL"})

    all_fields = ["relative_path", "size_bytes", "mtime_ns", "sha256", "file_type", "patient_id", "case_number", "modality", "modality_evidence", "role", "split", "volume_or_slice", "empty_file"]
    write_tsv(args.manifests / "mscmr_all_files.tsv", all_rows, all_fields)
    write_tsv(args.manifests / "mscmr_images.tsv", image_rows, all_fields)
    write_tsv(args.manifests / "mscmr_labels.tsv", label_rows, all_fields)
    write_tsv(args.manifests / "mscmr_patient_manifest.tsv", patient_rows, ["patient_id", "split", "modality", "image_count", "label_count", "image_paths", "label_paths", "pair_complete"])
    write_tsv(args.manifests / "mscmr_modality_manifest.tsv", modality_rows, ["split", "modality", "patient_count", "image_count", "label_count"])
    write_tsv(args.manifests / "mscmr_split_files.tsv", [], ["relative_path", "sha256", "notes"])
    write_tsv(args.audit / "mscmr_image_header_inventory.tsv", header_rows, list(header_rows[0]))
    write_tsv(args.audit / "mscmr_label_value_inventory.tsv", label_value_rows, list(label_value_rows[0]))
    write_tsv(args.audit / "mscmr_image_label_pair_audit.tsv", pair_rows, list(pair_rows[0]))
    write_tsv(args.audit / "mscmr_corrupt_file_inventory.tsv", corrupt_rows, ["relative_path", "error_type", "error"])
    write_tsv(args.audit / "mscmr_split_overlap_check.tsv", split_rows, ["scope", "left_split", "right_split", "overlap_count", "overlap_patient_ids", "decision"])

    summary = {
        "root": str(root),
        "file_count": len(files),
        "image_count": len(image_rows),
        "label_count": len(label_rows),
        "patient_modality_records": len(patient_rows),
        "corrupt_file_count": len(corrupt_rows),
        "pair_count": len(pair_rows),
        "failed_pair_count": sum(row["pair_decision"] != "PASS" for row in pair_rows),
        "orphan_image_count": len(orphan_images),
        "orphan_images": [path.relative_to(root).as_posix() for path in orphan_images],
        "label_value_sets": dict(Counter(row["unique_values"] for row in label_value_rows)),
        "split_overlap": split_rows,
        "decision": "PASS" if not corrupt_rows and not orphan_images and all(row["pair_decision"] == "PASS" for row in pair_rows) else "FAIL",
    }
    (args.audit / "mscmr_content_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
