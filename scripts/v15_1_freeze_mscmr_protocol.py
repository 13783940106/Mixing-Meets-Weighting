#!/usr/bin/env python3
"""Freeze the single BayeSeg MS-CMR LGE split and nominal 5%/10% subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    patient_rows = read_tsv(args.manifests / "mscmr_patient_manifest.tsv")
    header_rows = read_tsv(args.audit / "mscmr_image_header_inventory.tsv")
    slices = {
        (row["patient_id"], row["split"], row["modality"]): int(row["slice_count_axis2"])
        for row in header_rows if row["role"] == "image"
    }
    lge_rows = [row for row in patient_rows if row["modality"] == "LGE"]
    train_rows = sorted((row for row in lge_rows if row["split"] == "train"), key=lambda row: row["patient_id"])
    val_rows = sorted((row for row in lge_rows if row["split"] == "val"), key=lambda row: row["patient_id"])
    test_rows = sorted((row for row in lge_rows if row["split"] == "test"), key=lambda row: row["patient_id"])
    if (len(train_rows), len(val_rows), len(test_rows)) != (25, 5, 15):
        raise RuntimeError(f"Expected BayeSeg 25/5/15 LGE split, got {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    train_ids = [row["patient_id"] for row in train_rows]
    permutation = sorted(train_ids)
    random.Random(args.seed).shuffle(permutation)
    # Floor keeps the number of fully labeled patients within each nominal budget.
    n5 = max(1, math.floor(len(train_ids) * 0.05))
    n10 = max(1, math.floor(len(train_ids) * 0.10))
    labeled5 = set(permutation[:n5])
    labeled10 = set(permutation[:n10])
    if not labeled5 < labeled10:
        raise RuntimeError("The preregistered 5% subset must be a strict subset of the 10% subset")

    rank = {patient_id: index + 1 for index, patient_id in enumerate(permutation)}
    fields = ["patient_id", "split", "modality", "image_path", "label_path", "slice_count", "selection_rank", "seed", "patient_fraction", "slice_fraction", "training_label_access"]
    total_train_slices = sum(slices[(row["patient_id"], "train", "LGE")] for row in train_rows)

    def build_rows(selected: set[str], label_access: bool) -> list[dict[str, object]]:
        selected_slices = sum(slices[(row["patient_id"], "train", "LGE")] for row in train_rows if row["patient_id"] in selected)
        output = []
        for row in train_rows:
            include = row["patient_id"] in selected
            if not include:
                continue
            output.append({
                "patient_id": row["patient_id"],
                "split": "train",
                "modality": "LGE",
                "image_path": str((args.raw_root / row["image_paths"]).resolve()),
                "label_path": str((args.raw_root / row["label_paths"]).resolve()) if label_access else "WITHHELD_FROM_TRAINING",
                "slice_count": slices[(row["patient_id"], "train", "LGE")],
                "selection_rank": rank[row["patient_id"]],
                "seed": args.seed,
                "patient_fraction": len(selected) / len(train_rows),
                "slice_fraction": selected_slices / total_train_slices,
                "training_label_access": "ALLOWED" if label_access else "FORBIDDEN",
            })
        return output

    manifests = {
        "mscmr_5pct_labeled_manifest.tsv": build_rows(labeled5, True),
        "mscmr_5pct_unlabeled_manifest.tsv": build_rows(set(train_ids) - labeled5, False),
        "mscmr_10pct_labeled_manifest.tsv": build_rows(labeled10, True),
        "mscmr_10pct_unlabeled_manifest.tsv": build_rows(set(train_ids) - labeled10, False),
    }
    for name, rows in manifests.items():
        write_tsv(args.audit / name, rows, fields)

    for ratio, labeled in (("5pct", labeled5), ("10pct", labeled10)):
        unlabeled = set(train_ids) - labeled
        (args.audit / f"mscmr_{ratio}_labeled_patients.txt").write_text("\n".join(sorted(labeled)) + "\n", encoding="utf-8")
        (args.audit / f"mscmr_{ratio}_unlabeled_patients.txt").write_text("\n".join(sorted(unlabeled)) + "\n", encoding="utf-8")

    split_fields = ["patient_id", "split", "modality", "image_path", "label_path", "slice_count", "role", "training_access"]
    split_manifest = []
    for row in train_rows + val_rows + test_rows:
        split = row["split"]
        split_manifest.append({
            "patient_id": row["patient_id"],
            "split": split,
            "modality": "LGE",
            "image_path": str((args.raw_root / row["image_paths"]).resolve()),
            "label_path": str((args.raw_root / row["label_paths"]).resolve()),
            "slice_count": slices[(row["patient_id"], split, "LGE")],
            "role": "train_pool" if split == "train" else "validation" if split == "val" else "formal_test",
            "training_access": "IMAGE_ONLY_AFTER_RATIO_MANIFEST" if split == "train" else "METRIC_ONLY" if split == "val" else "FROZEN_BEST_ONE_TIME_ONLY",
        })
    write_tsv(args.audit / "mscmr_split_manifest.tsv", split_manifest, split_fields)

    protocol = {
        "dataset_variant": "BayeSeg MS-CMRSeg research package",
        "task_modality": "LGE_only",
        "split_count": 1,
        "split": {"train_patients": 25, "validation_patients": 5, "formal_test_patients": 15},
        "selection_seed": args.seed,
        "selection_algorithm": "sort canonical case IDs; random.Random(seed).shuffle; take prefix",
        "rounding_rule": "floor(number_of_train_patients * nominal_ratio), minimum one; never exceed nominal patient budget",
        "nesting_rule": "single preregistered permutation; 5pct is a strict prefix/subset of 10pct",
        "train_slice_count": total_train_slices,
        "5pct": {
            "labeled_patients": sorted(labeled5),
            "labeled_patient_count": len(labeled5),
            "labeled_slice_count": sum(slices[(case, "train", "LGE")] for case in labeled5),
            "unlabeled_patient_count": len(train_ids) - len(labeled5),
            "unlabeled_slice_count": total_train_slices - sum(slices[(case, "train", "LGE")] for case in labeled5),
        },
        "10pct": {
            "labeled_patients": sorted(labeled10),
            "labeled_patient_count": len(labeled10),
            "labeled_slice_count": sum(slices[(case, "train", "LGE")] for case in labeled10),
            "unlabeled_patient_count": len(train_ids) - len(labeled10),
            "unlabeled_slice_count": total_train_slices - sum(slices[(case, "train", "LGE")] for case in labeled10),
        },
        "test_isolation": "test/LGR images and labels excluded from all training and validation; one formal test after validation-best freeze per run",
    }
    protocol_path = args.audit / "mscmr_frozen_data_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    protocol["artifact_sha256"] = {name: sha256(args.audit / name) for name in manifests}
    protocol["artifact_sha256"]["mscmr_split_manifest.tsv"] = sha256(args.audit / "mscmr_split_manifest.tsv")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
