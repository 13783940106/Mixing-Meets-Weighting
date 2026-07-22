#!/usr/bin/env python3
"""Audit every V15.1 label volume against the frozen mapping contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root / "train_MSCMRSEG"))
    from v15_1_mscmr_protocol import (  # pylint: disable=import-outside-toplevel
        RAW_LABEL_TO_TRAIN_LABEL,
        TRAIN_LABEL_TO_STRUCTURE_NAME,
        raw_label_to_train_label,
        train_label_to_structure_name,
    )

    rows = []
    failures = []
    for path in sorted(args.root.rglob("labels/*.nii.gz")):
        label = np.asanyarray(nib.load(path).dataobj)
        try:
            mapped = raw_label_to_train_label(label)
            mapped_values = sorted(np.unique(mapped).tolist())
            if not set(mapped_values).issubset(TRAIN_LABEL_TO_STRUCTURE_NAME):
                raise AssertionError(f"mapped values outside contract: {mapped_values}")
            rows.append({
                "path": path.relative_to(args.root).as_posix(),
                "raw_values": sorted(np.unique(label).tolist()),
                "mapped_values": mapped_values,
                "voxel_count_preserved": int(mapped.size) == int(label.size),
                "foreground_count_preserved": int(np.count_nonzero(mapped)) == int(np.count_nonzero(label)),
            })
        except Exception as error:  # all failures are reported
            failures.append({"path": path.relative_to(args.root).as_posix(), "error": repr(error)})

    unknown_raw_rejected = False
    unknown_train_rejected = False
    try:
        raw_label_to_train_label(np.asarray([[0, 999]], dtype=np.int16))
    except ValueError:
        unknown_raw_rejected = True
    try:
        train_label_to_structure_name(99)
    except ValueError:
        unknown_train_rejected = True

    summary = {
        "raw_label_to_train_label": RAW_LABEL_TO_TRAIN_LABEL,
        "train_label_to_structure_name": TRAIN_LABEL_TO_STRUCTURE_NAME,
        "label_volume_count": len(rows),
        "failure_count": len(failures),
        "unknown_raw_rejected": unknown_raw_rejected,
        "unknown_train_rejected": unknown_train_rejected,
        "all_voxel_counts_preserved": all(row["voxel_count_preserved"] for row in rows),
        "all_foreground_counts_preserved": all(row["foreground_count_preserved"] for row in rows),
        "decision": "PASS" if not failures and unknown_raw_rejected and unknown_train_rejected and len(rows) == 75 else "FAIL",
        "failures": failures,
        "volumes": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "volumes"}, indent=2))
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
