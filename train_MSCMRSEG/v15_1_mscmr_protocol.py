"""Frozen V15.1 MS-CMR label semantics and safe mapping helpers."""

from __future__ import annotations

import numpy as np


RAW_LABEL_TO_TRAIN_LABEL = {0: 0, 600: 1, 200: 2, 500: 3}
TRAIN_LABEL_TO_STRUCTURE_NAME = {
    0: "Background",
    1: "RV blood pool",
    2: "LV myocardium",
    3: "LV blood pool",
}
TRAIN_VALUE_SET = frozenset(TRAIN_LABEL_TO_STRUCTURE_NAME)
RAW_VALUE_SET = frozenset(RAW_LABEL_TO_TRAIN_LABEL)


def raw_label_to_train_label(label: np.ndarray) -> np.ndarray:
    """Map raw 0/200/500/600 labels or validate already-mapped 0/1/2/3 labels.

    Unknown values are always fatal; they are never folded into background.
    """
    values = frozenset(np.unique(label).tolist())
    if values.issubset(TRAIN_VALUE_SET):
        return label.astype(np.uint8, copy=False)
    if not values.issubset(RAW_VALUE_SET):
        unknown = sorted(values - RAW_VALUE_SET)
        raise ValueError(f"Unknown MS-CMR raw label values: {unknown}")
    output = np.empty(label.shape, dtype=np.uint8)
    assigned = np.zeros(label.shape, dtype=bool)
    for raw_value, train_value in RAW_LABEL_TO_TRAIN_LABEL.items():
        mask = label == raw_value
        output[mask] = train_value
        assigned |= mask
    if not assigned.all():
        raise AssertionError("MS-CMR mapping left unassigned voxels")
    return output


def train_label_to_structure_name(train_label: int) -> str:
    try:
        return TRAIN_LABEL_TO_STRUCTURE_NAME[int(train_label)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unknown MS-CMR train label: {train_label}") from error

