#!/usr/bin/env python3
"""Patient-volume macro evaluator for COVID-19 Official20 V2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


@dataclass(frozen=True)
class SlicePrediction:
    patient_id: str
    slice_idx: int
    pred: np.ndarray
    gt: np.ndarray
    spacing: tuple[float, float, float]


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded


def surface_distances_mm(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    if not pred_surface.any() and not gt_surface.any():
        return np.array([0.0], dtype=np.float64)
    if not pred_surface.any() or not gt_surface.any():
        return np.array([np.inf], dtype=np.float64)
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    return np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]]).astype(np.float64)


def binary_volume_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = 2.0 * intersection / max(pred_sum + gt_sum, 1)
        iou = intersection / max(union, 1)
    distances = surface_distances_mm(pred, gt, spacing)
    hd95 = float(np.percentile(distances, 95))
    asd = float(np.mean(distances))
    return {"dice": float(dice), "iou": float(iou), "hd95_mm": hd95, "asd_mm": asd}


def reconstruct_patient_volume(slices: list[SlicePrediction]) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    if not slices:
        raise ValueError("missing patient slices")
    patient_ids = {item.patient_id for item in slices}
    if len(patient_ids) != 1:
        raise ValueError(f"mixed patient ids: {sorted(patient_ids)}")
    ordered = sorted(slices, key=lambda item: item.slice_idx)
    expected = list(range(ordered[0].slice_idx, ordered[-1].slice_idx + 1))
    actual = [item.slice_idx for item in ordered]
    if actual != expected or expected[0] != 0:
        raise ValueError(f"missing or non-contiguous slices for {ordered[0].patient_id}")
    if len(set(actual)) != len(actual):
        raise ValueError(f"duplicate slice indices for {ordered[0].patient_id}")
    shape_set = {item.pred.shape for item in ordered} | {item.gt.shape for item in ordered}
    if len(shape_set) != 1:
        raise ValueError(f"inconsistent 2D slice shapes for {ordered[0].patient_id}: {shape_set}")
    spacing_set = {item.spacing for item in ordered}
    if len(spacing_set) != 1:
        raise ValueError(f"inconsistent spacing for {ordered[0].patient_id}: {spacing_set}")
    pred_vol = np.stack([item.pred.astype(bool) for item in ordered], axis=2)
    gt_vol = np.stack([item.gt.astype(bool) for item in ordered], axis=2)
    return pred_vol, gt_vol, ordered[0].spacing


def patient_volume_macro(slices: Iterable[SlicePrediction]) -> dict[str, object]:
    grouped: dict[str, list[SlicePrediction]] = defaultdict(list)
    for item in slices:
        grouped[item.patient_id].append(item)
    if not grouped:
        raise ValueError("no patient predictions supplied")
    per_patient = {}
    for patient_id, patient_slices in sorted(grouped.items()):
        pred, gt, spacing = reconstruct_patient_volume(patient_slices)
        per_patient[patient_id] = binary_volume_metrics(pred, gt, spacing)
    macro = {
        key: float(np.mean([metrics[key] for metrics in per_patient.values()]))
        for key in ["dice", "iou", "hd95_mm", "asd_mm"]
    }
    return {
        "primary_validation_metric": "PATIENT_VOLUME_MACRO_DICE",
        "aggregation": "PATIENT_VOLUME_MACRO",
        "unit": "mm",
        "per_patient": per_patient,
        "macro": macro,
    }
