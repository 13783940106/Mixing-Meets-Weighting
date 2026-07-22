"""Frozen case-first 3D evaluator for V15.1 MS-CMRSeg LGE."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from train_MSCMRSEG.v15_1_mscmr_dataset import _normalize, read_patient_manifest, resize_slice
from train_MSCMRSEG.v15_1_mscmr_common import preserve_module_modes
from train_MSCMRSEG.v15_1_mscmr_protocol import raw_label_to_train_label

CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing) -> np.ndarray:
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return np.array([0.0])
    if not a.any() or not b.any():
        return np.array([np.inf])
    structure = np.ones((3, 3, 3), dtype=bool)
    a_surface = a ^ binary_erosion(a, structure=structure, border_value=0)
    b_surface = b ^ binary_erosion(b, structure=structure, border_value=0)
    dt_b = distance_transform_edt(~b_surface, sampling=spacing)
    dt_a = distance_transform_edt(~a_surface, sampling=spacing)
    return np.concatenate([dt_b[a_surface], dt_a[b_surface]])


def binary_metrics(pred: np.ndarray, target: np.ndarray, spacing) -> Dict[str, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = int(np.logical_and(pred, target).sum())
    denominator = int(pred.sum() + target.sum())
    union = int(np.logical_or(pred, target).sum())
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    distances = _surface_distances(pred, target, spacing)
    if not np.isfinite(distances).all():
        return {"dice": float(dice), "iou": float(iou), "hd95": float("inf"), "asd": float("inf")}
    return {"dice": float(dice), "iou": float(iou),
            "hd95": float(np.percentile(distances, 95)), "asd": float(distances.mean())}


@torch.inference_mode()
def predict_case(model, image: np.ndarray, image_size, device) -> np.ndarray:
    with preserve_module_modes(model):
        model.eval()
        prediction = np.zeros(image.shape, dtype=np.uint8)
        for slice_index in range(image.shape[2]):
            resized = resize_slice(image[:, :, slice_index], image_size, label=False)
            tensor = torch.from_numpy(resized)[None, None].to(device)
            pred_small = model(tensor).argmax(dim=1)[0].cpu().numpy()
            prediction[:, :, slice_index] = resize_slice(pred_small, image.shape[:2], label=True)
    return prediction


@torch.inference_mode()
def evaluate_manifest(model, data_root: str | Path, manifest: str | Path,
                      image_size=(256, 256), device="cuda", include_predictions=False,
                      role: str | None = None):
    root = Path(data_root)
    cases: List[Dict] = []
    for row in read_patient_manifest(manifest, role=role):
        image_nii = nib.load(str(root / row["image_path"]))
        label_nii = nib.load(str(root / row["label_path"]))
        image = _normalize(np.asanyarray(image_nii.dataobj))
        target = raw_label_to_train_label(np.asanyarray(label_nii.dataobj))
        prediction = predict_case(model, image, image_size, device)
        spacing = tuple(float(v) for v in label_nii.header.get_zooms()[:3])
        class_metrics = {CLASS_NAMES[c]: binary_metrics(prediction == c, target == c, spacing)
                         for c in CLASS_NAMES}
        case = {"patient_id": row["patient_id"], "spacing": spacing,
                "shape": list(target.shape), "classes": class_metrics,
                "mDice": float(np.mean([v["dice"] for v in class_metrics.values()])),
                "mIoU": float(np.mean([v["iou"] for v in class_metrics.values()])),
                "HD95": float(np.mean([v["hd95"] for v in class_metrics.values()])),
                "ASD": float(np.mean([v["asd"] for v in class_metrics.values()])),
                "predicted_foreground_ratio": float((prediction > 0).mean())}
        if include_predictions:
            case["prediction"] = prediction
        cases.append(case)
    aggregate = {}
    for name in CLASS_NAMES.values():
        aggregate[name] = {metric: float(np.mean([case["classes"][name][metric] for case in cases]))
                           for metric in ("dice", "iou", "hd95", "asd")}
    result = {
        "protocol": "case-first 3D raw prediction; background excluded; case mean then class macro mean",
        "class_order": ["RV", "MYO", "LV"],
        "postprocessing": "none",
        "empty_class": "both empty: Dice/IoU=1 and distances=0; one empty: distances=inf",
        "spacing_source": "label NIfTI header",
        "resize_restore": "bilinear image to network size; nearest prediction to original in-plane shape",
        "case_count": len(cases), "classes": aggregate, "cases": cases,
    }
    result.update({
        "mDice": float(np.mean([aggregate[n]["dice"] for n in aggregate])),
        "mIoU": float(np.mean([aggregate[n]["iou"] for n in aggregate])),
        "HD95": float(np.mean([aggregate[n]["hd95"] for n in aggregate])),
        "ASD": float(np.mean([aggregate[n]["asd"] for n in aggregate])),
    })
    return result
