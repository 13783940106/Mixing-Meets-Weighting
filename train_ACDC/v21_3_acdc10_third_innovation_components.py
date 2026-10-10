"""V21.3 ACDC10 third-innovation components.

Every mechanism is default-off and consumes only tensors already produced by
the canonical beta-FFT + Mask Token + AdaMix ACDC10 training loop.  No function
below reads test data, changes data loading, or mutates Mask Token/AdaMix state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


CANDIDATE_IDS = {
    "V21_3_X01_CLASS_DISTRIBUTION_GUARD",
    "V21_3_X02_UNCERTAINTY_CROSS_BRANCH_FILTER",
    "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT",
    "V21_3_X04_REGION_PROTOTYPE_SEMANTIC_ALIGN",
    "V21_3_X05_LABEL_ADAPTIVE_BOUNDARY_OPT",
    "V21_3_X06_LATE_STABILITY_RELIABILITY_RAMP",
}


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    return -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=1)


def _norm01(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dy[..., :, 1:] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    return dx + dy


def _class_distribution(prob: torch.Tensor) -> torch.Tensor:
    return prob.mean(dim=(0, 2, 3))


def _safe_scalar(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu())


def compute_v21_3_innovation_x(
    candidate_id: str,
    out_1: torch.Tensor,
    out_2: torch.Tensor,
    pre_a: torch.Tensor,
    pre_b: torch.Tensor,
    image_output_1_fft: torch.Tensor,
    image_output_2_fft: torch.Tensor,
    iter_num: int,
    max_iterations: int,
) -> tuple[torch.Tensor, dict]:
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"Unknown V21.3 candidate_id={candidate_id}")

    p1 = torch.softmax(out_1, dim=1)
    p2 = torch.softmax(out_2, dim=1)
    t1 = torch.softmax(pre_a.detach(), dim=1)
    t2 = torch.softmax(pre_b.detach(), dim=1)
    f1 = torch.softmax(image_output_1_fft, dim=1)
    f2 = torch.softmax(image_output_2_fft, dim=1)
    zero = p1.sum() * 0.0

    if candidate_id == "V21_3_X01_CLASS_DISTRIBUTION_GUARD":
        student_dist = 0.5 * (_class_distribution(p1) + _class_distribution(p2))
        teacher_dist = 0.5 * (_class_distribution(t1) + _class_distribution(t2))
        class_weight = torch.ones_like(student_dist)
        if class_weight.numel() > 1:
            class_weight[1] = 1.5
        loss = (class_weight * (student_dist - teacher_dist).abs()).mean()
        endpoint = {
            "class_distribution_drift": _safe_scalar((student_dist - teacher_dist).abs().mean()),
            "rv_reliability_weight": _safe_scalar(class_weight[1]) if class_weight.numel() > 1 else 0.0,
            "per_class_pseudo_coverage": [_safe_scalar(v) for v in student_dist],
        }
        return loss, endpoint

    if candidate_id == "V21_3_X02_UNCERTAINTY_CROSS_BRANCH_FILTER":
        mean_prob = 0.5 * (p1.detach() + p2.detach())
        reliability = (1.0 - _entropy(mean_prob) / torch.log(torch.tensor(float(p1.shape[1]), device=p1.device))).clamp(0.0, 1.0)
        loss = ((p1 - p2.detach()).pow(2).mean(dim=1) * reliability).mean()
        endpoint = {
            "uncertainty_reliability_mean": _safe_scalar(reliability.mean()),
            "suppressed_pixel_ratio": _safe_scalar((reliability < 0.5).float().mean()),
            "branch_disagreement_weighted": _safe_scalar(((p1.argmax(1) != p2.argmax(1)).float() * reliability).mean()),
        }
        return loss, endpoint

    if candidate_id == "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT":
        contour = _norm01(_spatial_gradient(0.5 * (p1.detach().max(1).values + p2.detach().max(1).values)))
        boundary_weight = (0.25 + contour).clamp(0.25, 1.25)
        loss = ((p1 - p2.detach()).abs().mean(dim=1) * boundary_weight).mean()
        endpoint = {
            "contour_uncertainty_mean": _safe_scalar(contour.mean()),
            "boundary_weight_mean": _safe_scalar(boundary_weight.mean()),
            "rv_boundary_pixel_ratio": _safe_scalar(((p1.argmax(1) == 1).float() * (contour > 0.5).float()).mean()),
        }
        return loss, endpoint

    if candidate_id == "V21_3_X04_REGION_PROTOTYPE_SEMANTIC_ALIGN":
        pred = p1.detach().argmax(dim=1)
        terms = []
        counts = []
        for cls in range(1, p1.shape[1]):
            mask = (pred == cls).float().unsqueeze(1)
            count = mask.sum().clamp_min(1.0)
            proto1 = (p1 * mask).sum(dim=(0, 2, 3)) / count
            proto2 = (p2 * mask).sum(dim=(0, 2, 3)) / count
            terms.append(F.mse_loss(proto1, proto2.detach()))
            counts.append(mask.mean())
        loss = torch.stack(terms).mean() if terms else zero
        endpoint = {
            "prototype_alignment_loss": _safe_scalar(loss),
            "per_class_prototype_count": [_safe_scalar(c) for c in counts],
            "rv_prototype_valid_ratio": _safe_scalar(counts[0]) if counts else 0.0,
        }
        return loss, endpoint

    if candidate_id == "V21_3_X05_LABEL_ADAPTIVE_BOUNDARY_OPT":
        mean_prob = 0.5 * (p1.detach() + p2.detach())
        entropy = _entropy(mean_prob) / torch.log(torch.tensor(float(p1.shape[1]), device=p1.device))
        boundary = _norm01(_spatial_gradient(mean_prob.max(1).values))
        weight = (entropy * boundary).clamp(0.0, 1.0)
        pseudo = mean_prob.argmax(dim=1)
        ce = F.cross_entropy(out_1, pseudo, reduction="none")
        loss = (ce * weight).mean()
        endpoint = {
            "label_adaptation_ratio": _safe_scalar((weight > 0.1).float().mean()),
            "boundary_optimization_loss": _safe_scalar(loss),
            "uncertain_boundary_ratio": _safe_scalar((weight > 0.5).float().mean()),
        }
        return loss, endpoint

    if candidate_id == "V21_3_X06_LATE_STABILITY_RELIABILITY_RAMP":
        progress = torch.tensor(float(iter_num) / max(float(max_iterations), 1.0), device=p1.device)
        late_weight = torch.sigmoid((progress - 0.65) * 12.0)
        variance_proxy = (p1.detach() - p2.detach()).pow(2).mean(dim=1)
        fft_variance_proxy = (f1.detach() - f2.detach()).pow(2).mean()
        reliability = (variance_proxy + fft_variance_proxy).clamp(0.0, 1.0)
        loss = late_weight * ((p1 - p2.detach()).pow(2).mean(dim=1) * reliability).mean()
        endpoint = {
            "late_reliability_weight": _safe_scalar(late_weight),
            "perturbation_variance_proxy": _safe_scalar(reliability.mean()),
            "weighted_consistency_mean": _safe_scalar(loss),
        }
        return loss, endpoint

    return zero, {"inactive": True}
