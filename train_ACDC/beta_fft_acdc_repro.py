import argparse
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment
from networks.net_factory import BCP_net
from utils import ramps, losses
from utils.displacement import process_image_batches
from val_2D import test_single_volume


def str2bool(x):
    return str(x).lower() in {"1", "true", "yes", "y"}


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data/ACDC")
parser.add_argument("--exp", type=str, default="beta_fft_acdc_official_10pct_seed42")
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--pre_iterations", type=int, default=10000)
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--labelnum", type=int, default=7)
parser.add_argument("--u_weight", type=float, default=0.5)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--consistency", type=float, default=0.1)
parser.add_argument("--consistency_rampup", type=float, default=200.0)
parser.add_argument("--magnitude", type=float, default=6.0)
parser.add_argument("--s_param", type=int, default=6)
parser.add_argument("--patch_size", type=int, default=64)
parser.add_argument("--h_size", type=int, default=4)
parser.add_argument("--w_size", type=int, default=4)
parser.add_argument("--top_num", type=int, default=4)
parser.add_argument("--run_root", type=str, default="../runs/beta_fft_acdc_official_10pct_seed42")
parser.add_argument("--pretrained_model", type=str, default="")
parser.add_argument("--data_manifest_name", type=str, default="official")
parser.add_argument("--resume_ckpt", type=str, default="")
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument("--split_name", type=str, default="official")
parser.add_argument("--labeled_list", type=str, default="")
parser.add_argument("--unlabeled_list", type=str, default="")
parser.add_argument("--val_list", type=str, default="")
parser.add_argument("--test_list", type=str, default="")
parser.add_argument("--input_mode", type=str, default="2d", choices=["2d", "25d_3ch", "25d_neighbor_relation"],
                    help="Input mode: 2d, 25d_3ch early fusion, or 25d_neighbor_relation")
parser.add_argument("--context_slices", type=int, default=1,
                    help="Number of context slices on each side (1 for 2.5D modes)")
parser.add_argument("--relation_type", type=str, default="center_anchored_neighbor", choices=["center_anchored_neighbor"])
parser.add_argument("--relation_stage", type=str, default="x0", choices=["x0"])
parser.add_argument("--relation_dim", type=int, default=8)
parser.add_argument("--use_ip_crm_weighting", type=str2bool, default=False,
                    help="Enable default-off Interior-Preserving Contour Risk Modeling weighting")
parser.add_argument("--ip_crm_contour_weight", type=float, default=0.5,
                    help="Downweight factor for IP-CRM safe contour-risk pixels")
parser.add_argument("--ip_crm_myo_core_conf_thresh", type=float, default=0.8,
                    help="Confidence threshold for protected MYO core pixels")
parser.add_argument("--ip_crm_contour_conf_thresh", type=float, default=0.7,
                    help="Confidence threshold for contour-risk uncertainty")
parser.add_argument("--ip_crm_boundary_width", type=int, default=2,
                    help="Prediction boundary width used by IP-CRM")
parser.add_argument("--ip_crm_log_interval", type=int, default=20,
                    help="Iteration interval for IP-CRM audit logging")
parser.add_argument("--use_ip_rdc", type=str2bool, default=False,
                    help="Enable default-off Interior-Preserving Regional Disagreement Control")
parser.add_argument("--ip_rdc_weight", type=float, default=0.0,
                    help="Auxiliary IP-RDC loss weight; zero keeps parent behavior")
parser.add_argument("--ip_rdc_confident_interior_thresh", type=float, default=0.8,
                    help="Confidence threshold for protected foreground interior")
parser.add_argument("--ip_rdc_boundary_width", type=int, default=2,
                    help="Predicted boundary width used by IP-RDC risk gate")
parser.add_argument("--ip_rdc_log_interval", type=int, default=200,
                    help="Iteration interval for IP-RDC audit logging")
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)
ce_loss = nn.CrossEntropyLoss()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dict_state():
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_dict_state(state):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and state.get("torch_cuda_random") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def get_ACDC_2DLargestCC(segmentation):
    from skimage.measure import label
    batch_list = []
    for i in range(segmentation.shape[0]):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        batch_list.append(class_list[0] + class_list[1] + class_list[2])
    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction="none")
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


def mix_loss_ip_crm_weighted(output, img_l, patch_l, mask, image_pixel_weight=None,
                             patch_pixel_weight=None, l_weight=1.0, u_weight=0.5, unlab=False):
    """Weighted variant for the explicitly enabled IP-CRM path only."""
    CE = nn.CrossEntropyLoss(reduction="none")
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    if image_pixel_weight is None:
        image_pixel_weight = torch.ones_like(mask, dtype=torch.float32)
    else:
        image_pixel_weight = image_pixel_weight.to(device=mask.device, dtype=torch.float32)
    if patch_pixel_weight is None:
        patch_pixel_weight = torch.ones_like(mask, dtype=torch.float32)
    else:
        patch_pixel_weight = patch_pixel_weight.to(device=mask.device, dtype=torch.float32)
    image_region_weight = mask.float() * image_pixel_weight
    patch_region_weight = patch_mask.float() * patch_pixel_weight
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), image_region_weight.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_region_weight.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * image_region_weight).sum() / (image_region_weight.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_region_weight).sum() / (patch_region_weight.sum() + 1e-16)
    return loss_dice, loss_ce


def _ip_crm_pool(mask, width):
    if width <= 0:
        return mask
    kernel = 2 * int(width) + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(width))


def _ip_crm_prediction_boundary(pred, width, num_classes):
    if width <= 0:
        return torch.zeros_like(pred, dtype=torch.bool)
    one_hot = F.one_hot(pred.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    foreground = one_hot[:, 1:, :, :]
    dilated = _ip_crm_pool(foreground, width)
    eroded = 1.0 - _ip_crm_pool(1.0 - foreground, width)
    return ((dilated > 0.5) & (eroded < 0.5)).any(dim=1)


def _ip_crm_lv_myo_transition(pred, width):
    myo_neighbor = _ip_crm_pool((pred == 2).float().unsqueeze(1), width).squeeze(1) > 0.5
    lv_neighbor = _ip_crm_pool((pred == 3).float().unsqueeze(1), width).squeeze(1) > 0.5
    return myo_neighbor & lv_neighbor


def compute_ip_crm_maps(logits, pseudo_label, num_classes, contour_weight, myo_core_conf_thresh,
                        contour_conf_thresh, boundary_width):
    """Compute inference-available IP-CRM maps without GT, test data, or external teacher."""
    prob = F.softmax(logits.detach(), dim=1)
    confidence, raw_pred = torch.max(prob, dim=1)
    pred = pseudo_label.detach().long() if pseudo_label is not None else raw_pred.long()
    boundary_band = _ip_crm_prediction_boundary(pred, boundary_width, num_classes)
    lv_myo_transition = _ip_crm_lv_myo_transition(pred, boundary_width)
    low_conf = confidence <= float(contour_conf_thresh)
    r_contour = boundary_band & (lv_myo_transition | low_conf)
    p_myo_core = (pred == 2) & (confidence >= float(myo_core_conf_thresh)) & (~boundary_band)
    r_safe = r_contour & (~p_myo_core)
    weight = torch.ones_like(confidence, dtype=torch.float32)
    weight = torch.where(r_safe, torch.full_like(weight, float(contour_weight)), weight)
    selected_fg = pred > 0
    finite = torch.isfinite(prob).all() and torch.isfinite(confidence).all() and torch.isfinite(weight).all()
    return {
        "weight": weight,
        "r_contour": r_contour,
        "p_myo_core": p_myo_core,
        "r_safe": r_safe,
        "lv_myo_transition": lv_myo_transition,
        "selected_fg_before": selected_fg.float().mean(),
        "selected_fg_after": (selected_fg.float() * weight).mean(),
        "effective_unsup_mask_ratio": weight.mean(),
        "finite": finite,
    }


def empty_ip_crm_stats():
    return {
        "ipcrm_enabled": 0,
        "R_contour_ratio": 0.0,
        "P_myo_core_ratio": 0.0,
        "R_safe_ratio": 0.0,
        "ipcrm_weight_mean": 1.0,
        "ipcrm_weight_min": 1.0,
        "ipcrm_weight_max": 1.0,
        "selected_fg_ratio_before_ipcrm": 0.0,
        "selected_fg_ratio_after_ipcrm": 0.0,
        "effective_unsup_mask_ratio": 1.0,
        "MYO_core_protected_ratio": 0.0,
        "LV_MYO_transition_ratio": 0.0,
        "loss_u_after_ipcrm": 0.0,
        "ipcrm_finite_check": 1,
    }


def summarize_ip_crm_maps(ipcrm_maps):
    stats = empty_ip_crm_stats()
    if not ipcrm_maps:
        return stats
    stats["ipcrm_enabled"] = 1
    stats["R_contour_ratio"] = float(torch.stack([m["r_contour"].float().mean() for m in ipcrm_maps]).mean().item())
    stats["P_myo_core_ratio"] = float(torch.stack([m["p_myo_core"].float().mean() for m in ipcrm_maps]).mean().item())
    stats["R_safe_ratio"] = float(torch.stack([m["r_safe"].float().mean() for m in ipcrm_maps]).mean().item())
    stats["ipcrm_weight_mean"] = float(torch.stack([m["weight"].mean() for m in ipcrm_maps]).mean().item())
    stats["ipcrm_weight_min"] = float(torch.stack([m["weight"].min() for m in ipcrm_maps]).min().item())
    stats["ipcrm_weight_max"] = float(torch.stack([m["weight"].max() for m in ipcrm_maps]).max().item())
    stats["selected_fg_ratio_before_ipcrm"] = float(torch.stack([m["selected_fg_before"] for m in ipcrm_maps]).mean().item())
    stats["selected_fg_ratio_after_ipcrm"] = float(torch.stack([m["selected_fg_after"] for m in ipcrm_maps]).mean().item())
    stats["effective_unsup_mask_ratio"] = float(torch.stack([m["effective_unsup_mask_ratio"] for m in ipcrm_maps]).mean().item())
    stats["MYO_core_protected_ratio"] = stats["P_myo_core_ratio"]
    stats["LV_MYO_transition_ratio"] = float(torch.stack([m["lv_myo_transition"].float().mean() for m in ipcrm_maps]).mean().item())
    stats["ipcrm_finite_check"] = int(all(bool(m["finite"].item() if torch.is_tensor(m["finite"]) else m["finite"]) for m in ipcrm_maps))
    return stats


def center_repeat_25d_view(batch):
    if args.input_mode != "25d_3ch" or batch.ndim != 4 or batch.shape[1] != 3:
        return batch
    center = batch[:, 1:2]
    return center.repeat(1, 3, 1, 1)


def empty_ip_rdc_stats():
    return {
        "ip_rdc_enabled": 0,
        "ip_rdc_weight": float(args.ip_rdc_weight),
        "ip_rdc_loss": 0.0,
        "risk_gate_mean": 0.0,
        "risk_gate_fg_mean": 0.0,
        "risk_gate_bg_mean": 0.0,
        "risk_gate_boundary_mean": 0.0,
        "interior_preserved_ratio": 1.0,
        "affected_fg_ratio": 0.0,
        "affected_boundary_ratio": 0.0,
        "center_neighbor_disagreement_mean": 0.0,
        "ip_rdc_finite_check": 1,
    }


def _safe_masked_mean(values, mask):
    mask_f = mask.float()
    return (values * mask_f).sum() / (mask_f.sum() + 1e-6)


def compute_ip_rdc_loss(model, batch, neighbor_logits):
    zero = neighbor_logits.sum() * 0.0
    if args.input_mode != "25d_3ch" or not args.use_ip_rdc or args.ip_rdc_weight <= 0:
        return zero, empty_ip_rdc_stats()

    with torch.no_grad():
        center_logits = forward_model(model, center_repeat_25d_view(batch), None)
        center_soft = torch.softmax(center_logits, dim=1).detach()
        center_conf, center_pred = torch.max(center_soft, dim=1)

    neighbor_soft = torch.softmax(neighbor_logits, dim=1)
    neighbor_soft_detached = neighbor_soft.detach()
    neighbor_conf, neighbor_pred = torch.max(neighbor_soft_detached, dim=1)
    class_diff = (neighbor_soft - center_soft) ** 2
    pixel_diff = class_diff.mean(dim=1)
    detached_diff = pixel_diff.detach()
    disagreement = (neighbor_pred != center_pred)

    boundary_neighbor = _ip_crm_prediction_boundary(neighbor_pred, args.ip_rdc_boundary_width, args.num_classes)
    boundary_center = _ip_crm_prediction_boundary(center_pred, args.ip_rdc_boundary_width, args.num_classes)
    boundary_band = boundary_neighbor | boundary_center
    fg_mask = (neighbor_pred > 0) | (center_pred > 0)
    bg_mask = ~fg_mask
    confident_interior = (
        fg_mask
        & (~boundary_band)
        & (neighbor_conf >= float(args.ip_rdc_confident_interior_thresh))
        & (center_conf >= float(args.ip_rdc_confident_interior_thresh))
    )

    diff_scale = detached_diff.flatten(1).mean(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    normalized_disagreement = (detached_diff / diff_scale).clamp(min=0.0, max=1.0)
    low_conf_risk = (1.0 - torch.minimum(neighbor_conf, center_conf)).clamp(min=0.0, max=1.0)
    risk_gate = torch.maximum(normalized_disagreement, low_conf_risk)
    risk_gate = torch.where(boundary_band | disagreement, risk_gate, torch.zeros_like(risk_gate))
    risk_gate = torch.where(confident_interior, torch.zeros_like(risk_gate), risk_gate)
    risk_gate = risk_gate.detach()

    raw_loss = (pixel_diff * risk_gate).sum() / (risk_gate.sum() + 1e-6)
    loss = float(args.ip_rdc_weight) * raw_loss
    affected = risk_gate > 1e-6
    interior_total = confident_interior.float().sum()
    if interior_total > 0:
        interior_preserved = ((~affected) & confident_interior).float().sum() / (interior_total + 1e-6)
    else:
        interior_preserved = torch.ones((), device=batch.device)
    stats = empty_ip_rdc_stats()
    stats.update({
        "ip_rdc_enabled": 1,
        "ip_rdc_loss": float(loss.detach().item()),
        "risk_gate_mean": float(risk_gate.mean().detach().item()),
        "risk_gate_fg_mean": float(_safe_masked_mean(risk_gate, fg_mask).detach().item()),
        "risk_gate_bg_mean": float(_safe_masked_mean(risk_gate, bg_mask).detach().item()),
        "risk_gate_boundary_mean": float(_safe_masked_mean(risk_gate, boundary_band).detach().item()),
        "interior_preserved_ratio": float(interior_preserved.detach().item()),
        "affected_fg_ratio": float(_safe_masked_mean(affected.float(), fg_mask).detach().item()),
        "affected_boundary_ratio": float(_safe_masked_mean(affected.float(), boundary_band).detach().item()),
        "center_neighbor_disagreement_mean": float(disagreement.float().mean().detach().item()),
        "ip_rdc_finite_check": int(torch.isfinite(loss).item() and torch.isfinite(risk_gate).all().item()),
    })
    return loss, stats


def summarize_ip_rdc_stats(stats_list):
    active = [s for s in stats_list if s.get("ip_rdc_enabled", 0)]
    if not active:
        return empty_ip_rdc_stats()
    out = empty_ip_rdc_stats()
    out["ip_rdc_enabled"] = 1
    out["ip_rdc_weight"] = float(args.ip_rdc_weight)
    for key in [
        "ip_rdc_loss", "risk_gate_mean", "risk_gate_fg_mean", "risk_gate_bg_mean",
        "risk_gate_boundary_mean", "interior_preserved_ratio", "affected_fg_ratio",
        "affected_boundary_ratio", "center_neighbor_disagreement_mean",
    ]:
        out[key] = float(np.mean([s[key] for s in active]))
    out["ip_rdc_finite_check"] = int(all(s.get("ip_rdc_finite_check", 0) for s in active))
    return out


RELATION_GRAD_ITERS = {1, 2, 5, 10, 20, 50, 100, 200}


def is_relation_mode():
    return args.input_mode == "25d_neighbor_relation"


def forward_model(model, image, neighbor_validity=None):
    if is_relation_mode():
        return model(image, neighbor_validity=neighbor_validity)
    return model(image)


def metric_to_percent(value: float) -> float:
    raw = float(value)
    if not (-1e-8 <= raw <= 1.0 + 1e-8):
        raise ValueError(f"metric_to_percent expects raw metric in [0,1], got {raw}")
    return raw * 100.0


def mix_validity(valid_a, valid_b, mask):
    if not is_relation_mode():
        return None
    return valid_a * mask.unsqueeze(1) + valid_b * (1 - mask.unsqueeze(1))


def beta_mix_validity(valid_a, valid_b, mix_factors):
    if not is_relation_mode():
        return None
    return valid_a * (1.0 - mix_factors) + valid_b * mix_factors


def relation_tensor_item(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return x


def relation_norm(model, pattern):
    total = 0.0
    for name, p in model.named_parameters():
        if pattern in name:
            total += float(p.detach().norm().item())
    return total


def relation_grad_norm(model, pattern):
    total = 0.0
    for name, p in model.named_parameters():
        if pattern in name and p.grad is not None:
            total += float(p.grad.detach().norm().item())
    return total


def get_relation_stats(model):
    if not hasattr(model, "relation_adapter"):
        return {}
    return {k: relation_tensor_item(v) for k, v in model.relation_adapter.last_stats.items()}


def write_relation_gradient_header(csv_path):
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter", "model", "p_weight_norm", "p_grad_norm",
            "neighbor_adapter_grad_norm", "wq_grad_norm", "wk_grad_norm", "wv_grad_norm",
            "prev_value_norm", "next_value_norm", "relation_output_norm", "residual_injection_norm",
            "mean_alpha_prev", "mean_alpha_next", "std_alpha_prev", "std_alpha_next", "attention_entropy",
            "valid_prev_pixel_ratio", "valid_next_pixel_ratio", "scores_shape", "alpha_shape",
        ])


def append_relation_gradient(csv_path, iter_num, model_name, model):
    if not is_relation_mode() or not hasattr(model, "relation_adapter"):
        return
    stats = get_relation_stats(model)
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            iter_num, model_name,
            relation_norm(model, "relation_adapter.out_proj.weight"),
            relation_grad_norm(model, "relation_adapter.out_proj"),
            relation_grad_norm(model, "relation_adapter.neighbor_adapter"),
            relation_grad_norm(model, "relation_adapter.query_proj"),
            relation_grad_norm(model, "relation_adapter.key_proj"),
            relation_grad_norm(model, "relation_adapter.value_proj"),
            stats.get("prev_value_norm", ""), stats.get("next_value_norm", ""),
            stats.get("relation_norm", ""), stats.get("residual_norm", ""),
            stats.get("mean_alpha_prev", ""), stats.get("mean_alpha_next", ""),
            stats.get("std_alpha_prev", ""), stats.get("std_alpha_next", ""),
            stats.get("attention_entropy", ""),
            stats.get("valid_prev_ratio", ""), stats.get("valid_next_ratio", ""),
            stats.get("scores_shape", ""), stats.get("alpha_shape", ""),
        ])


def patients_to_slices(dataset, patiens_num):
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    else:
        raise ValueError("Unsupported dataset")
    return ref_dict[str(patiens_num)]


def make_run_dirs():
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "audit").mkdir(exist_ok=True)
    shutil.copy(__file__, run_root / "source_snapshot.py")
    return run_root


def save_full_ckpt(path, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, global_step, best_metrics):
    ckpt = {
        "model_1": model_1.state_dict(),
        "model_2": model_2.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer1": optimizer1.state_dict(),
        "optimizer2": optimizer2.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "random_state": to_dict_state(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_full_ckpt(path, model_1, model_2, ema_model, optimizer1, optimizer2):
    ckpt = torch.load(path, map_location="cpu")
    model_1.load_state_dict(ckpt["model_1"])
    model_2.load_state_dict(ckpt["model_2"])
    ema_model.load_state_dict(ckpt["ema_model"])
    optimizer1.load_state_dict(ckpt["optimizer1"])
    optimizer2.load_state_dict(ckpt["optimizer2"])
    if "random_state" in ckpt:
        restore_dict_state(ckpt["random_state"])
    return ckpt


def mark_last_dynamics_checkpoint_saved(csv_path, global_step):
    lines = csv_path.read_text().splitlines()
    if len(lines) < 2:
        return
    header = lines[0].split(",")
    checkpoint_idx = header.index("checkpoint_saved") if "checkpoint_saved" in header else len(header) - 1
    rows = list(csv.reader(lines[1:]))
    for row in reversed(rows):
        if row and row[0] == str(global_step):
            if len(row) > checkpoint_idx:
                row[checkpoint_idx] = "1"
            break
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def finite_check(model, optimizer):
    for n, p in model.named_parameters():
        if torch.is_tensor(p) and p.is_floating_point() and not torch.isfinite(p).all():
            raise RuntimeError(f"Non-finite model tensor: {n}")
    for state in optimizer.state.values():
        for _, v in state.items():
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                raise RuntimeError("Non-finite optimizer tensor")


def write_dynamics_header(csv_path):
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter","global_step","total_loss","loss_1","loss_2","consistency_weight","repair_triggered",
            "fft_triggered","pred_fg_ratio_1","pred_fg_ratio_2","pseudo_fg_ratio_1","pseudo_fg_ratio_2",
            "branch_disagreement_ratio","branch_agreement_ratio","fft_mixing_triggered","fft_threshold",
            "loss_finite","grad_finite","logits_finite","checkpoint_saved",
            "model1_val_miou","model1_test_miou","model2_val_miou","model2_test_miou",
            "model1_val_rv_iou","model1_val_myo_iou","model1_val_lv_iou",
            "model2_val_rv_iou","model2_val_myo_iou","model2_val_lv_iou",
            "ipcrm_enabled","R_contour_ratio","P_myo_core_ratio","R_safe_ratio",
            "ipcrm_weight_mean","ipcrm_weight_min","ipcrm_weight_max",
            "selected_fg_ratio_before_ipcrm","selected_fg_ratio_after_ipcrm",
            "effective_unsup_mask_ratio","MYO_core_protected_ratio","LV_MYO_transition_ratio",
            "loss_u_after_ipcrm","ipcrm_finite_check",
            "ip_rdc_enabled","ip_rdc_weight","ip_rdc_loss","risk_gate_mean",
            "risk_gate_fg_mean","risk_gate_bg_mean","risk_gate_boundary_mean",
            "interior_preserved_ratio","affected_fg_ratio","affected_boundary_ratio",
            "center_neighbor_disagreement_mean","ip_rdc_finite_check",
        ])


def append_dynamics(csv_path, row):
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def convert_pretrained_center_identity(pre_state, in_chns_new=3):
    """Convert 1-channel pretrained weights to 3-channel center-identity.
    prev channel weight = 0, center = original, next = 0."""
    conv_key = None
    for k in pre_state.keys():
        if "inc.conv_conv.0.weight" in k or (".0.weight" in k and "conv" in k.lower()):
            # Find first Conv2d weight
            if pre_state[k].ndim == 4 and pre_state[k].shape[1] == 1:
                conv_key = k
                break
    if conv_key is None:
        for k in sorted(pre_state.keys()):
            if pre_state[k].ndim == 4 and pre_state[k].shape[1] == 1:
                conv_key = k
                break

    if conv_key is not None:
        W1 = pre_state[conv_key]
        Cout = W1.shape[0]
        ksize = W1.shape[2:]
        W3 = torch.zeros(Cout, in_chns_new, *ksize)
        W3[:, 1, :, :] = W1[:, 0, :, :]   # center channel = original
        # prev[:, 0] = 0, next[:, 2] = 0 by initialization
        logging.info(f"center_identity_conversion: key={conv_key} src_shape={list(W1.shape)} dst_shape={list(W3.shape)} prev_norm=0 center_norm={W1.norm().item():.4f} next_norm=0")
        pre_state[conv_key] = W3
        return pre_state, True
    logging.warning("center_identity_conversion: no matching 1ch Conv2d weight found")
    return pre_state, False


def train():
    run_root = make_run_dirs()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    seed_everything(args.seed)

    logging.basicConfig(filename=str(run_root / "train.log"), level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(str(args))
    logging.info(f"run_root={run_root}")

    in_chns = 3 if args.input_mode == "25d_3ch" else 1
    use_relation = is_relation_mode()
    logging.info(f"model_in_channels={in_chns} input_mode={args.input_mode} use_neighbor_relation={use_relation} relation_dim={args.relation_dim}")

    pre_snapshot_path = Path(args.pretrained_model) if args.pretrained_model else Path("./Pre_model/label7")
    pre_model = pre_snapshot_path / f"{args.model}_best_model.pth"
    if not pre_model.exists():
        pre_model = Path(args.root_path).parent / "best_results" / "ACDC_10" / "unet_best.pth"
    logging.info(f"pretrained_model={pre_model}")

    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model_1 = BCP_net(in_chns=in_chns, class_num=args.num_classes, ema=False, use_neighbor_relation=use_relation, relation_dim=args.relation_dim)
    model_2 = BCP_net(in_chns=in_chns, class_num=args.num_classes, ema=False, use_neighbor_relation=use_relation, relation_dim=args.relation_dim)
    ema_model = BCP_net(in_chns=in_chns, class_num=args.num_classes, ema=True, use_neighbor_relation=use_relation, relation_dim=args.relation_dim)

    if args.labeled_list or args.unlabeled_list:
        if not (args.labeled_list and args.unlabeled_list):
            raise ValueError("Both --labeled_list and --unlabeled_list are required for explicit clean split training")
        labeled_entries = [line.strip() for line in Path(args.labeled_list).read_text().splitlines() if line.strip()]
        unlabeled_entries = [line.strip() for line in Path(args.unlabeled_list).read_text().splitlines() if line.strip()]
        train_list_path = run_root / "audit" / f"{args.split_name}_train_slices.list"
        train_list_path.write_text("\n".join(labeled_entries + unlabeled_entries) + "\n")
        db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([WeakStrongAugment(args.image_size)]), list_file=str(train_list_path), input_mode=args.input_mode, context_slices=args.context_slices)
        labeled_slice = len(labeled_entries)
        logging.info("explicit_split split_name=%s labeled_list=%s unlabeled_list=%s labeled_slices=%s unlabeled_slices=%s", args.split_name, args.labeled_list, args.unlabeled_list, labeled_slice, len(unlabeled_entries))
    else:
        db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([WeakStrongAugment(args.image_size)]), input_mode=args.input_mode, context_slices=args.context_slices)
        labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    db_val = BaseDataSets(base_dir=args.root_path, split="val", list_file=args.val_list if args.val_list else None, input_mode=args.input_mode, context_slices=args.context_slices)
    db_test = BaseDataSets(base_dir=args.root_path, split="test", list_file=args.test_list if args.test_list else None, input_mode=args.input_mode, context_slices=args.context_slices)
    trainloader = DataLoader(
        db_train,
        batch_sampler=TwoStreamBatchSampler(list(range(0, labeled_slice)), list(range(labeled_slice, len(db_train))), args.batch_size, args.batch_size - args.labeled_bs),
        num_workers=16,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model_1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    optimizer2 = optim.SGD(model_2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    writer = SummaryWriter(str(run_root / "logs"))
    dynamics_csv = run_root / "audit" / "training_dynamics.csv"
    write_dynamics_header(dynamics_csv)
    relation_grad_csv = run_root / "audit" / "relation_gradients.csv"
    if use_relation:
        write_relation_gradient_header(relation_grad_csv)

    max_epoch = args.max_iterations // len(trainloader) + 1
    iter_num = 0
    start_epoch = 0
    best_performance1 = 0.0
    best_performance2 = 0.0

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        logging.info(f"resume_ckpt={resume_path}")
        resume_state = load_full_ckpt(resume_path, model_1, model_2, ema_model, optimizer1, optimizer2)
        iter_num = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        best_metrics = resume_state.get("best_metrics", {})
        best_performance1 = float(best_metrics.get("model1_best_val_dice", 0.0))
        best_performance2 = float(best_metrics.get("model2_best_val_dice", 0.0))
        logging.info("strict_resume_loaded epoch=%s global_step=%s", start_epoch, iter_num)
    else:
        load_state = torch.load(pre_model, map_location="cpu")
        pretrained_state = load_state["net"]
        if args.input_mode == "25d_3ch":
            pretrained_state, _converted = convert_pretrained_center_identity(pretrained_state, in_chns_new=3)
            logging.info("pretrained_center_identity_conversion_applied=%s", _converted)
        if use_relation:
            for _name, _model in [("ema", ema_model), ("model1", model_1), ("model2", model_2)]:
                missing, unexpected = _model.load_state_dict(pretrained_state, strict=False)
                bad_missing = [k for k in missing if not k.startswith("relation_adapter.")]
                if bad_missing or unexpected:
                    raise RuntimeError(f"relation pretrained load mismatch {_name}: bad_missing={bad_missing} unexpected={unexpected}")
                logging.info("relation_pretrained_load name=%s missing_relation_keys=%d unexpected=%d", _name, len(missing), len(unexpected))
        else:
            ema_model.load_state_dict(pretrained_state)
            model_1.load_state_dict(pretrained_state)
            model_2.load_state_dict(pretrained_state)
        if "opt" in load_state and args.input_mode == "2d":
            optimizer1.load_state_dict(load_state["opt"])
            optimizer2.load_state_dict(load_state["opt"])
        elif "opt" in load_state and args.input_mode in ("25d_3ch", "25d_neighbor_relation"):
            logging.info("pretrained_optimizer_skipped_for_%s", args.input_mode)

    start_batch_offset = iter_num % len(trainloader) if args.resume_ckpt else 0
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch in iterator:
        for batch_idx, sampled_batch in enumerate(trainloader):
            if epoch == start_epoch and batch_idx < start_batch_offset:
                continue
            if iter_num >= args.max_iterations:
                break
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            volume_batch_strong = sampled_batch["image_strong"].cuda()
            label_batch_strong = sampled_batch["label_strong"].cuda()
            if use_relation:
                validity_batch = sampled_batch["neighbor_validity"].cuda().float()
                validity_batch_strong = sampled_batch["neighbor_validity_strong"].cuda().float()
            else:
                validity_batch = None
                validity_batch_strong = None

            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_volume_batch_s = volume_batch_strong[:args.labeled_bs]
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]
            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            if use_relation:
                labeled_validity_batch = validity_batch[:args.labeled_bs]
                labeled_validity_batch_s = validity_batch_strong[:args.labeled_bs]
                valid_a, valid_b = validity_batch[:labeled_sub_bs], validity_batch[labeled_sub_bs:args.labeled_bs]
                uvalid_a, uvalid_b = validity_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], validity_batch[args.labeled_bs + unlabeled_sub_bs:]
                valid_a_s, valid_b_s = validity_batch_strong[:labeled_sub_bs], validity_batch_strong[labeled_sub_bs:args.labeled_bs]
                uvalid_a_s, uvalid_b_s = validity_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], validity_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            else:
                labeled_validity_batch = labeled_validity_batch_s = None
                valid_a = valid_b = uvalid_a = uvalid_b = None
                valid_a_s = valid_b_s = uvalid_a_s = uvalid_b_s = None

            with torch.no_grad():
                pre_a = forward_model(ema_model, uimg_a, uvalid_a)
                pre_b = forward_model(ema_model, uimg_b, uvalid_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                pre_a_s = forward_model(ema_model, uimg_a_s, uvalid_a_s)
                pre_b_s = forward_model(ema_model, uimg_b_s, uvalid_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)
                img_mask = torch.ones_like(img_a[:, 0])
                img_mask[:, : img_mask.shape[1] // 2, : img_mask.shape[2] // 2] = 0
                loss_mask = img_mask.clone()
                if args.use_ip_crm_weighting:
                    ipcrm_a = compute_ip_crm_maps(
                        pre_a, plab_a, args.num_classes, args.ip_crm_contour_weight,
                        args.ip_crm_myo_core_conf_thresh, args.ip_crm_contour_conf_thresh,
                        args.ip_crm_boundary_width,
                    )
                    ipcrm_b = compute_ip_crm_maps(
                        pre_b, plab_b, args.num_classes, args.ip_crm_contour_weight,
                        args.ip_crm_myo_core_conf_thresh, args.ip_crm_contour_conf_thresh,
                        args.ip_crm_boundary_width,
                    )
                    ipcrm_a_s = compute_ip_crm_maps(
                        pre_a_s, plab_a_s, args.num_classes, args.ip_crm_contour_weight,
                        args.ip_crm_myo_core_conf_thresh, args.ip_crm_contour_conf_thresh,
                        args.ip_crm_boundary_width,
                    )
                    ipcrm_b_s = compute_ip_crm_maps(
                        pre_b_s, plab_b_s, args.num_classes, args.ip_crm_contour_weight,
                        args.ip_crm_myo_core_conf_thresh, args.ip_crm_contour_conf_thresh,
                        args.ip_crm_boundary_width,
                    )
                    ipcrm_stats = summarize_ip_crm_maps([ipcrm_a, ipcrm_b, ipcrm_a_s, ipcrm_b_s])
                else:
                    ipcrm_a = ipcrm_b = ipcrm_a_s = ipcrm_b_s = None
                    ipcrm_stats = empty_ip_crm_stats()

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            net_input_unl_1 = uimg_a * img_mask.unsqueeze(1) + img_a * (1 - img_mask.unsqueeze(1))
            net_input_l_1 = img_b * img_mask.unsqueeze(1) + uimg_b * (1 - img_mask.unsqueeze(1))
            net_validity_unl_1 = mix_validity(uvalid_a, valid_a, img_mask) if use_relation else None
            net_validity_l_1 = mix_validity(valid_b, uvalid_b, img_mask) if use_relation else None
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0)
            net_validity_1 = torch.cat([net_validity_unl_1, net_validity_l_1], dim=0) if use_relation else None
            net_input_unl_2 = uimg_a_s * img_mask.unsqueeze(1) + img_a_s * (1 - img_mask.unsqueeze(1))
            net_input_l_2 = img_b_s * img_mask.unsqueeze(1) + uimg_b_s * (1 - img_mask.unsqueeze(1))
            net_validity_unl_2 = mix_validity(uvalid_a_s, valid_a_s, img_mask) if use_relation else None
            net_validity_l_2 = mix_validity(valid_b_s, uvalid_b_s, img_mask) if use_relation else None
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)
            net_validity_2 = torch.cat([net_validity_unl_2, net_validity_l_2], dim=0) if use_relation else None

            out_unl_1 = forward_model(model_1, net_input_unl_1, net_validity_unl_1)
            ip_rdc_loss_1, ip_rdc_stats_1 = compute_ip_rdc_loss(model_1, net_input_unl_1, out_unl_1)
            out_l_1 = forward_model(model_1, net_input_l_1, net_validity_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False)
            if args.use_ip_crm_weighting:
                unl_dice_1, unl_ce_1 = mix_loss_ip_crm_weighted(
                    out_unl_1, plab_a, lab_a, loss_mask, image_pixel_weight=ipcrm_a["weight"],
                    u_weight=args.u_weight, unlab=True,
                )
                l_dice_1, l_ce_1 = mix_loss_ip_crm_weighted(
                    out_l_1, lab_b, plab_b, loss_mask, patch_pixel_weight=ipcrm_b["weight"],
                    u_weight=args.u_weight,
                )
            else:
                unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
                l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            out_unl_2 = forward_model(model_2, net_input_unl_2, net_validity_unl_2)
            ip_rdc_loss_2, ip_rdc_stats_2 = compute_ip_rdc_loss(model_2, net_input_unl_2, out_unl_2)
            out_l_2 = forward_model(model_2, net_input_l_2, net_validity_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False)
            if args.use_ip_crm_weighting:
                unl_dice_2, unl_ce_2 = mix_loss_ip_crm_weighted(
                    out_unl_2, plab_a_s, lab_a_s, loss_mask, image_pixel_weight=ipcrm_a_s["weight"],
                    u_weight=args.u_weight, unlab=True,
                )
                l_dice_2, l_ce_2 = mix_loss_ip_crm_weighted(
                    out_l_2, lab_b_s, plab_b_s, loss_mask, patch_pixel_weight=ipcrm_b_s["weight"],
                    u_weight=args.u_weight,
                )
            else:
                unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
                l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))
            mix_factors = torch.tensor(np.random.beta(0.1, 0.1, size=(args.labeled_bs // 2, 1, 1, 1)), dtype=torch.float).cuda()
            batch_ux_mixed = uimg_a * (1.0 - mix_factors) + uimg_b * mix_factors
            batch_ux_validity = beta_mix_validity(uvalid_a, uvalid_b, mix_factors) if use_relation else None
            input_volume_batch = torch.cat([labeled_volume_batch, batch_ux_mixed], dim=0)
            input_validity_batch = torch.cat([labeled_validity_batch, batch_ux_validity], dim=0) if use_relation else None
            outputs = forward_model(model_1, input_volume_batch, input_validity_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(pre_a, dim=1)
                ema_output_ux1 = torch.softmax(pre_b, dim=1)
                batch_pred_mixed = ema_output_ux0 * (1.0 - mix_factors) + ema_output_ux1 * mix_factors
            loss_ce_mix = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_dice_mix = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss_mix = 0.5 * (loss_dice_mix + loss_ce_mix)
            consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - batch_pred_mixed) ** 2) * consistency_weight

            mix_factors_s = torch.tensor(np.random.beta(0.1, 0.1, size=(args.labeled_bs // 2, 1, 1, 1)), dtype=torch.float).cuda()
            batch_ux_mixed_s = uimg_a_s * (1.0 - mix_factors_s) + uimg_b_s * mix_factors_s
            batch_ux_validity_s = beta_mix_validity(uvalid_a_s, uvalid_b_s, mix_factors_s) if use_relation else None
            input_volume_batch_s = torch.cat([labeled_volume_batch_s, batch_ux_mixed_s], dim=0)
            input_validity_batch_s = torch.cat([labeled_validity_batch_s, batch_ux_validity_s], dim=0) if use_relation else None
            outputs_s = forward_model(model_1, input_volume_batch_s, input_validity_batch_s)
            outputs_soft_s = torch.softmax(outputs_s, dim=1)
            with torch.no_grad():
                ema_output_ux0_s = torch.softmax(pre_a_s, dim=1)
                ema_output_ux1_s = torch.softmax(pre_b_s, dim=1)
                batch_pred_mixed_s = ema_output_ux0_s * (1.0 - mix_factors_s) + ema_output_ux1_s * mix_factors_s
            loss_ce_mix_s = ce_loss(outputs_s[:args.labeled_bs], label_batch_strong[:args.labeled_bs].long())
            loss_dice_mix_s = dice_loss(outputs_soft_s[:args.labeled_bs], label_batch_strong[:args.labeled_bs].unsqueeze(1))
            supervised_loss_mix_s = 0.5 * (loss_dice_mix_s + loss_ce_mix_s)
            consistency_loss_s = torch.mean((outputs_soft_s[args.labeled_bs:] - batch_pred_mixed_s) ** 2) * consistency_weight

            image_patch_last_FFT = process_image_batches(net_input_1, net_input_2, 30)
            image_patch_last_FFT_validity = torch.cat([net_validity_1, net_validity_2], dim=0) if use_relation else None
            image_output_1_FFT = forward_model(model_1, image_patch_last_FFT, image_patch_last_FFT_validity)
            image_output_soft_1_FFT = torch.softmax(image_output_1_FFT, dim=1)
            pseudo_image_output_1_FFT = torch.argmax(image_output_soft_1_FFT.detach(), dim=1, keepdim=False)
            image_output_2_FFT = forward_model(model_2, image_patch_last_FFT, image_patch_last_FFT_validity)
            image_output_soft_2_FFT = torch.softmax(image_output_2_FFT, dim=1)
            pseudo_image_output_2_FFT = torch.argmax(image_output_soft_2_FFT.detach(), dim=1, keepdim=False)
            pseudo_supervision5 = dice_loss(image_output_soft_1_FFT, pseudo_image_output_2_FFT.unsqueeze(1))
            pseudo_supervision6 = dice_loss(image_output_soft_2_FFT, pseudo_image_output_1_FFT.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision5
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision6
            ip_rdc_loss = ip_rdc_loss_1 + ip_rdc_loss_2
            ip_rdc_stats = summarize_ip_rdc_stats([ip_rdc_stats_1, ip_rdc_stats_2])
            loss = loss_1 + loss_2 + consistency_loss + supervised_loss_mix + consistency_loss_s + supervised_loss_mix_s + ip_rdc_loss
            if args.use_ip_crm_weighting:
                ipcrm_stats["loss_u_after_ipcrm"] = float((unl_dice_1 + unl_ce_1 + l_dice_1 + l_ce_1 + unl_dice_2 + unl_ce_2 + l_dice_2 + l_ce_2).detach().item())

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            next_iter_num = iter_num + 1
            if use_relation and next_iter_num in RELATION_GRAD_ITERS:
                append_relation_gradient(relation_grad_csv, next_iter_num, "model1", model_1)
                append_relation_gradient(relation_grad_csv, next_iter_num, "model2", model_2)
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            update_model_ema(model_1, ema_model, 0.99)

            logits_finite = all(torch.isfinite(t).all() for t in [out_1, out_2, outputs, outputs_s, image_output_1_FFT, image_output_2_FFT])
            loss_finite = torch.isfinite(loss).item()
            grad_finite = all(torch.isfinite(p.grad).all().item() for p in list(model_1.parameters()) + list(model_2.parameters()) if p.grad is not None)
            fft_triggered = 1 if image_patch_last_FFT is not None else 0
            repair_triggered = 1
            branch_disagreement_ratio = (out_pseudo_1 != out_pseudo_2).float().mean().item()
            branch_agreement_ratio = 1.0 - branch_disagreement_ratio
            pred_fg_ratio_1 = (out_pseudo_1 > 0).float().mean().item()
            pred_fg_ratio_2 = (out_pseudo_2 > 0).float().mean().item()
            pseudo_fg_ratio_1 = (pseudo_image_output_1_FFT > 0).float().mean().item()
            pseudo_fg_ratio_2 = (pseudo_image_output_2_FFT > 0).float().mean().item()

            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/model1_loss", loss_1, iter_num)
            writer.add_scalar("info/model2_loss", loss_2, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            logging.info("iteration %d: loss: %f, model1_loss: %f, model2_loss: %f", iter_num, loss, loss_1, loss_2)
            logging.info("iteration %d: loss: %f, ICT_loss: %f, FFT_loss: %f", iter_num, loss, consistency_loss + supervised_loss_mix, pseudo_supervision5 + pseudo_supervision6)
            if args.use_ip_crm_weighting and (iter_num % max(1, args.ip_crm_log_interval) == 0):
                logging.info(
                    "iteration %d : ipcrm_enabled=%d R_contour_ratio=%.6f P_myo_core_ratio=%.6f "
                    "R_safe_ratio=%.6f weight_mean=%.6f selected_fg_before=%.6f selected_fg_after=%.6f "
                    "MYO_core_protected_ratio=%.6f LV_MYO_transition_ratio=%.6f finite=%d",
                    iter_num,
                    ipcrm_stats["ipcrm_enabled"],
                    ipcrm_stats["R_contour_ratio"],
                    ipcrm_stats["P_myo_core_ratio"],
                    ipcrm_stats["R_safe_ratio"],
                    ipcrm_stats["ipcrm_weight_mean"],
                    ipcrm_stats["selected_fg_ratio_before_ipcrm"],
                    ipcrm_stats["selected_fg_ratio_after_ipcrm"],
                    ipcrm_stats["MYO_core_protected_ratio"],
                    ipcrm_stats["LV_MYO_transition_ratio"],
                    ipcrm_stats["ipcrm_finite_check"],
                )
            if args.use_ip_rdc and (iter_num % max(1, args.ip_rdc_log_interval) == 0):
                logging.info(
                    "iteration %d : ip_rdc_enabled=%d ip_rdc_weight=%.6f ip_rdc_loss=%.8f "
                    "risk_gate_mean=%.6f risk_gate_fg_mean=%.6f risk_gate_bg_mean=%.6f "
                    "risk_gate_boundary_mean=%.6f interior_preserved_ratio=%.6f "
                    "affected_fg_ratio=%.6f affected_boundary_ratio=%.6f "
                    "center_neighbor_disagreement_mean=%.6f finite=%d",
                    iter_num,
                    ip_rdc_stats["ip_rdc_enabled"],
                    ip_rdc_stats["ip_rdc_weight"],
                    ip_rdc_stats["ip_rdc_loss"],
                    ip_rdc_stats["risk_gate_mean"],
                    ip_rdc_stats["risk_gate_fg_mean"],
                    ip_rdc_stats["risk_gate_bg_mean"],
                    ip_rdc_stats["risk_gate_boundary_mean"],
                    ip_rdc_stats["interior_preserved_ratio"],
                    ip_rdc_stats["affected_fg_ratio"],
                    ip_rdc_stats["affected_boundary_ratio"],
                    ip_rdc_stats["center_neighbor_disagreement_mean"],
                    ip_rdc_stats["ip_rdc_finite_check"],
                )

            # IoU fields placeholder (filled below when eval runs every 200 iters)
            m1_val_miou, m1_test_miou = "", ""
            m2_val_miou, m2_test_miou = "", ""
            m1_rv_iou, m1_myo_iou, m1_lv_iou = "", "", ""
            m2_rv_iou, m2_myo_iou, m2_lv_iou = "", "", ""

            if iter_num > 0 and iter_num % args.save_interval == 0:
                best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
                last_ckpt = run_root / "checkpoints" / "last.ckpt"
                save_full_ckpt(last_ckpt, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
                mark_last_dynamics_checkpoint_saved(dynamics_csv, iter_num)
                logging.info("checkpoint_saved path=%s epoch=%s global_step=%s", last_ckpt, epoch, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1, classes=args.num_classes, input_mode=args.input_mode)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                # metric_list: [num_classes-1, 3]  -> columns: 0=Dice, 1=HD95, 2=IoU
                performance1 = np.mean(metric_list, axis=0)[0]       # mDice (unchanged)
                performance1_iou = np.mean(metric_list, axis=0)[2]   # mIoU (new)
                rv_iou_1_val = metric_list[0, 2]
                myo_iou_1_val = metric_list[1, 2]
                lv_iou_1_val = metric_list[2, 2]
                metric_list = 0.0
                for _, sampled_batch in enumerate(testloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1, classes=args.num_classes, input_mode=args.input_mode)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                performance1_test = np.mean(metric_list, axis=0)[0]       # mDice test
                performance1_test_iou = np.mean(metric_list, axis=0)[2]   # mIoU test (new)
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
                    best_ckpt = run_root / "checkpoints" / "best.ckpt"
                    save_full_ckpt(best_ckpt, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
                    logging.info("best_checkpoint_updated path=%s epoch=%s global_step=%s metric=%f", best_ckpt, epoch, iter_num, performance1)
                logging.info("iteration %d : model1_val_mDice : %.2f%%", iter_num, metric_to_percent(performance1))
                logging.info("iteration %d : model1_val_mIoU : %.2f%%", iter_num, metric_to_percent(performance1_iou))
                logging.info("iteration %d : model1_test_mDice : %.2f%%", iter_num, metric_to_percent(performance1_test))
                logging.info("iteration %d : model1_test_mIoU : %.2f%%", iter_num, metric_to_percent(performance1_test_iou))
                logging.info("iteration %d : model1_val_IoU_pct RV: %.2f%%  MYO: %.2f%%  LV: %.2f%%", iter_num, metric_to_percent(rv_iou_1_val), metric_to_percent(myo_iou_1_val), metric_to_percent(lv_iou_1_val))
                # TensorBoard
                writer.add_scalar("model1/val_mean_dice", performance1, iter_num)
                writer.add_scalar("model1/val_mean_iou", performance1_iou, iter_num)
                writer.add_scalar("model1/test_mean_dice", performance1_test, iter_num)
                writer.add_scalar("model1/test_mean_iou", performance1_test_iou, iter_num)
                writer.add_scalar("model1/val_rv_iou", rv_iou_1_val, iter_num)
                writer.add_scalar("model1/val_myo_iou", myo_iou_1_val, iter_num)
                writer.add_scalar("model1/val_lv_iou", lv_iou_1_val, iter_num)
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_2, classes=args.num_classes, input_mode=args.input_mode)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance2 = np.mean(metric_list, axis=0)[0]       # mDice (unchanged)
                performance2_iou = np.mean(metric_list, axis=0)[2]   # mIoU (new)
                rv_iou_2_val = metric_list[0, 2]
                myo_iou_2_val = metric_list[1, 2]
                lv_iou_2_val = metric_list[2, 2]
                metric_list = 0.0
                for _, sampled_batch in enumerate(testloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_2, classes=args.num_classes, input_mode=args.input_mode)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                performance2_test = np.mean(metric_list, axis=0)[0]       # mDice test
                performance2_test_iou = np.mean(metric_list, axis=0)[2]   # mIoU test (new)
                if performance2 > best_performance2:
                    best_performance2 = performance2
                logging.info("iteration %d : model2_val_mDice : %.2f%%", iter_num, metric_to_percent(performance2))
                logging.info("iteration %d : model2_val_mIoU : %.2f%%", iter_num, metric_to_percent(performance2_iou))
                logging.info("iteration %d : model2_test_mDice : %.2f%%", iter_num, metric_to_percent(performance2_test))
                logging.info("iteration %d : model2_test_mIoU : %.2f%%", iter_num, metric_to_percent(performance2_test_iou))
                logging.info("iteration %d : model2_val_IoU_pct RV: %.2f%%  MYO: %.2f%%  LV: %.2f%%", iter_num, metric_to_percent(rv_iou_2_val), metric_to_percent(myo_iou_2_val), metric_to_percent(lv_iou_2_val))
                # TensorBoard
                writer.add_scalar("model2/val_mean_dice", performance2, iter_num)
                writer.add_scalar("model2/val_mean_iou", performance2_iou, iter_num)
                writer.add_scalar("model2/test_mean_dice", performance2_test, iter_num)
                writer.add_scalar("model2/test_mean_iou", performance2_test_iou, iter_num)
                writer.add_scalar("model2/val_rv_iou", rv_iou_2_val, iter_num)
                writer.add_scalar("model2/val_myo_iou", myo_iou_2_val, iter_num)
                writer.add_scalar("model2/val_lv_iou", lv_iou_2_val, iter_num)

                # Weight/gradient audit for 2.5D (every 200 iters)
                if args.input_mode == "25d_3ch":
                    def _get_conv1_weights(m):
                        for n, p in m.named_parameters():
                            if p.ndim == 4 and p.shape[1] == 3 and 'weight' in n:
                                return n, p
                        return None, None
                    m1_name, m1_w = _get_conv1_weights(model_1)
                    m2_name, m2_w = _get_conv1_weights(model_2)
                    if m1_w is not None:
                        pn = m1_w[:, 0, :, :].norm().item()
                        cn = m1_w[:, 1, :, :].norm().item()
                        nn = m1_w[:, 2, :, :].norm().item()
                        pg = m1_w.grad[:, 0, :, :].norm().item() if m1_w.grad is not None else 0
                        cg = m1_w.grad[:, 1, :, :].norm().item() if m1_w.grad is not None else 0
                        ng = m1_w.grad[:, 2, :, :].norm().item() if m1_w.grad is not None else 0
                        logging.info("iteration %d : weight_norms model1 %s prev=%.6f center=%.6f next=%.6f", iter_num, m1_name, pn, cn, nn)
                        logging.info("iteration %d : grad_norms model1 %s prev_grad=%.6f center_grad=%.6f next_grad=%.6f", iter_num, m1_name, pg, cg, ng)
                    if m2_w is not None:
                        pn2 = m2_w[:, 0, :, :].norm().item()
                        cn2 = m2_w[:, 1, :, :].norm().item()
                        nn2 = m2_w[:, 2, :, :].norm().item()
                        pg2 = m2_w.grad[:, 0, :, :].norm().item() if m2_w.grad is not None else 0
                        cg2 = m2_w.grad[:, 1, :, :].norm().item() if m2_w.grad is not None else 0
                        ng2 = m2_w.grad[:, 2, :, :].norm().item() if m2_w.grad is not None else 0
                        logging.info("iteration %d : weight_norms model2 %s prev=%.6f center=%.6f next=%.6f", iter_num, m2_name, pn2, cn2, nn2)
                        logging.info("iteration %d : grad_norms model2 %s prev_grad=%.6f center_grad=%.6f next_grad=%.6f", iter_num, m2_name, pg2, cg2, ng2)

                model_2.train()

                # Store IoU for dynamics CSV
                m1_val_miou, m1_test_miou = performance1_iou, performance1_test_iou
                m2_val_miou, m2_test_miou = performance2_iou, performance2_test_iou
                m1_rv_iou, m1_myo_iou, m1_lv_iou = rv_iou_1_val, myo_iou_1_val, lv_iou_1_val
                m2_rv_iou, m2_myo_iou, m2_lv_iou = rv_iou_2_val, myo_iou_2_val, lv_iou_2_val
                # Overwrite dynamics CSV row with IoU values for this eval iteration
                mark_last_dynamics_checkpoint_saved(dynamics_csv, iter_num)

            append_dynamics(dynamics_csv, [
                iter_num, iter_num, float(loss.item()), float(loss_1.item()), float(loss_2.item()), float(consistency_weight),
                repair_triggered, fft_triggered, pred_fg_ratio_1, pred_fg_ratio_2, pseudo_fg_ratio_1, pseudo_fg_ratio_2,
                branch_disagreement_ratio, branch_agreement_ratio, fft_triggered, 30, int(loss_finite), int(grad_finite), int(logits_finite), 0,
                m1_val_miou, m1_test_miou, m2_val_miou, m2_test_miou,
                m1_rv_iou, m1_myo_iou, m1_lv_iou,
                m2_rv_iou, m2_myo_iou, m2_lv_iou,
                ipcrm_stats["ipcrm_enabled"], ipcrm_stats["R_contour_ratio"],
                ipcrm_stats["P_myo_core_ratio"], ipcrm_stats["R_safe_ratio"],
                ipcrm_stats["ipcrm_weight_mean"], ipcrm_stats["ipcrm_weight_min"],
                ipcrm_stats["ipcrm_weight_max"], ipcrm_stats["selected_fg_ratio_before_ipcrm"],
                ipcrm_stats["selected_fg_ratio_after_ipcrm"], ipcrm_stats["effective_unsup_mask_ratio"],
                ipcrm_stats["MYO_core_protected_ratio"], ipcrm_stats["LV_MYO_transition_ratio"],
                ipcrm_stats["loss_u_after_ipcrm"], ipcrm_stats["ipcrm_finite_check"],
                ip_rdc_stats["ip_rdc_enabled"], ip_rdc_stats["ip_rdc_weight"],
                ip_rdc_stats["ip_rdc_loss"], ip_rdc_stats["risk_gate_mean"],
                ip_rdc_stats["risk_gate_fg_mean"], ip_rdc_stats["risk_gate_bg_mean"],
                ip_rdc_stats["risk_gate_boundary_mean"], ip_rdc_stats["interior_preserved_ratio"],
                ip_rdc_stats["affected_fg_ratio"], ip_rdc_stats["affected_boundary_ratio"],
                ip_rdc_stats["center_neighbor_disagreement_mean"], ip_rdc_stats["ip_rdc_finite_check"],
            ])

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    final_ckpt = run_root / "checkpoints" / "last.ckpt"
    best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
    save_full_ckpt(final_ckpt, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
    mark_last_dynamics_checkpoint_saved(dynamics_csv, iter_num)
    if not (run_root / "checkpoints" / "best.ckpt").exists():
        save_full_ckpt(run_root / "checkpoints" / "best.ckpt", model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
    append_dynamics(dynamics_csv, [iter_num, iter_num, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, "", "", "", "", "", "", "", "", "", "", 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, float(args.ip_rdc_weight), 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])

    re_model_1 = BCP_net(in_chns=in_chns, class_num=args.num_classes, ema=False, use_neighbor_relation=use_relation, relation_dim=args.relation_dim)
    re_model_2 = BCP_net(in_chns=in_chns, class_num=args.num_classes, ema=False, use_neighbor_relation=use_relation, relation_dim=args.relation_dim)
    re_ema = BCP_net(in_chns=in_chns, class_num=args.num_classes, ema=True, use_neighbor_relation=use_relation, relation_dim=args.relation_dim)
    re_opt1 = optim.SGD(re_model_1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    re_opt2 = optim.SGD(re_model_2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    re_ckpt = load_full_ckpt(final_ckpt, re_model_1, re_model_2, re_ema, re_opt1, re_opt2)
    finite_check(re_model_1, re_opt1)
    finite_check(re_model_2, re_opt2)
    finite_check(re_ema, re_opt1)
    logging.info("resume_check_ok epoch=%s global_step=%s", re_ckpt.get("epoch"), re_ckpt.get("global_step"))
    writer.close()


if __name__ == "__main__":
    train()
