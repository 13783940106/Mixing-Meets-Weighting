"""Shared, innovation-free V15.1 training and checkpoint primitives."""

from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from train_MSCMRSEG.networks.unet import UNet


@contextmanager
def preserve_module_modes(*modules):
    """Restore every module's exact train/eval flag after a temporary mode change."""
    states = [module.training for module in modules]
    try:
        yield
    finally:
        for module, was_training in zip(modules, states):
            module.train(was_training)


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def source_sha(paths: Iterable[str | Path]) -> Dict[str, str]:
    return {str(Path(path)): sha256_file(path) for path in paths}


def make_model(device, num_classes=4):
    return UNet(in_chns=1, class_num=num_classes).to(device)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes=4):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = (probs.square() + one_hot.square()).sum(dim=(0, 2, 3))
    return (1.0 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def supervised_loss(logits, target):
    ce = F.cross_entropy(logits, target.long())
    dice = soft_dice_loss(logits, target)
    return 0.5 * (ce + dice), ce, dice


def rectangular_mask(batch: int, height: int, width: int, device, fraction=2 / 3):
    patch_h, patch_w = int(height * fraction), int(width * fraction)
    top = int(np.random.randint(0, max(1, height - patch_h)))
    left = int(np.random.randint(0, max(1, width - patch_w)))
    mask = torch.ones((batch, height, width), device=device)
    mask[:, top:top + patch_h, left:left + patch_w] = 0
    return mask


def mixed_supervised_loss(logits, target_a, target_b, keep_mask, weight_a=1.0, weight_b=1.0):
    patch_mask = 1.0 - keep_mask
    ce_map_a = F.cross_entropy(logits, target_a.long(), reduction="none")
    ce_map_b = F.cross_entropy(logits, target_b.long(), reduction="none")
    ce = weight_a * (ce_map_a * keep_mask).sum() / keep_mask.sum().clamp_min(1)
    ce += weight_b * (ce_map_b * patch_mask).sum() / patch_mask.sum().clamp_min(1)
    mixed_target = torch.where(keep_mask.bool(), target_a, target_b)
    dice = soft_dice_loss(logits, mixed_target)
    return 0.5 * (ce + dice), ce, dice


def fft_low_frequency_swap(first: torch.Tensor, second: torch.Tensor, radius: int):
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("FFT inputs must be matched BCHW tensors")
    h, w = first.shape[-2:]
    radius = min(radius, h // 2 - 1, w // 2 - 1)
    mask = torch.zeros((h, w), device=first.device)
    mask[h // 2 - radius:h // 2 + radius, w // 2 - radius:w // 2 + radius] = 1
    def low(x):
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)))
    low_first, low_second = low(first), low(second)
    return torch.cat([first - low_first + low_second, second - low_second + low_first], dim=0)


@torch.no_grad()
def update_ema(source, target, decay: float):
    source_state = source.state_dict()
    for name, value in target.state_dict().items():
        value.copy_(value * decay + source_state[name] * (1.0 - decay))


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_checkpoint(path: str | Path, payload: Dict):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def load_checkpoint(path: str | Path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def make_poly_scheduler(optimizer, max_iterations: int, start_iteration=0, power=0.9):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / max_iterations) ** power,
        last_epoch=start_iteration - 1)
    return scheduler


def checkpoint_identity(args, source_paths, split_path, pretrain_path=None):
    args_dict = vars(args).copy()
    return {
        "args": args_dict,
        "config_sha256": canonical_json_sha(args_dict),
        "source_sha256": source_sha(source_paths),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "pretrain_path": str(pretrain_path) if pretrain_path else None,
        "pretrain_sha256": sha256_file(pretrain_path) if pretrain_path else None,
    }
