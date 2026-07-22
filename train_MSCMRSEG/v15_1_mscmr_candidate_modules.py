"""Candidate-only Mask Token routing and AdaMix primitives.

These modules are deliberately external to the frozen MS-CMRSeg model.  The
router is attached only by the candidate trainer, as a forward-pre-hook on the
existing decoder output head; OFF never constructs either module or hook.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class MaskTokenAdapter(nn.Module):
    """Route the feature immediately before ``decoder.out_conv`` via tokens."""

    def __init__(self, channels: int = 16, token_num: int = 4):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(token_num, channels))
        nn.init.normal_(self.tokens, std=0.02)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.tokens.shape[1]:
            raise ValueError(f"token feature mismatch: {tuple(feature.shape)} vs {tuple(self.tokens.shape)}")
        descriptor = feature.mean(dim=(-2, -1))
        affinity = torch.softmax(descriptor @ self.tokens.t() / math.sqrt(feature.shape[1]), dim=1)
        routed = affinity @ self.tokens
        # This is a residual, channel-wise feature routing operation.  It has
        # no new convolutional parameters and preserves the feature shape.
        return feature * (1.0 + torch.tanh(routed).unsqueeze(-1).unsqueeze(-1))


class DecoderHeadRouter:
    """Own exactly one removable pre-hook on a model's existing output head."""

    def __init__(self, model: nn.Module, adapter: MaskTokenAdapter):
        self.model, self.adapter, self.handle = model, adapter, None

    def attach(self) -> None:
        if self.handle is not None:
            raise RuntimeError("candidate token hook already attached")
        head = self.model.decoder.out_conv

        def route(_module, inputs):
            return (self.adapter(inputs[0]),)

        self.handle = head.register_forward_pre_hook(route)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove(); self.handle = None


@dataclass
class AdaMixResult:
    image: torch.Tensor
    first_target: torch.Tensor
    second_target: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor
    permutation: list[int]


def adamix_pair(first: torch.Tensor, second: torch.Tensor, first_target: torch.Tensor,
                second_target: torch.Tensor, first_aux_target: torch.Tensor,
                second_aux_target: torch.Tensor, first_confidence: torch.Tensor,
                second_confidence: torch.Tensor) -> AdaMixResult:
    """Patch-copy the paired BCP branches, including both targets/confidence.

    The random mask is intentionally made only in the ON path by the caller.
    Its recipient/donor mapping is the fixed paired branch permutation [1, 0].
    """
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("AdaMix requires matched BCHW inputs")
    b, _, h, w = first.shape
    if any(x.shape != (b, h, w) for x in (first_target, second_target, first_aux_target,
                                           second_aux_target, first_confidence, second_confidence)):
        raise ValueError("AdaMix targets/confidence must be BHW and pair-aligned")
    mask = torch.zeros((b, 1, h, w), dtype=first.dtype, device=first.device)
    patch_h, patch_w = max(1, h // 4), max(1, w // 4)
    top = int(torch.randint(0, h - patch_h + 1, (1,), device=first.device).item())
    left = int(torch.randint(0, w - patch_w + 1, (1,), device=first.device).item())
    mask[:, :, top:top + patch_h, left:left + patch_w] = 1
    choose = mask[:, 0].bool()
    return AdaMixResult(
        image=first * (1.0 - mask) + second * mask,
        first_target=torch.where(choose, second_target, first_target),
        second_target=torch.where(choose, second_aux_target, first_aux_target),
        confidence=torch.where(choose, second_confidence, first_confidence),
        mask=mask,
        permutation=[1, 0],
    )
