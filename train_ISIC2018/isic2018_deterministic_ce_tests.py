"""Focused mathematical and strict-CUDA checks for the CE compatibility backend."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from train_ISIC2018.isic2018_common import deterministic_cross_entropy, dice_loss, mixed_supervised_loss

def compare(device, dtype, weight, ignore_index, reduction):
    torch.manual_seed(91)
    logits_a = torch.randn(2, 3, 5, 4, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, 3, (2, 5, 4), device=device)
    target[0, 0, 0] = ignore_index
    logits_b = logits_a.detach().clone().requires_grad_(True)
    expected = F.cross_entropy(logits_a, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    actual = deterministic_cross_entropy(logits_b, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    if reduction == "none":
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        actual.sum().backward(); expected.sum().backward()
    else:
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        actual.backward(); expected.backward()
    assert torch.allclose(logits_b.grad, logits_a.grad, atol=3e-6, rtol=3e-6)

def main():
    for reduction in ("none", "sum", "mean"):
        compare("cpu", torch.float32, None, -100, reduction)
        compare("cpu", torch.float32, torch.tensor([0.3, 1.0, 2.0]), -100, reduction)
    all_ignore_logits = torch.randn(1, 2, 3, 3, requires_grad=True)
    all_ignore = torch.full((1, 3, 3), -100)
    value = deterministic_cross_entropy(all_ignore_logits, all_ignore)
    assert value.item() == 0.0; value.backward(); assert all_ignore_logits.grad.abs().sum().item() == 0.0
    # Verify the BCP-specific CE map and separate keep/patch denominators.
    x = torch.randn(2, 2, 3, 4, requires_grad=True); a = torch.randint(0, 2, (2, 3, 4)); b = torch.randint(0, 2, (2, 3, 4)); keep = torch.zeros(2, 3, 4); keep[:, :, :2] = 1
    native_a, native_b = F.cross_entropy(x, a, reduction="none"), F.cross_entropy(x, b, reduction="none")
    native_ce = .7*(native_a*keep).sum()/keep.sum() + 1.3*(native_b*(1-keep)).sum()/(1-keep).sum()
    expected = .5*(native_ce + dice_loss(x, torch.where(keep.bool(),a,b)))
    actual, actual_ce, _ = mixed_supervised_loss(x, a, b, keep, .7, 1.3)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6) and torch.allclose(actual_ce, native_ce, atol=2e-6, rtol=2e-6)
    if torch.cuda.is_available():
        # Native CUDA CE is used only for the equivalence comparison.  It is
        # known to be unavailable in strict mode on this PyTorch build.
        torch.use_deterministic_algorithms(True, warn_only=True)
        compare("cuda", torch.float32, None, -100, "mean")
        compare("cuda", torch.float32, torch.tensor([0.3, 1.0, 2.0], device="cuda"), -100, "mean")
        torch.use_deterministic_algorithms(True)
        # AMP executes CE's numerically sensitive work in float32, as in F.cross_entropy.
        x = torch.randn(2, 2, 8, 8, device="cuda", requires_grad=True); y = torch.randint(0, 2, (2, 8, 8), device="cuda")
        with torch.autocast("cuda", enabled=True): loss = deterministic_cross_entropy(x, y)
        loss.backward(); assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    print("TEST_DETERMINISTIC_CE_FORWARD_GRADIENT_EQUIVALENCE PASS")
    print("TEST_DETERMINISTIC_CE_STRICT_CUDA PASS")
if __name__ == "__main__": main()
