"""V14.0 isolated ACDC transfer components.

The two mechanisms below are literal multiclass-safe ports of the frozen
PROMISE12 Mask Token + AdaMix implementation.  Importing this module creates
no tensors and consumes no RNG.  Candidate objects are created only when the
corresponding flag is true.
"""

import types

import numpy as np
import torch
import torch.nn as nn

from networks.net_factory import BCP_net as _parent_bcp_net


def process_image_batches_parent_2d(image_batch_a, image_batch_b, threshold):
    """Reconstructed June-17 ACDC 2D FFT dependency, returning [2N,H,W]."""
    if image_batch_a.shape != image_batch_b.shape or image_batch_a.shape[1] != 1:
        raise ValueError("Frozen ACDC parent FFT expects matched single-channel batches")

    def low_frequency(image):
        _, _, rows, cols = image.shape
        transformed = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
        mask = torch.zeros((rows, cols), dtype=torch.float32, device=image.device)
        row_center, col_center = rows // 2, cols // 2
        mask[row_center - threshold:row_center + threshold,
             col_center - threshold:col_center + threshold] = 1
        return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(transformed * mask, dim=(-2, -1)), dim=(-2, -1)))

    swapped_a, swapped_b = [], []
    for index in range(image_batch_a.shape[0]):
        image_a = image_batch_a[index:index + 1]
        image_b = image_batch_b[index:index + 1]
        low_a, low_b = low_frequency(image_a), low_frequency(image_b)
        swapped_a.append(image_a - low_a + low_b)
        swapped_b.append(image_b - low_b + low_a)
    result = torch.cat([torch.cat(swapped_a, dim=0), torch.cat(swapped_b, dim=0)], dim=0)
    return result.view(-1, result.shape[-2], result.shape[-1])


class SpatialMaskTokenRouter(nn.Module):
    """Frozen PROMISE12 decoder pre-head residual token router."""

    def __init__(self, in_channels, token_num=4, token_dim=16):
        super().__init__()
        self.in_channels = in_channels
        self.token_num = token_num
        self.token_dim = token_dim
        self.project = nn.Conv2d(in_channels, token_dim, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(token_dim)
        self.tokens = nn.Parameter(torch.empty(token_num, token_dim))
        self.inverse_project = nn.Conv2d(token_dim, in_channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = token_dim ** -0.5
        self.last_stats = {}
        nn.init.normal_(self.tokens, std=0.02)

    def forward(self, x):
        proj = self.project(x)
        proj = self.norm(proj.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        logits = torch.einsum("bdhw,nd->bnhw", proj, self.tokens) * self.scale
        weights = torch.softmax(logits, dim=1)
        routed = torch.einsum("bnhw,nd->bdhw", weights, self.tokens)
        residual = self.inverse_project(routed)
        out = x + self.gamma * residual
        with torch.no_grad():
            entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1).mean()
            max_weight = weights.max(dim=1).values
            self.last_stats = {
                "token_utilization": weights.mean(dim=(0, 2, 3)).detach().cpu().tolist(),
                "routing_entropy": float(entropy.detach().cpu()),
                "activation_coverage": float((max_weight > (1.0 / self.token_num)).float().mean().detach().cpu()),
                "gamma": float(self.gamma.detach().cpu()),
                "token_norm": float(self.tokens.norm().detach().cpu()),
                "feature_norm": float(x.norm().detach().cpu()),
                "residual_norm": float(residual.norm().detach().cpu()),
                "residual_to_feature_ratio": float((residual.norm() / x.norm().clamp_min(1e-8)).detach().cpu()),
                "finite": bool(torch.isfinite(out).all().item()),
            }
        return out


def _install_router(model, token_num, token_dim):
    decoder = model.decoder
    decoder.mask_token_router = SpatialMaskTokenRouter(16, token_num, token_dim).cuda()

    def candidate_forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_f = self.up4(x, x0)
        x_f = self.mask_token_router(x_f)
        output = self.out_conv(x_f)
        with torch.no_grad():
            pred = output.argmax(dim=1)
            # The router is class-shared; per-class activation is summarized
            # by output-class spatial prevalence without changing its formula.
            self.mask_token_router.last_stats["predicted_class_fraction"] = [
                float((pred == c).float().mean().detach().cpu()) for c in range(output.shape[1])
            ]
        return output, x_f

    decoder.forward = types.MethodType(candidate_forward, decoder)
    return model


def v14_bcp_net(in_chns=1, class_num=4, ema=False, mask_token_enabled=False,
                 mask_token_num=4, mask_token_dim=16):
    if not mask_token_enabled:
        return _parent_bcp_net(in_chns=in_chns, class_num=class_num, ema=ema)
    model = _parent_bcp_net(in_chns=in_chns, class_num=class_num, ema=False)
    model = _install_router(model, mask_token_num, mask_token_dim)
    if ema:
        for parameter in model.parameters():
            parameter.detach_()
    return model


def generate_adamix_bcp_lite_v1_mask(img, logits_a, logits_b, min_side=0.18, max_side=0.38):
    """Frozen PROMISE12 AdaMix sampling/mixing rule; class-count independent."""
    batch_size, _, img_x, img_y = img.shape
    min_side = max(0.05, min(0.75, float(min_side)))
    max_side = max(min_side, min(0.90, float(max_side)))
    with torch.no_grad():
        conf_a = torch.softmax(logits_a.detach(), dim=1).max(dim=1)[0].mean()
        conf_b = torch.softmax(logits_b.detach(), dim=1).max(dim=1)[0].mean()
        teacher_conf = torch.clamp(0.5 * (conf_a + conf_b), 0.0, 1.0)
        side_fraction = min_side + (max_side - min_side) * teacher_conf.item()
    patch_x = max(1, min(img_x - 1, int(round(img_x * side_fraction))))
    patch_y = max(1, min(img_y - 1, int(round(img_y * side_fraction))))
    w = np.random.randint(0, max(1, img_x - patch_x))
    h = np.random.randint(0, max(1, img_y - patch_y))
    mask = torch.ones(img_x, img_y, device=img.device)
    loss_mask = torch.ones(batch_size, img_x, img_y, device=img.device)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    with torch.no_grad():
        pseudo = logits_a.argmax(dim=1)
        patch_labels = pseudo[:, w:w + patch_x, h:h + patch_y]
        class_histogram = [float((patch_labels == c).float().mean().cpu()) for c in range(logits_a.shape[1])]
    stats = {
        "teacher_conf": float(teacher_conf.detach().cpu()),
        "side_fraction": float(side_fraction),
        "patch_area_ratio": float((patch_x * patch_y) / float(img_x * img_y)),
        "class_histogram": class_histogram,
        "foreground_coverage": float(1.0 - class_histogram[0]),
        "finite": bool(torch.isfinite(teacher_conf).item() and np.isfinite(side_fraction)),
    }
    # ACDC parent multiplies masks after an explicit channel unsqueeze and
    # therefore expects [B,H,W]; PROMISE used [H,W] broadcasting directly.
    # Expanding the same sampled rectangle is an interface-only adaptation.
    batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return batch_mask.long(), loss_mask.long(), stats


def load_parent_pretrain(model, state, candidate_enabled):
    if not candidate_enabled:
        model.load_state_dict(state)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if not key.startswith("decoder.mask_token_router.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"Mask Token pretrain mismatch: missing={bad_missing}, unexpected={unexpected}")


def transplant_parent_optimizer_state(optimizer, parent_state, candidate_enabled):
    if not candidate_enabled:
        optimizer.load_state_dict(parent_state)
        return
    # Candidate optimizer has base parameters in group 0 and router parameters
    # in group 1. Load the historical base state into a temporary one-group
    # optimizer, then transplant state/hyperparameters without dropping router.
    base_params = optimizer.param_groups[0]["params"]
    temporary = torch.optim.SGD(base_params, lr=optimizer.defaults["lr"],
                                momentum=optimizer.defaults["momentum"],
                                weight_decay=optimizer.defaults["weight_decay"])
    temporary.load_state_dict(parent_state)
    optimizer.state.update(temporary.state)
    for key, value in temporary.param_groups[0].items():
        if key != "params":
            optimizer.param_groups[0][key] = value


def make_optimizer(model, lr, mask_token_enabled):
    if not mask_token_enabled:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    base, router = [], []
    for name, parameter in model.named_parameters():
        (router if name.startswith("decoder.mask_token_router.") else base).append(parameter)
    return torch.optim.SGD([{"params": base}, {"params": router}], lr=lr, momentum=0.9, weight_decay=1e-4)


def mask_token_endpoint(model):
    router = getattr(getattr(model, "decoder", None), "mask_token_router", None)
    if router is None:
        return None
    stats = dict(router.last_stats)
    grad = router.tokens.grad
    stats["token_gradient_norm"] = float(grad.norm().detach().cpu()) if grad is not None else 0.0
    stats["gradient_finite"] = bool(grad is not None and torch.isfinite(grad).all().item())
    return stats
