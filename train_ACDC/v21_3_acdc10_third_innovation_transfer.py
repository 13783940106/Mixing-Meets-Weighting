"""V21.3 ACDC10 third-innovation overlay.

This entry starts from the frozen V14 ACDC Mask Token + AdaMix overlay and adds
one default-off Innovation X hook.  With ``--enable_v21_3_innovation_x false``
the effective source remains scientifically equivalent to the canonical
ACDC10 beta-FFT + Mask Token + AdaMix parent.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V14 = ROOT / "train_ACDC" / "v14_acdc_mask_adamix_transfer.py"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"V21.3 overlay anchor {label!r} count={count}, expected=1")
    return source.replace(old, new, 1)


def build_effective_source() -> str:
    namespace = {"__file__": str(V14), "__name__": "v14_overlay_build"}
    exec(compile(V14.read_text(), str(V14), "exec"), namespace)
    source = namespace["V14_EFFECTIVE_SOURCE"]
    source = replace_once(
        source,
        "from v14_mask_adamix_components import (v14_bcp_net as BCP_net, generate_adamix_bcp_lite_v1_mask, load_parent_pretrain, transplant_parent_optimizer_state, make_optimizer, mask_token_endpoint, process_image_batches_parent_2d)\n",
        "from v14_mask_adamix_components import (v14_bcp_net as BCP_net, generate_adamix_bcp_lite_v1_mask, load_parent_pretrain, transplant_parent_optimizer_state, make_optimizer, mask_token_endpoint, process_image_batches_parent_2d)\n"
        "from v21_3_acdc10_third_innovation_components import compute_v21_3_innovation_x\n",
        "V21.3 component import",
    )
    source = replace_once(
        source,
        'parser.add_argument("--mechanism_log_interval", type=int, default=200)\n',
        'parser.add_argument("--mechanism_log_interval", type=int, default=200)\n'
        'parser.add_argument("--innovation_x_id", type=str, default="")\n'
        'parser.add_argument("--enable_v21_3_innovation_x", type=str2bool, default=False)\n'
        'parser.add_argument("--v21_3_innovation_x_weight", type=float, default=0.05)\n',
        "V21.3 parser flags",
    )
    source = replace_once(
        source,
        '            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision5\n'
        '            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision6\n'
        '            loss = loss_1 + loss_2 + consistency_loss + supervised_loss_mix + consistency_loss_s + supervised_loss_mix_s\n',
        '            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision5\n'
        '            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision6\n'
        '            innovation_x_loss = loss_1 * 0.0\n'
        '            innovation_x_endpoint = None\n'
        '            if args.enable_v21_3_innovation_x:\n'
        '                innovation_x_loss, innovation_x_endpoint = compute_v21_3_innovation_x(\n'
        '                    args.innovation_x_id, out_1, out_2, pre_a, pre_b,\n'
        '                    image_output_1_FFT, image_output_2_FFT, iter_num, args.max_iterations)\n'
        '            loss = loss_1 + loss_2 + consistency_loss + supervised_loss_mix + consistency_loss_s + supervised_loss_mix_s + args.v21_3_innovation_x_weight * innovation_x_loss\n',
        "V21.3 loss hook",
    )
    source = source.replace("V14_EFFECTIVE_SOURCE", "V21_3_EFFECTIVE_SOURCE")
    source = replace_once(
        source,
        '            if args.adamix_enabled and iter_num % args.mechanism_log_interval == 0:\n'
        '                adamix_stats["mixed_loss_contribution"] = float((loss_dice_1 + loss_ce_1 + loss_dice_2 + loss_ce_2).detach().cpu())\n'
        '                adamix_stats["model1_total_gradient_norm"] = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model_1.parameters() if p.grad is not None)).cpu())\n'
        '                logging.info("iteration %d : adamix_endpoint=%s", iter_num, json.dumps(adamix_stats, sort_keys=True))\n',
        '            if args.adamix_enabled and iter_num % args.mechanism_log_interval == 0:\n'
        '                adamix_stats["mixed_loss_contribution"] = float((loss_dice_1 + loss_ce_1 + loss_dice_2 + loss_ce_2).detach().cpu())\n'
        '                adamix_stats["model1_total_gradient_norm"] = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model_1.parameters() if p.grad is not None)).cpu())\n'
        '                logging.info("iteration %d : adamix_endpoint=%s", iter_num, json.dumps(adamix_stats, sort_keys=True))\n'
        '            if args.enable_v21_3_innovation_x and iter_num % args.mechanism_log_interval == 0:\n'
        '                payload = dict(innovation_x_endpoint or {})\n'
        '                payload["candidate_id"] = args.innovation_x_id\n'
        '                payload["innovation_x_loss"] = float(innovation_x_loss.detach().cpu())\n'
        '                payload["weighted_innovation_x_loss"] = float((args.v21_3_innovation_x_weight * innovation_x_loss).detach().cpu())\n'
        '                payload["finite"] = bool(torch.isfinite(innovation_x_loss).item())\n'
        '                logging.info("iteration %d : v21_3_innovation_x_endpoint=%s", iter_num, json.dumps(payload, sort_keys=True))\n',
        "V21.3 endpoint log",
    )
    return source


V21_3_EFFECTIVE_SOURCE = build_effective_source()

if __name__ == "__main__":
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "V21_3_EFFECTIVE_SOURCE": V21_3_EFFECTIVE_SOURCE,
    }
    exec(compile(V21_3_EFFECTIVE_SOURCE, str(Path(__file__).resolve()), "exec"), namespace)
