"""Isolated V14.0 overlay on the frozen ACDC seed-42 parent source.

The effective trainer is constructed from the SHA-pinned parent source using
asserted textual patches.  This keeps the parent immutable and makes every
candidate change explicit in this file.  With both flags false, execution
uses the exact parent model, mask, losses, RNG draws and optimizer layout.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "train_ACDC" / "beta_fft_acdc_repro.py.bak_clean_split_minimal"
EXPECTED_PARENT_SHA256 = "16aa2c9f4030e17e0300c33731938d8bc3f2cb76a3e7ff46e1e407a42cc0dbe7"


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"V14 overlay anchor {label!r} count={count}, expected=1")
    return source.replace(old, new, 1)


def build_effective_source():
    import hashlib

    source = PARENT.read_text()
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"Frozen parent SHA mismatch: {actual}")

    source = replace_once(
        source,
        "from networks.net_factory import BCP_net\n",
        "from v14_mask_adamix_components import (v14_bcp_net as BCP_net, generate_adamix_bcp_lite_v1_mask, load_parent_pretrain, transplant_parent_optimizer_state, make_optimizer, mask_token_endpoint, process_image_batches_parent_2d)\n",
        "isolated component import",
    )
    source = replace_once(
        source,
        "from utils.displacement import process_image_batches\n",
        "process_image_batches = process_image_batches_parent_2d\n",
        "frozen 2D FFT dependency",
    )
    source = replace_once(
        source,
        'parser.add_argument("--save_interval", type=int, default=200)\n',
        'parser.add_argument("--save_interval", type=int, default=200)\n'
        'parser.add_argument("--mask_token_enabled", type=str2bool, default=False)\n'
        'parser.add_argument("--adamix_enabled", type=str2bool, default=False)\n'
        'parser.add_argument("--mask_token_num", type=int, default=4)\n'
        'parser.add_argument("--mask_token_dim", type=int, default=16)\n'
        'parser.add_argument("--adamix_min_side", type=float, default=0.18)\n'
        'parser.add_argument("--adamix_max_side", type=float, default=0.38)\n'
        'parser.add_argument("--mechanism_log_interval", type=int, default=200)\n',
        "two default-off flags and frozen defaults",
    )
    source = replace_once(
        source,
        '    shutil.copy(__file__, run_root / "source_snapshot.py")\n',
        '    shutil.copy(__file__, run_root / "source_snapshot.py")\n'
        '    (run_root / "effective_source.py").write_text(V14_EFFECTIVE_SOURCE)\n',
        "effective source snapshot",
    )
    source = source.replace(
        'BCP_net(in_chns=1, class_num=args.num_classes, ema=False)',
        'BCP_net(in_chns=1, class_num=args.num_classes, ema=False, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)',
    )
    source = source.replace(
        'BCP_net(in_chns=1, class_num=args.num_classes, ema=True)',
        'BCP_net(in_chns=1, class_num=args.num_classes, ema=True, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)',
    )
    source = replace_once(
        source,
        '    optimizer1 = optim.SGD(model_1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)\n'
        '    optimizer2 = optim.SGD(model_2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)\n',
        '    optimizer1 = make_optimizer(model_1, args.base_lr, args.mask_token_enabled)\n'
        '    optimizer2 = make_optimizer(model_2, args.base_lr, args.mask_token_enabled)\n',
        "candidate optimizer",
    )
    source = replace_once(
        source,
        '    db_test = BaseDataSets(base_dir=args.root_path, split="test")\n',
        '    # True test access is deferred until validation-best identity is frozen.\n'
        '    db_test = BaseDataSets(base_dir=args.root_path, split="val")\n',
        "validation-only model selection",
    )
    source = replace_once(
        source,
        '        ema_model.load_state_dict(load_state["net"])\n'
        '        model_1.load_state_dict(load_state["net"])\n'
        '        model_2.load_state_dict(load_state["net"])\n'
        '        if "opt" in load_state:\n'
        '            optimizer1.load_state_dict(load_state["opt"])\n'
        '            optimizer2.load_state_dict(load_state["opt"])\n',
        '        load_parent_pretrain(ema_model, load_state["net"], args.mask_token_enabled)\n'
        '        load_parent_pretrain(model_1, load_state["net"], args.mask_token_enabled)\n'
        '        load_parent_pretrain(model_2, load_state["net"], args.mask_token_enabled)\n'
        '        if "opt" in load_state:\n'
        '            transplant_parent_optimizer_state(optimizer1, load_state["opt"], args.mask_token_enabled)\n'
        '            transplant_parent_optimizer_state(optimizer2, load_state["opt"], args.mask_token_enabled)\n',
        "multiclass pretrain load",
    )
    source = replace_once(
        source,
        '                img_mask = torch.ones_like(img_a[:, 0])\n'
        '                img_mask[:, : img_mask.shape[1] // 2, : img_mask.shape[2] // 2] = 0\n'
        '                loss_mask = img_mask.clone()\n',
        '                adamix_stats = None\n'
        '                if args.adamix_enabled:\n'
        '                    img_mask, loss_mask, adamix_stats = generate_adamix_bcp_lite_v1_mask(\n'
        '                        img_a, pre_a, pre_b, args.adamix_min_side, args.adamix_max_side)\n'
        '                else:\n'
        '                    img_mask = torch.ones_like(img_a[:, 0])\n'
        '                    img_mask[:, : img_mask.shape[1] // 2, : img_mask.shape[2] // 2] = 0\n'
        '                    loss_mask = img_mask.clone()\n',
        "AdaMix branch",
    )
    source = replace_once(
        source,
        '            optimizer1.step()\n'
        '            optimizer2.step()\n\n'
        '            iter_num += 1\n',
        '            optimizer1.step()\n'
        '            optimizer2.step()\n\n'
        '            iter_num += 1\n'
        '            if args.mask_token_enabled and iter_num % args.mechanism_log_interval == 0:\n'
        '                logging.info("iteration %d : mask_token_endpoint=%s", iter_num, json.dumps({"model1": mask_token_endpoint(model_1), "model2": mask_token_endpoint(model_2)}, sort_keys=True))\n'
        '            if args.adamix_enabled and iter_num % args.mechanism_log_interval == 0:\n'
        '                adamix_stats["mixed_loss_contribution"] = float((loss_dice_1 + loss_ce_1 + loss_dice_2 + loss_ce_2).detach().cpu())\n'
        '                adamix_stats["model1_total_gradient_norm"] = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model_1.parameters() if p.grad is not None)).cpu())\n'
        '                logging.info("iteration %d : adamix_endpoint=%s", iter_num, json.dumps(adamix_stats, sort_keys=True))\n',
        "mechanism endpoints",
    )
    source = replace_once(
        source,
        '    re_opt1 = optim.SGD(re_model_1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)\n'
        '    re_opt2 = optim.SGD(re_model_2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)\n',
        '    re_opt1 = make_optimizer(re_model_1, args.base_lr, args.mask_token_enabled)\n'
        '    re_opt2 = make_optimizer(re_model_2, args.base_lr, args.mask_token_enabled)\n',
        "strict reload optimizer",
    )
    return source


V14_EFFECTIVE_SOURCE = build_effective_source()

if __name__ == "__main__":
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "V14_EFFECTIVE_SOURCE": V14_EFFECTIVE_SOURCE,
    }
    exec(compile(V14_EFFECTIVE_SOURCE, str(Path(__file__).resolve()), "exec"), namespace)
