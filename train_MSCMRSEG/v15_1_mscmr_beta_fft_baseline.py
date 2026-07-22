#!/usr/bin/env python3
"""Independent V15.1 original 2D beta-FFT baseline (all innovations absent)."""

from __future__ import annotations

import argparse, json, logging, math, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from train_MSCMRSEG.v15_1_mscmr_common import (
    checkpoint_identity, fft_low_frequency_swap, load_checkpoint, make_model,
    make_poly_scheduler, mixed_supervised_loss, rectangular_mask, restore_rng_state,
    preserve_module_modes, rng_state, save_checkpoint, seed_everything, soft_dice_loss,
    supervised_loss, update_ema,
)
from train_MSCMRSEG.v15_1_mscmr_dataset import MSCMRSliceDataset, PairedInfiniteBatchSampler
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

SOURCES = [__file__, "train_MSCMRSEG/v15_1_mscmr_common.py", "train_MSCMRSEG/v15_1_mscmr_dataset.py",
           "train_MSCMRSEG/v15_1_mscmr_eval.py", "train_MSCMRSEG/v15_1_mscmr_protocol.py",
           "train_MSCMRSEG/networks/unet.py"]


def parser():
    p = argparse.ArgumentParser(description="V15.1 MS-CMRSeg original 2D beta-FFT baseline")
    p.add_argument("--data_root", required=True); p.add_argument("--labeled_manifest", required=True)
    p.add_argument("--unlabeled_manifest", required=True); p.add_argument("--validation_manifest", required=True)
    p.add_argument("--validation_role", default="validation"); p.add_argument("--label_ratio", choices=("5pct", "10pct"), required=True)
    p.add_argument("--split", default="bayeseg_fixed_split"); p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gpu", default="0"); p.add_argument("--max_iterations", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=8); p.add_argument("--labeled_batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, nargs=2, default=(256, 256)); p.add_argument("--base_lr", type=float, default=0.01)
    p.add_argument("--u_weight", type=float, default=0.5); p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--consistency", type=float, default=0.1); p.add_argument("--consistency_rampup", type=int, default=30000)
    p.add_argument("--fft_radius", type=int, default=30); p.add_argument("--validation_interval", type=int, default=200)
    p.add_argument("--checkpoint_interval", type=int, default=1000); p.add_argument("--pretrain_checkpoint", required=True)
    p.add_argument("--run_root", required=True); p.add_argument("--resume_checkpoint"); p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true"); p.add_argument("--dry_run", action="store_true")
    return p


def consistency_weight(iteration, maximum, coefficient):
    progress = min(1.0, iteration / max(1, maximum))
    return coefficient * math.exp(-5.0 * (1.0 - progress) ** 2)


def assert_training_roles(model1, model2, teacher, stage):
    states = (model1.training, model2.training, teacher.training)
    if states != (True, True, False):
        raise RuntimeError(
            f"MODE_LIFECYCLE_REGRESSION stage={stage} "
            f"model1_training={states[0]} model2_training={states[1]} teacher_training={states[2]}"
        )


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size - args.labeled_batch_size < 2 or args.labeled_batch_size < 2:
        raise ValueError("both streams need at least two samples")
    if args.labeled_batch_size % 2 or (args.batch_size - args.labeled_batch_size) % 2:
        raise ValueError("stream batch sizes must be even")
    run_root = Path(args.run_root); (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_root / "console.log")])
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    seed_everything(args.seed); device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    labeled = MSCMRSliceDataset(args.data_root, args.labeled_manifest, args.image_size, augment=True, include_label=True)
    unlabeled = MSCMRSliceDataset(args.data_root, args.unlabeled_manifest, args.image_size, augment=True, include_label=False)
    sampler = PairedInfiniteBatchSampler(len(labeled), len(unlabeled), args.labeled_batch_size,
        args.batch_size - args.labeled_batch_size, args.seed)
    loader = DataLoader(ConcatDataset([labeled, unlabeled]), batch_sampler=sampler,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model1, model2, teacher = make_model(device), make_model(device), make_model(device)
    pretrain = load_checkpoint(args.pretrain_checkpoint, device)
    for model in (model1, model2, teacher): model.load_state_dict(pretrain["model"], strict=True)
    teacher.requires_grad_(False); teacher.eval()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler1 = make_poly_scheduler(optimizer1, args.max_iterations); scheduler2 = make_poly_scheduler(optimizer2, args.max_iterations)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    identity = checkpoint_identity(args, SOURCES, args.labeled_manifest, args.pretrain_checkpoint)
    identity["unlabeled_split_path"] = args.unlabeled_manifest
    from train_MSCMRSEG.v15_1_mscmr_common import sha256_file
    identity["unlabeled_split_sha256"] = sha256_file(args.unlabeled_manifest)
    iteration, best, best_iteration, best_model_name = 0, -1.0, 0, None
    last_validation = None
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, device)
        model1.load_state_dict(checkpoint["model1"], strict=True); model2.load_state_dict(checkpoint["model2"], strict=True)
        teacher.load_state_dict(checkpoint["ema_teacher"], strict=True)
        optimizer1.load_state_dict(checkpoint["optimizer1"]); optimizer2.load_state_dict(checkpoint["optimizer2"])
        scheduler1.load_state_dict(checkpoint["scheduler"]["model1"]); scheduler2.load_state_dict(checkpoint["scheduler"]["model2"])
        scaler.load_state_dict(checkpoint["amp_scaler"]); iteration = int(checkpoint["global_iteration"])
        best, best_iteration = float(checkpoint["best_validation_metric"]), int(checkpoint["best_iteration"])
        best_model_name = checkpoint["best_model_name"]; last_validation = checkpoint.get("last_validation")
        restore_rng_state(checkpoint["random_state"]); sampler.iteration = iteration; teacher.eval()
    assert_training_roles(model1, model2, teacher, "initialization_or_resume")
    logging.info("START baseline ratio=%s seed=%d labeled_slices=%d unlabeled_slices=%d device=%s iteration=%d "
                 "model1_mode=TRAIN model2_mode=TRAIN teacher_mode=EVAL",
                 args.label_ratio, args.seed, len(labeled), len(unlabeled), device, iteration)
    if args.dry_run:
        batch = next(iter(loader)); logging.info("DRY_RUN image=%s label=%s labeled_flags=%s values=%s",
            tuple(batch["image"].shape), tuple(batch["label"].shape), batch["is_labeled"].tolist(), torch.unique(batch["label"][:args.labeled_batch_size]).tolist())
        return 0

    timeline = (run_root / "training_timeline.jsonl").open("a", buffering=1); iterator = iter(loader)
    while iteration < args.max_iterations:
        assert_training_roles(model1, model2, teacher, f"train_step_{iteration + 1}_entry")
        batch = next(iterator); weak = batch["image"].to(device); strong = batch["image_strong"].to(device)
        labels, labels_strong = batch["label"].to(device), batch["label_strong"].to(device)
        lbs, ubs = args.labeled_batch_size, args.batch_size - args.labeled_batch_size
        lh, uh = lbs // 2, ubs // 2
        la, lb = weak[:lh], weak[lh:lbs]; ua, ub = weak[lbs:lbs + uh], weak[lbs + uh:]
        la_s, lb_s = strong[:lh], strong[lh:lbs]; ua_s, ub_s = strong[lbs:lbs + uh], strong[lbs + uh:]
        ta, tb = labels[:lh], labels[lh:lbs]; ta_s, tb_s = labels_strong[:lh], labels_strong[lh:lbs]
        teacher.eval(); assert not teacher.training
        with torch.no_grad():
            pseudo_a = teacher(ua).argmax(1); pseudo_b = teacher(ub).argmax(1)
            pseudo_a_s = teacher(ua_s).argmax(1); pseudo_b_s = teacher(ub_s).argmax(1)
        teacher_foreground_proxy = float(torch.cat([pseudo_a, pseudo_b, pseudo_a_s, pseudo_b_s]).gt(0).float().mean())
        keep = rectangular_mask(lh, weak.shape[-2], weak.shape[-1], device)
        input_u1 = ua * keep[:, None] + la * (1 - keep[:, None]); input_l1 = lb * keep[:, None] + ub * (1 - keep[:, None])
        input_u2 = ua_s * keep[:, None] + la_s * (1 - keep[:, None]); input_l2 = lb_s * keep[:, None] + ub_s * (1 - keep[:, None])
        optimizer1.zero_grad(set_to_none=True); optimizer2.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            out_u1, out_l1 = model1(input_u1), model1(input_l1); out_u2, out_l2 = model2(input_u2), model2(input_l2)
            lu1, _, _ = mixed_supervised_loss(out_u1, pseudo_a, ta, keep, args.u_weight, 1.0)
            ll1, _, _ = mixed_supervised_loss(out_l1, tb, pseudo_b, keep, 1.0, args.u_weight)
            lu2, _, _ = mixed_supervised_loss(out_u2, pseudo_a_s, ta_s, keep, args.u_weight, 1.0)
            ll2, _, _ = mixed_supervised_loss(out_l2, tb_s, pseudo_b_s, keep, 1.0, args.u_weight)
            logits1, logits2 = torch.cat([out_u1, out_l1]), torch.cat([out_u2, out_l2])
            cross1 = soft_dice_loss(logits1, logits2.detach().argmax(1)); cross2 = soft_dice_loss(logits2, logits1.detach().argmax(1))
            fft_images = fft_low_frequency_swap(torch.cat([input_u1, input_l1]), torch.cat([input_u2, input_l2]), args.fft_radius)
            fft1, fft2 = model1(fft_images), model2(fft_images)
            fft_peer1 = soft_dice_loss(fft1, fft2.detach().argmax(1)); fft_peer2 = soft_dice_loss(fft2, fft1.detach().argmax(1))
            beta = torch.from_numpy(np.random.beta(0.1, 0.1, size=(uh, 1, 1, 1))).float().to(device)
            mixed_unlabeled = ua * (1 - beta) + ub * beta
            licr_logits = model1(torch.cat([weak[:lbs], mixed_unlabeled]))
            supervised_licr, _, _ = supervised_loss(licr_logits[:lbs], labels[:lbs])
            assert not teacher.training
            with torch.no_grad(): teacher_mix = torch.softmax(teacher(ua), 1) * (1 - beta) + torch.softmax(teacher(ub), 1) * beta
            weight = consistency_weight(iteration, args.consistency_rampup, args.consistency)
            licr = F.mse_loss(torch.softmax(licr_logits[lbs:], 1), teacher_mix) * weight
            loss1 = lu1 + ll1 + cross1 + fft_peer1
            loss2 = lu2 + ll2 + cross2 + fft_peer2
            loss = loss1 + loss2 + supervised_licr + licr
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {iteration + 1}")
        scaler.scale(loss).backward(); scaler.step(optimizer1); scaler.step(optimizer2); scaler.update()
        scheduler1.step(); scheduler2.step(); iteration += 1; update_ema(model1, teacher, args.ema_decay)
        assert_training_roles(model1, model2, teacher, f"ema_update_{iteration}_exit")
        if iteration == 1 or iteration % 20 == 0:
            record = {"iteration": iteration, "loss": float(loss.detach()), "model1_loss": float(loss1.detach()),
                      "model2_loss": float(loss2.detach()), "licr": float(licr.detach()), "consistency_weight": weight,
                      "fft_finite": bool(torch.isfinite(fft_images).all()), "ema_finite": all(torch.isfinite(p).all() for p in teacher.parameters()),
                      "teacher_foreground_proxy": teacher_foreground_proxy,
                      "model1_mode": "TRAIN", "model2_mode": "TRAIN", "teacher_mode": "EVAL"}
            timeline.write(json.dumps(record) + "\n"); logging.info(
                "iteration %d loss %.6f model1 %.6f model2 %.6f teacher_foreground_proxy=%.8f "
                "model1_mode=TRAIN model2_mode=TRAIN teacher_mode=EVAL",
                iteration, record["loss"], record["model1_loss"], record["model2_loss"], teacher_foreground_proxy)
        validation = None
        if iteration % args.validation_interval == 0 or iteration == args.max_iterations:
            assert_training_roles(model1, model2, teacher, f"validation_{iteration}_entry")
            with preserve_module_modes(model1, model2, teacher):
                model1.eval(); model2.eval(); teacher.eval()
                assert not model1.training and not model2.training and not teacher.training
                val1 = evaluate_manifest(model1, args.data_root, args.validation_manifest, args.image_size, device, role=args.validation_role)
                val2 = evaluate_manifest(model2, args.data_root, args.validation_manifest, args.image_size, device, role=args.validation_role)
            assert_training_roles(model1, model2, teacher, f"validation_{iteration}_exit")
            selected_name, validation = ("model1", val1) if val1["mDice"] >= val2["mDice"] else ("model2", val2)
            last_validation = {"selected_model": selected_name, "selected": validation, "model1": val1, "model2": val2}
            selected_foreground = float(np.mean([case["predicted_foreground_ratio"] for case in validation["cases"]]))
            logging.info("VALIDATION iteration=%d selected=%s mDice=%.8f model1=%.8f model2=%.8f RV=%.8f MYO=%.8f LV=%.8f "
                "selected_foreground=%.8f teacher_foreground_proxy=%.8f "
                "post_validation_model1_mode=TRAIN post_validation_model2_mode=TRAIN post_validation_teacher_mode=EVAL",
                iteration, selected_name, validation["mDice"], val1["mDice"], val2["mDice"], validation["classes"]["RV"]["dice"],
                validation["classes"]["MYO"]["dice"], validation["classes"]["LV"]["dice"], selected_foreground, teacher_foreground_proxy)
            if validation["mDice"] > best: best, best_iteration, best_model_name = validation["mDice"], iteration, selected_name
        payload = {**identity, "format": "V15_1_BETA_FFT_FULL_CHECKPOINT", "model1": model1.state_dict(),
                   "model2": model2.state_dict(), "ema_teacher": teacher.state_dict(), "optimizer1": optimizer1.state_dict(),
                   "optimizer2": optimizer2.state_dict(), "scheduler": {"model1": scheduler1.state_dict(), "model2": scheduler2.state_dict()},
                   "amp_scaler": scaler.state_dict(), "global_iteration": iteration,
                   "current_epoch": iteration * args.batch_size / (len(labeled) + len(unlabeled)),
                   "best_validation_metric": best, "best_iteration": best_iteration, "best_model_name": best_model_name,
                   "last_validation": last_validation, "random_state": rng_state(), "sampler_state": {"iteration": iteration, "rule": "seed+iteration"},
                   "mechanisms": {"beta_fft": True, "bcp": True, "ema": True, "mask_token": False, "adamix": False, "eems": False, "emr": False}}
        if validation is not None and best_iteration == iteration:
            sha = save_checkpoint(run_root / "checkpoints/best.ckpt", payload); logging.info("BEST_CHECKPOINT iteration=%d model=%s sha256=%s", iteration, best_model_name, sha)
        if iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations:
            save_checkpoint(run_root / f"checkpoints/iter_{iteration:06d}.ckpt", payload); save_checkpoint(run_root / "checkpoints/last.ckpt", payload)
    timeline.close(); logging.info("NATURAL_COMPLETION iteration=%d best=%.8f@%d model=%s", iteration, best, best_iteration, best_model_name)
    return 0


if __name__ == "__main__": raise SystemExit(main())
