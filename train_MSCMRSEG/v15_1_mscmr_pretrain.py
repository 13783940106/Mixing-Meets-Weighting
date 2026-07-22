#!/usr/bin/env python3
"""Dedicated supervised pretraining for one frozen V15.1 label budget."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from train_MSCMRSEG.v15_1_mscmr_common import (
    checkpoint_identity, load_checkpoint, make_model, make_poly_scheduler,
    mixed_supervised_loss, rectangular_mask, restore_rng_state, rng_state,
    save_checkpoint, seed_everything,
)
from train_MSCMRSEG.v15_1_mscmr_dataset import InfiniteBatchSampler, MSCMRSliceDataset
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

SOURCES = [__file__, "train_MSCMRSEG/v15_1_mscmr_common.py",
           "train_MSCMRSEG/v15_1_mscmr_dataset.py", "train_MSCMRSEG/v15_1_mscmr_eval.py",
           "train_MSCMRSEG/v15_1_mscmr_protocol.py", "train_MSCMRSEG/networks/unet.py"]


def parser():
    p = argparse.ArgumentParser(description="V15.1 MS-CMRSeg supervised pretrain")
    p.add_argument("--data_root", required=True); p.add_argument("--labeled_manifest", required=True)
    p.add_argument("--validation_manifest", required=True); p.add_argument("--validation_role", default="validation")
    p.add_argument("--label_ratio", choices=("5pct", "10pct"), required=True)
    p.add_argument("--split", default="bayeseg_fixed_split"); p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gpu", default="0"); p.add_argument("--max_iterations", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4); p.add_argument("--image_size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--base_lr", type=float, default=0.01); p.add_argument("--validation_interval", type=int, default=200)
    p.add_argument("--checkpoint_interval", type=int, default=1000); p.add_argument("--run_root", required=True)
    p.add_argument("--resume_checkpoint"); p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true"); p.add_argument("--dry_run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    run_root = Path(args.run_root); (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_root / "console.log")])
    config = vars(args).copy(); (run_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = MSCMRSliceDataset(args.data_root, args.labeled_manifest, args.image_size, augment=True)
    sampler = InfiniteBatchSampler(len(dataset), args.batch_size, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = make_model(device); optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = make_poly_scheduler(optimizer, args.max_iterations)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    iteration, best, best_iteration = 0, -1.0, 0
    identity = checkpoint_identity(args, SOURCES, args.labeled_manifest)
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, device)
        model.load_state_dict(checkpoint["model"], strict=True); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["amp_scaler"])
        iteration = int(checkpoint["global_iteration"]); best = float(checkpoint["best_validation_metric"])
        best_iteration = int(checkpoint["best_iteration"]); restore_rng_state(checkpoint["random_state"])
        sampler.iteration = iteration
    logging.info("START pretrain ratio=%s seed=%d slices=%d device=%s iteration=%d", args.label_ratio, args.seed, len(dataset), device, iteration)
    if args.dry_run:
        batch = next(iter(loader)); logging.info("DRY_RUN image=%s label=%s values=%s patient=%s",
            tuple(batch["image"].shape), tuple(batch["label"].shape), torch.unique(batch["label"]).tolist(), batch["patient_id"])
        return 0

    timeline = (run_root / "training_timeline.jsonl").open("a", buffering=1)
    iterator = iter(loader)
    while iteration < args.max_iterations:
        batch = next(iterator); image = batch["image"].to(device); target = batch["label"].to(device)
        half = image.shape[0] // 2
        if half < 1:
            raise RuntimeError("supervised mix requires batch_size >= 2")
        first, second = image[:half], image[-half:]; target_first, target_second = target[:half], target[-half:]
        keep = rectangular_mask(half, image.shape[-2], image.shape[-1], device)
        mixed = first * keep[:, None] + second * (1 - keep[:, None])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(mixed); loss, ce, dice = mixed_supervised_loss(logits, target_first, target_second, keep)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {iteration + 1}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step(); iteration += 1
        if iteration == 1 or iteration % 20 == 0:
            record = {"iteration": iteration, "loss": float(loss.detach()), "ce": float(ce.detach()),
                      "dice_loss": float(dice.detach()), "lr": optimizer.param_groups[0]["lr"], "finite": True}
            timeline.write(json.dumps(record) + "\n"); logging.info("iteration %d loss %.6f ce %.6f dice %.6f", iteration, record["loss"], record["ce"], record["dice_loss"])
        validation = None
        if iteration % args.validation_interval == 0 or iteration == args.max_iterations:
            validation = evaluate_manifest(model, args.data_root, args.validation_manifest, args.image_size, device, role=args.validation_role)
            logging.info("VALIDATION iteration=%d mDice=%.8f RV=%.8f MYO=%.8f LV=%.8f", iteration, validation["mDice"],
                         validation["classes"]["RV"]["dice"], validation["classes"]["MYO"]["dice"], validation["classes"]["LV"]["dice"])
            if validation["mDice"] > best:
                best, best_iteration = validation["mDice"], iteration
        payload = {**identity, "format": "V15_1_PRETRAIN_FULL_CHECKPOINT", "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "amp_scaler": scaler.state_dict(),
                   "global_iteration": iteration, "current_epoch": iteration * args.batch_size / len(dataset),
                   "best_validation_metric": best, "best_iteration": best_iteration, "last_validation": validation,
                   "random_state": rng_state(), "sampler_state": {"iteration": iteration, "rule": "seed+iteration"}}
        if validation is not None and best_iteration == iteration:
            sha = save_checkpoint(run_root / "checkpoints/best.ckpt", payload); logging.info("BEST_CHECKPOINT iteration=%d sha256=%s", iteration, sha)
        if iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations:
            save_checkpoint(run_root / f"checkpoints/iter_{iteration:06d}.ckpt", payload)
            save_checkpoint(run_root / "checkpoints/last.ckpt", payload)
    timeline.close(); logging.info("NATURAL_COMPLETION iteration=%d best=%.8f@%d", iteration, best, best_iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
