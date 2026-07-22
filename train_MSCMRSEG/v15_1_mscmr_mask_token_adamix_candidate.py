#!/usr/bin/env python3
"""Isolated, full-lifecycle MS-CMRSeg Mask Token + AdaMix candidate trainer.

With both candidate flags absent this entrypoint delegates to the frozen
baseline before any candidate module/hook/RNG construction.  ON reuses the
baseline's dataset, losses, BCP, FFT, EMA, scheduler and frozen evaluator.
"""
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

from train_MSCMRSEG import v15_1_mscmr_beta_fft_baseline as baseline
from train_MSCMRSEG.v15_1_mscmr_candidate_modules import DecoderHeadRouter, MaskTokenAdapter, adamix_pair
from train_MSCMRSEG.v15_1_mscmr_common import (
    checkpoint_identity, fft_low_frequency_swap, load_checkpoint, make_model,
    make_poly_scheduler, mixed_supervised_loss, rectangular_mask, restore_rng_state,
    preserve_module_modes, rng_state, save_checkpoint, seed_everything, sha256_file,
    soft_dice_loss, supervised_loss, update_ema,
)
from train_MSCMRSEG.v15_1_mscmr_dataset import MSCMRSliceDataset, PairedInfiniteBatchSampler
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

SOURCES = [__file__, "train_MSCMRSEG/v15_1_mscmr_candidate_modules.py",
           "train_MSCMRSEG/v15_1_mscmr_common.py", "train_MSCMRSEG/v15_1_mscmr_dataset.py",
           "train_MSCMRSEG/v15_1_mscmr_eval.py", "train_MSCMRSEG/v15_1_mscmr_protocol.py",
           "train_MSCMRSEG/networks/unet.py"]


def parser():
    p = baseline.parser()
    p.description = "V15.1 isolated MS-CMRSeg Mask Token + AdaMix candidate"
    p.add_argument("--enable-mask-token", action="store_true")
    p.add_argument("--enable-adamix", action="store_true")
    p.add_argument("--mask-token-num", type=int, default=4)
    return p


def assert_roles(model1, model2, teacher, stage):
    baseline.assert_training_roles(model1, model2, teacher, stage)


def candidate_config(args):
    return {"enable_mask_token": bool(args.enable_mask_token), "enable_adamix": bool(args.enable_adamix),
            "mask_token_num": int(args.mask_token_num), "token_insertion_point": "decoder.out_conv.forward_pre_hook",
            "adamix_order": "after_bcp_before_fft_and_mixed_supervised_loss", "formal_prediction_source": "model1"}


def _make_payload(identity, args, model1, model2, teacher, optimizer1, optimizer2, scheduler1, scheduler2, scaler,
                  iteration, best, best_iteration, last_validation, sampler, token1, token2):
    payload = {**identity, "format": "V15_1_MASK_TOKEN_ADAMIX_CANDIDATE_CHECKPOINT",
        "model1": model1.state_dict(), "model2": model2.state_dict(), "ema_teacher": teacher.state_dict(),
        "optimizer1": optimizer1.state_dict(), "optimizer2": optimizer2.state_dict(),
        "scheduler": {"model1": scheduler1.state_dict(), "model2": scheduler2.state_dict()},
        "amp_scaler": scaler.state_dict(), "global_iteration": iteration,
        "current_epoch": iteration * args.batch_size, "best_validation_metric": best,
        "best_iteration": best_iteration, "best_model_name": "model1", "last_validation": last_validation,
        "random_state": rng_state(), "sampler_state": {"iteration": iteration, "rule": "seed+iteration"},
        "mechanisms": {"beta_fft": True, "bcp": True, "ema": True, "mask_token": bool(token1),
                       "adamix": bool(args.enable_adamix), "eems": False, "emr": False},
        "candidate_metadata": {"innovation_config": candidate_config(args), "candidate_source_sha256": source_hashes(),
                               "formal_prediction_source": "model1"}}
    if token1 is not None:
        payload["mask_token_model1"] = token1.state_dict(); payload["mask_token_model2"] = token2.state_dict()
    return payload


def source_hashes():
    return {str(Path(item)): sha256_file(PROJECT_ROOT / item if not Path(item).is_absolute() else item) for item in SOURCES}


def _resume(checkpoint, args, model1, model2, teacher, optimizer1, optimizer2, scheduler1, scheduler2, scaler, sampler, token1, token2):
    metadata = checkpoint.get("candidate_metadata", {}).get("innovation_config")
    if metadata != candidate_config(args):
        raise RuntimeError(f"candidate resume configuration conflict: checkpoint={metadata} requested={candidate_config(args)}")
    model1.load_state_dict(checkpoint["model1"], strict=True); model2.load_state_dict(checkpoint["model2"], strict=True)
    teacher.load_state_dict(checkpoint["ema_teacher"], strict=True)
    if token1 is not None:
        token1.load_state_dict(checkpoint["mask_token_model1"], strict=True); token2.load_state_dict(checkpoint["mask_token_model2"], strict=True)
    elif "mask_token_model1" in checkpoint:
        raise RuntimeError("cannot resume a token checkpoint with mask token disabled")
    optimizer1.load_state_dict(checkpoint["optimizer1"]); optimizer2.load_state_dict(checkpoint["optimizer2"])
    scheduler1.load_state_dict(checkpoint["scheduler"]["model1"]); scheduler2.load_state_dict(checkpoint["scheduler"]["model2"])
    scaler.load_state_dict(checkpoint["amp_scaler"]); restore_rng_state(checkpoint["random_state"])
    iteration = int(checkpoint["global_iteration"]); sampler.iteration = iteration; teacher.eval()
    return iteration, float(checkpoint["best_validation_metric"]), int(checkpoint["best_iteration"]), checkpoint.get("last_validation")


def main(argv=None):
    args = parser().parse_args(argv)
    # This exact early delegation is the Default-Off contract.
    if not args.enable_mask_token and not args.enable_adamix:
        return baseline.main(argv)
    if args.batch_size - args.labeled_batch_size < 2 or args.labeled_batch_size < 2:
        raise ValueError("both streams need at least two samples")
    if args.labeled_batch_size % 2 or (args.batch_size - args.labeled_batch_size) % 2:
        raise ValueError("stream batch sizes must be even")
    run_root = Path(args.run_root); (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_root / "console.log")], force=True)
    seed_everything(args.seed); device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    labeled = MSCMRSliceDataset(args.data_root, args.labeled_manifest, args.image_size, augment=True, include_label=True)
    unlabeled = MSCMRSliceDataset(args.data_root, args.unlabeled_manifest, args.image_size, augment=True, include_label=False)
    sampler = PairedInfiniteBatchSampler(len(labeled), len(unlabeled), args.labeled_batch_size, args.batch_size - args.labeled_batch_size, args.seed)
    loader = DataLoader(ConcatDataset([labeled, unlabeled]), batch_sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model1, model2, teacher = make_model(device), make_model(device), make_model(device)
    pretrain = load_checkpoint(args.pretrain_checkpoint, device)
    for model in (model1, model2, teacher): model.load_state_dict(pretrain["model"], strict=True)
    teacher.requires_grad_(False); teacher.eval()
    token1 = token2 = None; hook1 = hook2 = None
    if args.enable_mask_token:
        token1, token2 = MaskTokenAdapter(16, args.mask_token_num).to(device), MaskTokenAdapter(16, args.mask_token_num).to(device)
        hook1, hook2 = DecoderHeadRouter(model1, token1), DecoderHeadRouter(model2, token2); hook1.attach(); hook2.attach()
    params1 = list(model1.parameters()) + ([] if token1 is None else list(token1.parameters()))
    params2 = list(model2.parameters()) + ([] if token2 is None else list(token2.parameters()))
    optimizer1 = torch.optim.SGD(params1, lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(params2, lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler1, scheduler2 = make_poly_scheduler(optimizer1, args.max_iterations), make_poly_scheduler(optimizer2, args.max_iterations)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    identity = checkpoint_identity(args, SOURCES, args.labeled_manifest, args.pretrain_checkpoint); identity["unlabeled_split_path"] = args.unlabeled_manifest; identity["unlabeled_split_sha256"] = sha256_file(args.unlabeled_manifest)
    (run_root / "config.json").write_text(json.dumps({**vars(args), "candidate_config": candidate_config(args)}, indent=2, sort_keys=True) + "\n")
    iteration, best, best_iteration, last_validation = 0, -1.0, 0, None
    if args.resume_checkpoint:
        iteration, best, best_iteration, last_validation = _resume(load_checkpoint(args.resume_checkpoint, device), args, model1, model2, teacher, optimizer1, optimizer2, scheduler1, scheduler2, scaler, sampler, token1, token2)
    assert_roles(model1, model2, teacher, "initialization_or_resume")
    logging.info("START candidate ratio=%s seed=%d iteration=%d mask_token=%s adamix=%s model1_mode=TRAIN model2_mode=TRAIN teacher_mode=EVAL", args.label_ratio, args.seed, iteration, args.enable_mask_token, args.enable_adamix)
    if args.dry_run:
        batch = next(iter(loader)); logging.info("DRY_RUN image=%s label=%s labeled_flags=%s", tuple(batch["image"].shape), tuple(batch["label"].shape), batch["is_labeled"].tolist()); return 0
    timeline = (run_root / "training_timeline.jsonl").open("a", buffering=1); iterator = iter(loader)
    try:
      while iteration < args.max_iterations:
        assert_roles(model1, model2, teacher, f"train_step_{iteration + 1}_entry")
        batch = next(iterator); weak, strong = batch["image"].to(device), batch["image_strong"].to(device)
        labels, labels_strong = batch["label"].to(device), batch["label_strong"].to(device)
        lbs, ubs, lh, uh = args.labeled_batch_size, args.batch_size - args.labeled_batch_size, args.labeled_batch_size // 2, (args.batch_size - args.labeled_batch_size) // 2
        la, lb, ua, ub = weak[:lh], weak[lh:lbs], weak[lbs:lbs+uh], weak[lbs+uh:]
        la_s, lb_s, ua_s, ub_s = strong[:lh], strong[lh:lbs], strong[lbs:lbs+uh], strong[lbs+uh:]
        ta, tb, ta_s, tb_s = labels[:lh], labels[lh:lbs], labels_strong[:lh], labels_strong[lh:lbs]
        with torch.no_grad():
            prob_a, prob_b, prob_a_s, prob_b_s = torch.softmax(teacher(ua), 1), torch.softmax(teacher(ub), 1), torch.softmax(teacher(ua_s), 1), torch.softmax(teacher(ub_s), 1)
        pseudo_a, pseudo_b, pseudo_a_s, pseudo_b_s = prob_a.argmax(1), prob_b.argmax(1), prob_a_s.argmax(1), prob_b_s.argmax(1)
        teacher_foreground_proxy = float(torch.cat([pseudo_a, pseudo_b, pseudo_a_s, pseudo_b_s]).gt(0).float().mean())
        keep = rectangular_mask(lh, weak.shape[-2], weak.shape[-1], device)
        input_u1, input_l1 = ua*keep[:,None]+la*(1-keep[:,None]), lb*keep[:,None]+ub*(1-keep[:,None])
        input_u2, input_l2 = ua_s*keep[:,None]+la_s*(1-keep[:,None]), lb_s*keep[:,None]+ub_s*(1-keep[:,None])
        adamix_record = None
        if args.enable_adamix:
            r1 = adamix_pair(input_u1, input_u2, pseudo_a, pseudo_a_s, ta, ta_s, prob_a.max(1).values, prob_a_s.max(1).values)
            r2 = adamix_pair(input_u2, input_u1, pseudo_a_s, pseudo_a, ta_s, ta, prob_a_s.max(1).values, prob_a.max(1).values)
            input_u1, pseudo_a, ta = r1.image, r1.first_target, r1.second_target
            input_u2, pseudo_a_s, ta_s = r2.image, r2.first_target, r2.second_target
            adamix_record = {"trigger_count": 2, "permutation": r1.permutation, "mask_shape": list(r1.mask.shape), "mask_dtype": str(r1.mask.dtype), "mask_min": float(r1.mask.min()), "mask_max": float(r1.mask.max()), "mask_nonzero_ratio": float(r1.mask.mean()), "mixed_image_shape": list(r1.image.shape), "mixed_target_shape": list(r1.first_target.shape), "label_min": int(torch.minimum(r1.first_target.min(), r1.second_target.min())), "label_max": int(torch.maximum(r1.first_target.max(), r1.second_target.max())), "ignore_pixel_count": int((r1.first_target < 0).sum() + (r1.second_target < 0).sum()), "confidence_min": float(r1.confidence.min()), "confidence_max": float(r1.confidence.max()), "confidence_mean": float(r1.confidence.mean()), "entered_unsupervised_loss": True}
        optimizer1.zero_grad(set_to_none=True); optimizer2.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            out_u1, out_l1, out_u2, out_l2 = model1(input_u1), model1(input_l1), model2(input_u2), model2(input_l2)
            lu1, _, _ = mixed_supervised_loss(out_u1, pseudo_a, ta, keep, args.u_weight, 1.0); ll1, _, _ = mixed_supervised_loss(out_l1, tb, pseudo_b, keep, 1.0, args.u_weight)
            lu2, _, _ = mixed_supervised_loss(out_u2, pseudo_a_s, ta_s, keep, args.u_weight, 1.0); ll2, _, _ = mixed_supervised_loss(out_l2, tb_s, pseudo_b_s, keep, 1.0, args.u_weight)
            logits1, logits2 = torch.cat([out_u1,out_l1]), torch.cat([out_u2,out_l2]); cross1, cross2 = soft_dice_loss(logits1, logits2.detach().argmax(1)), soft_dice_loss(logits2, logits1.detach().argmax(1))
            fft_images = fft_low_frequency_swap(torch.cat([input_u1,input_l1]), torch.cat([input_u2,input_l2]), args.fft_radius); fft1, fft2 = model1(fft_images), model2(fft_images)
            fft_peer1, fft_peer2 = soft_dice_loss(fft1, fft2.detach().argmax(1)), soft_dice_loss(fft2, fft1.detach().argmax(1))
            beta = torch.from_numpy(np.random.beta(0.1,0.1,size=(uh,1,1,1))).float().to(device); mixed_unlabeled = ua*(1-beta)+ub*beta; licr_logits = model1(torch.cat([weak[:lbs], mixed_unlabeled])); supervised_licr, _, _ = supervised_loss(licr_logits[:lbs], labels[:lbs])
            with torch.no_grad(): teacher_mix = prob_a*(1-beta)+prob_b*beta
            weight = baseline.consistency_weight(iteration,args.consistency_rampup,args.consistency); licr = F.mse_loss(torch.softmax(licr_logits[lbs:],1),teacher_mix)*weight
            loss1, loss2, loss = lu1+ll1+cross1+fft_peer1, lu2+ll2+cross2+fft_peer2, lu1+ll1+cross1+fft_peer1+lu2+ll2+cross2+fft_peer2+supervised_licr+licr
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {iteration+1}")
        before_token = token1.tokens.detach().clone() if token1 is not None else None
        scaler.scale(loss).backward()
        token_grad_norm = None if token1 is None else float(token1.tokens.grad.norm())
        if token1 is not None and (token1.tokens.grad is None or not torch.isfinite(token1.tokens.grad).all() or token_grad_norm <= 0): raise FloatingPointError("Mask Token gradient is missing/non-finite/zero")
        scaler.step(optimizer1); scaler.step(optimizer2); scaler.update(); scheduler1.step(); scheduler2.step(); iteration += 1; update_ema(model1,teacher,args.ema_decay); assert_roles(model1,model2,teacher,f"ema_update_{iteration}_exit")
        validation = None
        if iteration == 1 or iteration % 20 == 0:
            record = {"iteration":iteration,"loss":float(loss.detach()),"model1_loss":float(loss1.detach()),"model2_loss":float(loss2.detach()),"licr":float(licr.detach()),"consistency_weight":weight,"fft_finite":bool(torch.isfinite(fft_images).all()),"ema_finite":all(torch.isfinite(p).all() for p in teacher.parameters()),"teacher_foreground_proxy":teacher_foreground_proxy,"model1_mode":"TRAIN","model2_mode":"TRAIN","teacher_mode":"EVAL","adamix":adamix_record,"mask_token":{"parameter":"tokens","shape":list(token1.tokens.shape),"grad_norm":token_grad_norm,"update_max_abs_delta":None if token1 is None else float((token1.tokens-before_token).abs().max())}}
            timeline.write(json.dumps(record)+"\n")
        if iteration % args.validation_interval == 0 or iteration == args.max_iterations:
            with preserve_module_modes(model1,model2,teacher,*(x for x in (token1,token2) if x is not None)):
                model1.eval(); model2.eval(); teacher.eval(); val1 = evaluate_manifest(model1,args.data_root,args.validation_manifest,args.image_size,device,role=args.validation_role)
            assert_roles(model1,model2,teacher,f"validation_{iteration}_exit"); validation = val1; last_validation={"selected_model":"model1","selected":val1,"model1":val1}
            if val1["mDice"] > best: best,best_iteration=val1["mDice"],iteration
            logging.info("VALIDATION iteration=%d selected=model1 mDice=%.8f RV=%.8f MYO=%.8f LV=%.8f",iteration,val1["mDice"],val1["classes"]["RV"]["dice"],val1["classes"]["MYO"]["dice"],val1["classes"]["LV"]["dice"])
        payload = _make_payload(identity,args,model1,model2,teacher,optimizer1,optimizer2,scheduler1,scheduler2,scaler,iteration,best,best_iteration,last_validation,sampler,token1,token2)
        if validation is not None and best_iteration == iteration: save_checkpoint(run_root/"checkpoints/best.ckpt",payload)
        if iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations: save_checkpoint(run_root/f"checkpoints/iter_{iteration:06d}.ckpt",payload); save_checkpoint(run_root/"checkpoints/last.ckpt",payload)
      logging.info("NATURAL_COMPLETION iteration=%d best=%.8f@%d model=model1",iteration,best,best_iteration)
    finally:
      timeline.close()
      if hook1: hook1.remove(); hook2.remove()
    return 0


if __name__ == "__main__": raise SystemExit(main())
