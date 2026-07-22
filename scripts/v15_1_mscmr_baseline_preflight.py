#!/usr/bin/env python3
"""One-shot engineering acceptance for the independent V15.1 baseline stack."""

from __future__ import annotations

import argparse, csv, json, pickle, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import torch

from train_MSCMRSEG.v15_1_mscmr_common import (fft_low_frequency_swap, load_checkpoint,
    make_model, make_poly_scheduler, mixed_supervised_loss, rectangular_mask, restore_rng_state,
    rng_state, save_checkpoint, seed_everything, sha256_file, soft_dice_loss, supervised_loss, update_ema)
from train_MSCMRSEG.v15_1_mscmr_dataset import MSCMRSliceDataset, read_patient_manifest, resize_slice
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest
from train_MSCMRSEG.v15_1_mscmr_protocol import raw_label_to_train_label


class OracleSequence(torch.nn.Module):
    def __init__(self, label, image_size):
        super().__init__(); self.slices = [resize_slice(label[:, :, z], image_size, True) for z in range(label.shape[2])]; self.index = 0
    def forward(self, image):
        target = torch.from_numpy(self.slices[self.index]).to(image.device); self.index += 1
        return torch.nn.functional.one_hot(target.long(), 4).permute(2, 0, 1)[None].float() * 20 - 10


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data_root", required=True); p.add_argument("--labeled_manifest", required=True)
    p.add_argument("--split_manifest", required=True); p.add_argument("--run_root", required=True); p.add_argument("--gpu", default="0")
    args = p.parse_args(); run = Path(args.run_root); run.mkdir(parents=True, exist_ok=True)
    seed_everything(1337); device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = MSCMRSliceDataset(args.data_root, args.labeled_manifest, (256, 256), augment=False)
    samples = [dataset[i] for i in range(4)]; image = torch.stack([s["image"] for s in samples]).to(device)
    label = torch.stack([s["label"] for s in samples]).to(device)
    model1, model2, teacher = make_model(device), make_model(device), make_model(device); teacher.load_state_dict(model1.state_dict()); teacher.requires_grad_(False)
    opt1 = torch.optim.SGD(model1.parameters(), 0.01, momentum=.9, weight_decay=1e-4)
    opt2 = torch.optim.SGD(model2.parameters(), 0.01, momentum=.9, weight_decay=1e-4)
    sch1, sch2 = make_poly_scheduler(opt1, 20), make_poly_scheduler(opt2, 20)
    logits1, logits2 = model1(image), model2(image)
    supervised, ce, dice = supervised_loss(logits1, label)
    keep = rectangular_mask(2, 256, 256, device); mixed = image[:2] * keep[:, None] + image[2:] * (1 - keep[:, None])
    bcp_logits = model1(mixed); bcp, _, _ = mixed_supervised_loss(bcp_logits, label[:2], label[2:], keep)
    fft_image = fft_low_frequency_swap(image[:2], image[2:], 30); fft1, fft2 = model1(fft_image), model2(fft_image)
    fft_peer = soft_dice_loss(fft1, fft2.detach().argmax(1)) + soft_dice_loss(fft2, fft1.detach().argmax(1))
    loss = supervised + bcp + fft_peer; opt1.zero_grad(); opt2.zero_grad(); loss.backward(); opt1.step(); opt2.step(); sch1.step(); sch2.step(); update_ema(model1, teacher, .99)
    finite = all(torch.isfinite(value).all().item() for value in (logits1, bcp_logits, fft_image, fft1, fft2, loss))
    saved_rng = rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    payload = {"model1": model1.state_dict(), "model2": model2.state_dict(), "ema_teacher": teacher.state_dict(),
        "optimizer1": opt1.state_dict(), "optimizer2": opt2.state_dict(), "scheduler": {"model1": sch1.state_dict(), "model2": sch2.state_dict()},
        "amp_scaler": {}, "global_iteration": 1, "current_epoch": 0.1, "best_validation_metric": 0., "best_iteration": 0,
        "random_state": saved_rng, "sampler_state": {"iteration": 1}, "args": vars(args)}
    checkpoint = run / "preflight.ckpt"; checkpoint_sha = save_checkpoint(checkpoint, payload)
    loaded = load_checkpoint(checkpoint, device); clone = make_model(device); clone.load_state_dict(loaded["model1"], strict=True)
    parameter_equal = all(torch.equal(a, b) for a, b in zip(model1.state_dict().values(), clone.state_dict().values()))
    restore_rng_state(loaded["random_state"]); actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    validation_rows = read_patient_manifest(args.split_manifest, role="validation"); first = validation_rows[0]
    single_manifest = run / "one_case_validation.tsv"
    with single_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=first.keys(), delimiter="\t"); writer.writeheader(); writer.writerow(first)
    label_volume = raw_label_to_train_label(np.asanyarray(nib.load(str(Path(args.data_root) / first["label_path"])).dataobj))
    oracle = OracleSequence(label_volume, (256, 256)).to(device)
    evaluation = evaluate_manifest(oracle, args.data_root, single_manifest, (256, 256), device)
    result = {"decision": "PASS" if finite and parameter_equal and expected == actual and evaluation["case_count"] == 1 else "FAIL",
        "device": str(device), "data": {"slices": len(dataset), "batch_shape": list(image.shape), "image_dtype": str(image.dtype),
        "label_dtype": str(label.dtype), "label_values": torch.unique(label).tolist(), "patient_ids": [s["patient_id"] for s in samples]},
        "model": {"logits_shape": list(logits1.shape), "supervised_loss": float(supervised), "ce_loss": float(ce), "dice_loss": float(dice),
        "bcp_loss": float(bcp), "fft_peer_loss": float(fft_peer), "fft_shape": list(fft_image.shape), "finite": finite,
        "backward": True, "optimizer_step": True, "ema_step": True},
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha, "strict_load": parameter_equal,
        "optimizer_present": True, "scheduler_present": True, "rng_exact": expected == actual, "resume_iteration": loaded["global_iteration"]},
        "evaluator": evaluation}
    (run / "preflight_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"decision": result["decision"], "finite": finite, "strict_load": parameter_equal,
                      "rng_exact": expected == actual, "case_mDice": evaluation["mDice"]}))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
