#!/usr/bin/env python3
"""Evaluate an explicitly provided validation-selected checkpoint; never select a model."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def promise_binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from medpy import metric

    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    denom = int(pred.sum() + target.sum())
    if pred.any() and target.any():
        hd95 = float(metric.binary.hd95(pred, target))
        asd = float(metric.binary.asd(pred, target))
    elif not pred.any() and not target.any():
        hd95 = asd = 0.0
    else:
        hd95 = asd = float(np.linalg.norm(pred.shape))
    return {
        "dice": 0.0 if denom == 0 else float(2.0 * inter / denom),
        "iou": 0.0 if union == 0 else float(inter / union),
        "hd95": hd95,
        "asd": asd,
    }


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config, checkpoint, data_root, split, output):
    import numpy as np
    import torch

    data = data_root / config["data_subdir"]
    if config["dataset"] == "acdc":
        # Exact existing ACDC model construction, inference and metric functions.
        from tools.v22_0_acdc_test_frozen_checkpoint import infer_case, summarize, v14_bcp_net
        model = v14_bcp_net(in_chns=1, class_num=4, ema=False, mask_token_enabled=config["components"]["TRC"], mask_token_num=4, mask_token_dim=16)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_1"], strict=True)
        model.eval()
        cases = (ROOT / config["split_root"] / (split + ".list")).read_text().splitlines()
        with torch.inference_mode():
            rows = [row for case in cases for row in infer_case(model, data, case, "cuda:0")]
        write_rows(output / "case_class_metrics.tsv", rows)
        return summarize(rows)

    sys.path.insert(0, str(ROOT / "train_promise12"))
    from networks.net_factory import net_factory
    model = net_factory(net_type="unet", in_chns=1, class_num=2,
                        use_mask_token_prehead_router=config["components"]["TRC"], mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    if config["dataset"] == "promise12":
        cases = (ROOT / config["split_root"] / (split + ".list")).read_text().splitlines()
        rows = []
        with torch.inference_mode():
            for case in cases:
                image = np.load(data / "npy_image" / (case + ".npy"))
                target = np.load(data / "npy_image" / (case + "_segmentation.npy"))
                pred = np.zeros_like(target, dtype=np.uint8)
                for z in range(image.shape[0]):
                    tensor = torch.from_numpy(image[z]).unsqueeze(0).unsqueeze(0).float().cuda()
                    pred[z] = torch.softmax(model(tensor), dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                rows.append({"case": case, **promise_binary_metrics(pred == 1, target == 1)})
        write_rows(output / "case_metrics.tsv", rows)
        return {key: float(np.mean([row[key] for row in rows])) for key in ("dice", "iou", "hd95", "asd")}

    from train_COVID19.covid19_dataset_adapter import Covid19Official20PatientVolumeDataset
    from train_COVID19.covid19_volume_evaluator import SlicePrediction, patient_volume_macro
    dataset = Covid19Official20PatientVolumeDataset(ROOT / config["manifests"][split])
    slices = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            image = sample["image"].float()
            target = sample["mask"].cpu().numpy().astype(np.uint8)
            spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
            for z in range(image.shape[0]):
                tensor = image[z].unsqueeze(0).unsqueeze(0).float().cuda()
                pred = torch.softmax(model(tensor), dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                slices.append(SlicePrediction(str(sample["patient_id"]), int(sample["slice_idx"][z]), pred > 0, target[z] > 0, spacing))
    result = patient_volume_macro(slices)
    (output / "patient_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    macro = result["macro"]
    return {"dice": macro["dice"], "iou": macro["iou"], "hd95": macro["hd95_mm"], "asd": macro["asd_mm"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Defaults to the validation-best path in the config")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text())
    checkpoint = resolve(args.checkpoint or config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data_root = resolve(args.data_root)
    os.environ["DATA_ROOT"] = str(data_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = resolve(args.output_dir or str(Path(config["output_dir"]) / ("eval_" + args.split)))
    output.mkdir(parents=True, exist_ok=False)
    metrics = evaluate(config, checkpoint, data_root, args.split, output)
    receipt = {"task_id": config["task_id"], "split": args.split, "checkpoint": str(checkpoint),
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "unit": config["evaluation"]["unit"], "metrics": metrics,
               "training_started": False, "checkpoint_reselected": False}
    (output / "metrics.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
