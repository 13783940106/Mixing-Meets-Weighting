#!/usr/bin/env python3
"""Launch the existing paper trainers with recorded settings and portable paths."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_command(config, data_root, weights_root, output_root, gpu, arm=None):
    settings = dict(config["training_args"])
    acdc = config["dataset"] == "acdc"
    if arm is not None:
        if not acdc or config["task_id"] != "Full_main":
            raise ValueError("--arm applies to the ACDC10 factorial configuration")
        settings.update(mask_token_enabled=arm[0] == "1", adamix_enabled=arm[1] == "1",
                        enable_v21_3_innovation_x=arm[2] == "1")
    output = output_root / (Path(config["output_dir"]).name if arm is None else "acdc10_arm_" + arm)
    settings["root_path"] = str(data_root / config["data_subdir"])
    settings["gpu"] = str(gpu)
    if acdc:
        settings["run_root"] = str(output)
        settings["pretrained_model"] = str(weights_root / config["pretrain"]["initialization"])
    else:
        settings["v22_output_root_override"] = str(output)
        settings["cfg"] = str(ROOT / "train_promise12/configs/swin_tiny_patch4_window7_224_lite.yaml")
        if config["dataset"] == "covid19":
            for role in ("labeled", "unlabeled", "val"):
                settings["covid19_" + role + "_manifest"] = str(ROOT / config["manifests"][role])
    command = [sys.executable, str(ROOT / config["entry"])]
    hyphenated = {"cache_mode", "amp_opt_level"}
    for key, value in settings.items():
        flag = "--" + (key.replace("_", "-") if key in hyphenated else key)
        if isinstance(value, bool) and not acdc:
            if value:
                command.append(flag)
        else:
            command.append(flag)
            if isinstance(value, list):
                command.extend(map(str, value))
            else:
                command.append(str(value).lower() if isinstance(value, bool) else str(value))
    return command, output


def check_inputs(config, data_root, weights_root, output):
    data = data_root / config["data_subdir"]
    if not data.is_dir():
        raise FileNotFoundError(data)
    if output.exists():
        raise FileExistsError(f"Choose a fresh output root; existing run: {output}")
    if config["dataset"] == "acdc":
        initialization = weights_root / config["pretrain"]["initialization"] / "unet_best_model.pth"
        if not initialization.is_file():
            raise FileNotFoundError(initialization)
        expected_sha = config["pretrain"]["initialization_sha256"]
        if hashlib.sha256(initialization.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Frozen initialization differs: {initialization}")
        for filename in ("train_slices.list", "val.list"):
            expected = ROOT / config["split_root"] / filename
            actual = data / filename
            if actual.read_bytes() != expected.read_bytes():
                raise ValueError(f"Frozen split differs: {actual}")
    elif config["dataset"] == "promise12":
        for filename in ("val.list", "test.list"):
            if (data / filename).read_text().splitlines() != (ROOT / config["split_root"] / filename).read_text().splitlines():
                raise ValueError(f"Frozen split differs: {data / filename}")
        for filename in ("X_train.npy", "y_train.npy"):
            if not (data / "npy_image" / filename).is_file():
                raise FileNotFoundError(data / "npy_image" / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--weights-root", default=os.environ.get("WEIGHTS_ROOT", "weights"))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "outputs"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--arm", choices=[format(i, "03b") for i in range(8)])
    parser.add_argument("--print-command", action="store_true", help="Print argv without importing the trainer")
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text())
    data_root, weights_root, output_root = map(resolve, (args.data_root, args.weights_root, args.output_root))
    command, output = build_command(config, data_root, weights_root, output_root, args.gpu, args.arm)
    if args.print_command:
        print(json.dumps({"argv": command, "cwd": str(ROOT), "output": str(output)}, indent=2))
        return
    check_inputs(config, data_root, weights_root, output)
    env = dict(os.environ, DATA_ROOT=str(data_root), CUDA_VISIBLE_DEVICES=str(args.gpu))
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
