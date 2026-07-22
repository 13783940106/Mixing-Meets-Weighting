#!/usr/bin/env python3
"""Clean-process Default-Off equivalence harness (validation only; never test)."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
BASELINE = ROOT / "train_MSCMRSEG/v15_1_mscmr_beta_fft_baseline.py"
CANDIDATE = ROOT / "train_MSCMRSEG/v15_1_mscmr_mask_token_adamix_candidate.py"


def compare(a, b, path=""):
    if isinstance(a, torch.Tensor):
        return (float((a.detach().cpu()-b.detach().cpu()).abs().max()) if a.numel() else 0.0, a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict):
        if set(a) != set(b): return (float("inf"), False)
        values = [compare(a[k], b[k], path+"."+str(k)) for k in a]
        return (max((x[0] for x in values), default=0.0), all(x[1] for x in values))
    if isinstance(a, (tuple, list)):
        if len(a) != len(b): return (float("inf"), False)
        values = [compare(x,y,path) for x,y in zip(a,b)]
        return (max((x[0] for x in values), default=0.0), all(x[1] for x in values))
    if isinstance(a, np.ndarray):
        return (float(np.max(np.abs(a-b))) if a.size else 0.0, bool(np.array_equal(a,b)))
    return (0.0, a == b)


def command(script, ratio, run_root):
    p5 = ROOT / "runs/mscmr_5pct_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt"
    p10 = ROOT / "runs/mscmr_10pct_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt"
    return [str(PYTHON), str(script), "--data_root", "/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw",
        "--labeled_manifest", str(ROOT/f"audit/mscmr_{ratio}_labeled_manifest.tsv"), "--unlabeled_manifest", str(ROOT/f"audit/mscmr_{ratio}_unlabeled_manifest.tsv"),
        "--validation_manifest", str(ROOT/"audit/mscmr_split_manifest.tsv"), "--validation_role", "validation", "--label_ratio", ratio,
        "--pretrain_checkpoint", str(p5 if ratio == "5pct" else p10), "--run_root", str(run_root), "--seed", "1337", "--gpu", "0",
        "--max_iterations", "1", "--validation_interval", "1", "--checkpoint_interval", "1", "--num_workers", "0"]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ratio", choices=("5pct","10pct"), required=True); ap.add_argument("--output", required=True); args=ap.parse_args()
    out=Path(args.output).resolve(); base, cand=out/"baseline",out/"candidate_off"
    shutil.rmtree(out, ignore_errors=True); base.mkdir(parents=True); cand.mkdir(parents=True)
    # CPU makes the two clean processes bitwise comparable.  This is an
    # equivalence gate, not a performance run, and still uses the real data,
    # augmentation, training step, validation evaluator and checkpoint path.
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=""
    subprocess.run(command(BASELINE,args.ratio,base),check=True,env=env); subprocess.run(command(CANDIDATE,args.ratio,cand),check=True,env=env)
    a=torch.load(base/"checkpoints/last.ckpt",map_location="cpu",weights_only=False); b=torch.load(cand/"checkpoints/last.ckpt",map_location="cpu",weights_only=False)
    tensor_keys=("model1","model2","ema_teacher","optimizer1","optimizer2","scheduler","amp_scaler","random_state","sampler_state")
    results={k: compare(a[k],b[k]) for k in tensor_keys}
    timeline_a=json.loads((base/"training_timeline.jsonl").read_text().splitlines()[-1]); timeline_b=json.loads((cand/"training_timeline.jsonl").read_text().splitlines()[-1])
    report={"ratio":args.ratio,"formal_test_access":False,"validation_role":"validation","baseline_command":command(BASELINE,args.ratio,base),"candidate_command":command(CANDIDATE,args.ratio,cand),"checkpoint_tensor_keys_equal":set(a["model1"])==set(b["model1"]),"state_comparisons":{k:{"max_abs_error":v[0],"exact":v[1]} for k,v in results.items()},"timeline_exact":timeline_a==timeline_b,"global_max_abs_error":max(v[0] for v in results.values()),"pass":all(v[1] for v in results.values()) and timeline_a==timeline_b}
    (out/"result.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps(report,indent=2,default=str)); return 0 if report["pass"] else 1

if __name__ == "__main__": raise SystemExit(main())
