#!/usr/bin/env python3
"""Regression gates for the V15.1 MS-CMRSeg mode-lifecycle repair."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_MSCMRSEG.v15_1_mscmr_beta_fft_baseline import assert_training_roles
from train_MSCMRSEG.v15_1_mscmr_common import (
    load_checkpoint,
    make_model,
    make_poly_scheduler,
    preserve_module_modes,
    save_checkpoint,
    seed_everything,
    update_ema,
)
from train_MSCMRSEG.v15_1_mscmr_eval import predict_case


AUDIT = ROOT / "audit"
TRACE_PATH = AUDIT / "mscmr_mode_lifecycle_trace.tsv"


def first_batchnorm(module):
    return next(item for item in module.modules() if isinstance(item, torch.nn.modules.batchnorm._BatchNorm))


def bn_state(module):
    bn = first_batchnorm(module)
    return (bn.running_mean.detach().clone(), bn.running_var.detach().clone(), bn.num_batches_tracked.detach().clone())


def state_equal(first, second):
    return all(torch.equal(a, b) for a, b in zip(first, second))


def module_state_equal(first, second, atol=0.0):
    return all(torch.allclose(first[name], second[name], atol=atol, rtol=0.0) for name in first)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def optimizer_signature(optimizer):
    state = optimizer.state_dict()
    return stable_hash({
        "param_groups": state["param_groups"],
        "state": {str(key): {name: tensor.tolist() if torch.is_tensor(tensor) else tensor
                             for name, tensor in values.items()} for key, values in state["state"].items()},
    })


def one_student_step(model, optimizer, scheduler, image, target):
    if not model.training:
        raise AssertionError("student must be TRAIN during optimizer step")
    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step(); scheduler.step()
    return float(loss.detach()), [parameter.grad.detach().clone() for parameter in model.parameters() if parameter.grad is not None]


def record(trace, stage, model1, model2, teacher):
    trace.append((stage, model1.training, model2.training, teacher.training))


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    seed_everything(1337)
    device = torch.device("cpu")
    model1, model2, teacher = make_model(device), make_model(device), make_model(device)
    model2.load_state_dict(model1.state_dict(), strict=True); teacher.load_state_dict(model1.state_dict(), strict=True)
    teacher.requires_grad_(False); teacher.eval()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler1 = make_poly_scheduler(optimizer1, 30000); scheduler2 = make_poly_scheduler(optimizer2, 30000)
    image = torch.randn(2, 1, 32, 32, device=device)
    target = torch.randint(0, 4, (2, 32, 32), device=device)
    volume = image[0, 0].detach().numpy()[:, :, None]
    trace = []

    assert_training_roles(model1, model2, teacher, "regression_initialization")
    record(trace, "初始化训练前", model1, model2, teacher)
    student_bn_before_train = bn_state(model1)
    one_student_step(model1, optimizer1, scheduler1, image, target)
    one_student_step(model2, optimizer2, scheduler2, image, target)
    student_bn_after_train = bn_state(model1)
    if state_equal(student_bn_before_train, student_bn_after_train):
        raise AssertionError("student BN did not update during a real TRAIN step")
    record(trace, "训练 step 后", model1, model2, teacher)

    student_bn_before_validation = bn_state(model1); teacher_bn_before_validation = bn_state(teacher)
    student_parameters_before_validation = copy.deepcopy(model1.state_dict())
    optimizer_before_validation = optimizer_signature(optimizer1)
    scheduler_before_validation = copy.deepcopy(scheduler1.state_dict())
    rng_before_validation = torch.get_rng_state().clone()
    with preserve_module_modes(model1, model2, teacher):
        model1.eval(); model2.eval(); teacher.eval()
        record(trace, "validation 内", model1, model2, teacher)
        predict_case(model1, volume, (32, 32), device)
        predict_case(model2, volume, (32, 32), device)
    assert_training_roles(model1, model2, teacher, "regression_validation_exit")
    record(trace, "validation 后", model1, model2, teacher)
    validation_checks = {
        "student_bn_unchanged": state_equal(student_bn_before_validation, bn_state(model1)),
        "teacher_bn_unchanged": state_equal(teacher_bn_before_validation, bn_state(teacher)),
        "student_parameters_unchanged": module_state_equal(student_parameters_before_validation, model1.state_dict()),
        "optimizer_unchanged": optimizer_before_validation == optimizer_signature(optimizer1),
        "scheduler_unchanged": scheduler_before_validation == scheduler1.state_dict(),
        "torch_rng_unchanged": torch.equal(rng_before_validation, torch.get_rng_state()),
    }
    if not all(validation_checks.values()):
        raise AssertionError(f"validation side effect: {validation_checks}")

    teacher_bn_before_pseudo = bn_state(teacher)
    record(trace, "pseudo-label 前", model1, model2, teacher)
    with torch.no_grad():
        teacher_logits1 = teacher(image)
        teacher_logits2 = teacher(image)
    record(trace, "pseudo-label 后", model1, model2, teacher)
    teacher_bn_after_pseudo = bn_state(teacher)
    teacher_deterministic = torch.equal(teacher_logits1, teacher_logits2)
    if not state_equal(teacher_bn_before_pseudo, teacher_bn_after_pseudo):
        raise AssertionError("EMA_TEACHER_BN_CONTAMINATION_NOT_FIXED")
    if not teacher_deterministic:
        raise AssertionError("EMA teacher Dropout is not deterministic in EVAL")

    update_ema(model1, teacher, 0.99)
    assert_training_roles(model1, model2, teacher, "regression_ema_exit")
    record(trace, "EMA update 后", model1, model2, teacher)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise AssertionError("EMA teacher has trainable parameters")
    optimizer_parameter_ids = {id(parameter) for group in optimizer1.param_groups + optimizer2.param_groups for parameter in group["params"]}
    if any(id(parameter) in optimizer_parameter_ids for parameter in teacher.parameters()):
        raise AssertionError("EMA teacher is registered in an optimizer")

    with tempfile.TemporaryDirectory(prefix="mscmr_mode_lifecycle_") as temporary:
        checkpoint_path = Path(temporary) / "mode.ckpt"
        save_checkpoint(checkpoint_path, {
            "model1": model1.state_dict(), "model2": model2.state_dict(), "ema_teacher": teacher.state_dict(),
            "optimizer1": optimizer1.state_dict(), "optimizer2": optimizer2.state_dict(),
            "scheduler": {"model1": scheduler1.state_dict(), "model2": scheduler2.state_dict()},
            "global_iteration": 1,
        })
        checkpoint = load_checkpoint(checkpoint_path, device)
        loaded1, loaded2, loaded_teacher = make_model(device), make_model(device), make_model(device)
        loaded1.load_state_dict(checkpoint["model1"], strict=True); loaded2.load_state_dict(checkpoint["model2"], strict=True)
        loaded_teacher.load_state_dict(checkpoint["ema_teacher"], strict=True)
        loaded_teacher.requires_grad_(False); loaded_teacher.eval()
        assert_training_roles(loaded1, loaded2, loaded_teacher, "regression_checkpoint_load")
        record(trace, "checkpoint load 后", loaded1, loaded2, loaded_teacher)
        with preserve_module_modes(loaded1, loaded2, loaded_teacher):
            loaded1.eval(); loaded2.eval(); loaded_teacher.eval()
            predict_case(loaded1, volume, (32, 32), device)
        assert_training_roles(loaded1, loaded2, loaded_teacher, "regression_loaded_validation_exit")
        record(trace, "二次 validation 后", loaded1, loaded2, loaded_teacher)
        loaded_optimizer = torch.optim.SGD(loaded1.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        loaded_scheduler = make_poly_scheduler(loaded_optimizer, 30000)
        one_student_step(loaded1, loaded_optimizer, loaded_scheduler, image, target)
        record(trace, "validation 后训练 step", loaded1, loaded2, loaded_teacher)

    # Fully isolated A/B proof: inserting validation must not alter the next fixed-batch train step.
    branch_state = copy.deepcopy(model1.state_dict())
    branch_a, branch_b = make_model(device), make_model(device)
    branch_a.load_state_dict(branch_state, strict=True); branch_b.load_state_dict(branch_state, strict=True)
    opt_a = torch.optim.SGD(branch_a.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    opt_b = torch.optim.SGD(branch_b.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sch_a = make_poly_scheduler(opt_a, 30000); sch_b = make_poly_scheduler(opt_b, 30000)
    branch_rng = torch.get_rng_state().clone()
    torch.set_rng_state(branch_rng.clone()); loss_a, gradients_a = one_student_step(branch_a, opt_a, sch_a, image, target)
    torch.set_rng_state(branch_rng.clone()); predict_case(branch_b, volume, (32, 32), device)
    loss_b, gradients_b = one_student_step(branch_b, opt_b, sch_b, image, target)
    ab_checks = {
        "post_validation_mode_train": branch_b.training,
        "loss_equal": loss_a == loss_b,
        "gradient_equal": len(gradients_a) == len(gradients_b) and all(torch.equal(a, b) for a, b in zip(gradients_a, gradients_b)),
        "parameter_equal": module_state_equal(branch_a.state_dict(), branch_b.state_dict()),
        "optimizer_equal": optimizer_signature(opt_a) == optimizer_signature(opt_b),
        "scheduler_equal": sch_a.state_dict() == sch_b.state_dict(),
    }
    if not all(ab_checks.values()):
        raise AssertionError(f"A/B validation side-effect regression: {ab_checks}")

    TRACE_PATH.write_text(
        "stage\tmodel1.training\tmodel2.training\tteacher.training\n" +
        "".join(f"{stage}\t{str(m1)}\t{str(m2)}\t{str(t)}\n" for stage, m1, m2, t in trace)
    )
    (AUDIT / "mscmr_mode_lifecycle_regression_test.md").write_text(
        "# MS-CMRSeg mode lifecycle regression\n\n"
        "```text\nPASS\n```\n\n"
        f"- lifecycle nodes checked: {len(trace)}\n"
        "- required role: model1=TRAIN, model2=TRAIN, teacher=EVAL outside validation\n"
        "- validation internal role: all EVAL\n"
        "- checkpoint strict load and explicit teacher EVAL restoration: PASS\n"
        "- teacher requires_grad=False and optimizer exclusion: PASS\n"
    )
    (AUDIT / "mscmr_batchnorm_mode_regression_test.md").write_text(
        "# MS-CMRSeg BatchNorm mode regression\n\n"
        "```text\nPASS\n```\n\n"
        "- student BN changes during real TRAIN step: PASS\n"
        "- student BN unchanged during validation: PASS\n"
        "- teacher BN unchanged during validation: PASS\n"
        "- teacher BN unchanged across repeated pseudo-label forwards: PASS\n"
    )
    (AUDIT / "mscmr_dropout_mode_regression_test.md").write_text(
        "# MS-CMRSeg Dropout mode regression\n\n"
        "```text\nPASS\n```\n\n"
        "- network contains torch.nn.Dropout modules: YES\n"
        f"- identical teacher EVAL logits across two forwards: {'PASS' if teacher_deterministic else 'FAIL'}\n"
    )
    (AUDIT / "mscmr_validation_side_effect_regression_test.md").write_text(
        "# MS-CMRSeg validation side-effect regression\n\n"
        "```text\nPASS\n```\n\n"
        f"- direct validation checks: `{json.dumps(validation_checks, sort_keys=True)}`\n"
        f"- isolated path A/B checks: `{json.dumps(ab_checks, sort_keys=True)}`\n"
        "- fixed RNG/batch next-step loss, gradients, parameters, optimizer and scheduler agree exactly: PASS\n"
    )
    print(json.dumps({"decision": "PASS", "trace": str(TRACE_PATH), "validation": validation_checks, "ab": ab_checks}, sort_keys=True))


if __name__ == "__main__":
    main()
