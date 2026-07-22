"""V13.6 observation-only real-guidance provenance acquisition.

This module is imported and called only when --enable_real_guidance_trace is set.
It never consumes a training batch and never returns a value used by training.
"""

import csv
import hashlib
import json
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from scipy import ndimage


MODEL_NAMES = ("student", "teacher", "peer")
KEY_ITERS = {0, 5000, 10000, 15000, 20000, 25000, 30000}


def _rng_snapshot():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def observation_context(models):
    """Evaluate without gradients, BN mutation, dropout, or RNG side effects."""
    rng = _rng_snapshot()
    modes = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for model, training in zip(models, modes):
            model.train(training)
        _rng_restore(rng)


def state_sha256(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = int(pred.sum()) + int(gt.sum())
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _safe_mean(values, mask=None):
    values = np.asarray(values)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    return float(values.mean()) if values.size else float("nan")


def _binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(labels.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    auroc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    descending = np.argsort(-scores, kind="mergesort")
    y = labels[descending]
    precision = np.cumsum(y) / np.arange(1, y.size + 1)
    auprc = float((precision * y).sum() / positives)
    return float(auroc), auprc


def _js_binary(a, b, eps=1e-7):
    a = np.clip(a.astype(np.float64), eps, 1.0 - eps)
    b = np.clip(b.astype(np.float64), eps, 1.0 - eps)
    m = 0.5 * (a + b)
    kl_a = a * np.log(a / m) + (1.0 - a) * np.log((1.0 - a) / (1.0 - m))
    kl_b = b * np.log(b / m) + (1.0 - b) * np.log((1.0 - b) / (1.0 - m))
    return 0.5 * (kl_a + kl_b)


def _regions(gt, student_pred):
    fg = gt.astype(bool)
    bg = ~fg
    eroded = ndimage.binary_erosion(fg, iterations=2)
    dilated = ndimage.binary_dilation(fg, iterations=2)
    boundary = np.logical_xor(dilated, eroded)
    interior = np.logical_and(fg, ~boundary)
    labels, count = ndimage.label(fg)
    small = np.zeros_like(fg)
    for component in range(1, count + 1):
        component_mask = labels == component
        if int(component_mask.sum()) <= 500:
            small |= component_mask
    return {
        "all": np.ones_like(fg),
        "background": bg,
        "foreground": fg,
        "boundary": boundary,
        "interior": interior,
        "small_component": small,
        "false_positive": np.logical_and(student_pred, bg),
        "false_negative": np.logical_and(~student_pred, fg),
    }


def _case_id(dataset, index):
    if hasattr(dataset, "sample_list") and index < len(dataset.sample_list):
        return str(dataset.sample_list[index])
    return "case_%03d" % index


class RealGuidanceTracer:
    def __init__(self, output_dir, validation_loader, validation_dataset, input_adapter,
                 boundary_radius=2, storage_budget_bytes=8 * 1024 ** 3):
        self.output_dir = os.path.abspath(output_dir)
        self.loader = validation_loader
        self.dataset = validation_dataset
        self.input_adapter = input_adapter
        self.boundary_radius = int(boundary_radius)
        self.storage_budget_bytes = int(storage_budget_bytes)
        self.dense_dir = os.path.join(self.output_dir, "dense")
        os.makedirs(self.dense_dir, exist_ok=True)
        self.timeline_path = os.path.join(self.output_dir, "guidance_trace_timeline.tsv")
        self.case_path = os.path.join(self.output_dir, "guidance_case_metrics.tsv")
        self.stage_path = os.path.join(self.output_dir, "guidance_stage_metrics.tsv")
        self.event_path = os.path.join(self.output_dir, "guidance_trace_events.jsonl")
        self._dense_bytes = 0

    @staticmethod
    def _predict_case(model, image):
        raise RuntimeError("bound adapter required")

    def _predict(self, model, image):
        image_np = image.squeeze(0).cpu().numpy()
        fg_prob = np.zeros_like(image_np, dtype=np.float32)
        for slice_index in range(image_np.shape[0]):
            tensor = torch.from_numpy(image_np[slice_index]).unsqueeze(0).unsqueeze(0).float().cuda()
            logits = model(self.input_adapter(tensor))
            fg_prob[slice_index] = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
        return fg_prob

    @staticmethod
    def _append_tsv(path, fieldnames, rows):
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def _save_dense(self, iteration, reason, cases, hashes):
        payload = {
            "iteration": np.asarray(iteration, dtype=np.int32),
            "reason": np.asarray(reason),
            "boundary_radius": np.asarray(self.boundary_radius, dtype=np.int16),
        }
        for name, value in hashes.items():
            payload["sha256_%s" % name] = np.asarray(value)
        for case in cases:
            prefix = case["case_id"].replace("/", "_")
            payload[prefix + "__gt"] = case["gt"].astype(np.uint8)
            for name in MODEL_NAMES:
                prob = case[name]
                pred = prob >= 0.5
                conf = np.maximum(prob, 1.0 - prob)
                entropy = -(prob * np.log(np.clip(prob, 1e-7, 1.0)) +
                            (1.0 - prob) * np.log(np.clip(1.0 - prob, 1e-7, 1.0))) / math.log(2.0)
                payload[prefix + "__%s_pred" % name] = pred.astype(np.uint8)
                payload[prefix + "__%s_conf" % name] = conf.astype(np.float16)
                payload[prefix + "__%s_entropy" % name] = entropy.astype(np.float16)
            sp = case["student"]
            tp, pp = case["teacher"], case["peer"]
            payload[prefix + "__st_disagree"] = ((sp >= 0.5) != (tp >= 0.5)).astype(np.uint8)
            payload[prefix + "__sp_disagree"] = ((sp >= 0.5) != (pp >= 0.5)).astype(np.uint8)
        filename = "rgpa_iter_%06d_%s.npz" % (iteration, reason.replace("+", "_"))
        path = os.path.join(self.dense_dir, filename)
        np.savez_compressed(path, **payload)
        self._dense_bytes += os.path.getsize(path)
        if self._dense_bytes > self.storage_budget_bytes:
            raise RuntimeError("RGPA dense storage budget exceeded: %d > %d" %
                               (self._dense_bytes, self.storage_budget_bytes))
        return path

    def capture(self, iteration, student, teacher, peer, student_best_refreshed=False):
        models = (student, teacher, peer)
        cases = []
        rng_before = _rng_snapshot()
        with observation_context(models):
            for case_index, batch in enumerate(self.loader):
                gt = batch["mask"].squeeze(0).cpu().numpy().astype(np.uint8)
                record = {"case_id": _case_id(self.dataset, case_index), "gt": gt}
                for name, model in zip(MODEL_NAMES, models):
                    record[name] = self._predict(model, batch["image"])
                cases.append(record)
        rng_after = _rng_snapshot()
        rng_unchanged = (
            rng_before["python"] == rng_after["python"] and
            np.array_equal(rng_before["numpy"][1], rng_after["numpy"][1]) and
            torch.equal(rng_before["torch_cpu"], rng_after["torch_cpu"]) and
            all(torch.equal(a, b) for a, b in zip(rng_before.get("torch_cuda", []), rng_after.get("torch_cuda", [])))
        )

        case_rows = []
        pooled = []
        finite = True
        for case in cases:
            gt = case["gt"].astype(bool)
            probs = {name: case[name] for name in MODEL_NAMES}
            preds = {name: probs[name] >= 0.5 for name in MODEL_NAMES}
            conf = {name: np.maximum(probs[name], 1.0 - probs[name]) for name in MODEL_NAMES}
            finite = finite and all(bool(np.isfinite(probs[name]).all()) for name in MODEL_NAMES)
            st = np.logical_and(preds["student"] == gt, preds["teacher"] != gt)
            ts = np.logical_and(preds["teacher"] == gt, preds["student"] != gt)
            sp = np.logical_and(preds["student"] == gt, preds["peer"] != gt)
            ps = np.logical_and(preds["peer"] == gt, preds["student"] != gt)
            row = {
                "iteration": iteration, "case_id": case["case_id"],
                "student_dsc": _dice(preds["student"], gt),
                "teacher_dsc": _dice(preds["teacher"], gt),
                "peer_dsc": _dice(preds["peer"], gt),
                "student_foreground_ratio": float(preds["student"].mean()),
                "teacher_foreground_ratio": float(preds["teacher"].mean()),
                "peer_foreground_ratio": float(preds["peer"].mean()),
                "student_confidence": float(conf["student"].mean()),
                "teacher_confidence": float(conf["teacher"].mean()),
                "peer_confidence": float(conf["peer"].mean()),
                "student_teacher_disagreement": float((preds["student"] != preds["teacher"]).mean()),
                "student_peer_disagreement": float((preds["student"] != preds["peer"]).mean()),
                "teacher_peer_disagreement": float((preds["teacher"] != preds["peer"]).mean()),
                "student_correct_teacher_wrong": float(st.mean()),
                "teacher_correct_student_wrong": float(ts.mean()),
                "student_correct_peer_wrong": float(sp.mean()),
                "peer_correct_student_wrong": float(ps.mean()),
                "student_high_conf_wrong": float(np.logical_and(preds["student"] != gt, conf["student"] >= 0.9).mean()),
                "teacher_high_conf_wrong": float(np.logical_and(preds["teacher"] != gt, conf["teacher"] >= 0.9).mean()),
                "peer_high_conf_wrong": float(np.logical_and(preds["peer"] != gt, conf["peer"] >= 0.9).mean()),
                "finite": finite, "rng_unchanged": rng_unchanged,
            }
            case_rows.append(row)
            pooled.append((case, preds, conf))
        self._append_tsv(self.case_path, list(case_rows[0]), case_rows)

        weights = np.asarray([case["gt"].size for case in cases], dtype=np.float64)
        timeline = {"iteration": iteration}
        for key in case_rows[0]:
            if key in ("iteration", "case_id", "finite", "rng_unchanged"):
                continue
            timeline[key] = float(np.average([row[key] for row in case_rows], weights=weights))
        timeline["finite"] = finite
        timeline["rng_unchanged"] = rng_unchanged
        timeline["student_best_refreshed"] = bool(student_best_refreshed)
        self._append_tsv(self.timeline_path, list(timeline), [timeline])

        full = iteration % 1000 == 0
        if full:
            stage_rows = []
            for case, preds, conf in pooled:
                gt = case["gt"].astype(bool)
                regions = _regions(gt, preds["student"])
                entropy = {}
                margin = {}
                for name in MODEL_NAMES:
                    p = case[name]
                    entropy[name] = -(p * np.log(np.clip(p, 1e-7, 1.0)) +
                                      (1.0-p) * np.log(np.clip(1.0-p, 1e-7, 1.0))) / math.log(2.0)
                    margin[name] = np.abs(2.0 * p - 1.0)
                js_st = _js_binary(case["student"], case["teacher"])
                js_sp = _js_binary(case["student"], case["peer"])
                js_tp = _js_binary(case["teacher"], case["peer"])
                for region_name, region in regions.items():
                    teacher_positive = np.logical_and(preds["teacher"] == gt, preds["student"] != gt)
                    teacher_negative = np.logical_and(preds["student"] == gt, preds["teacher"] != gt)
                    teacher_directional = np.logical_and(region, np.logical_or(teacher_positive, teacher_negative))
                    teacher_label = teacher_positive[teacher_directional].astype(np.uint8)
                    peer_positive = np.logical_and(preds["peer"] == gt, preds["student"] != gt)
                    peer_negative = np.logical_and(preds["student"] == gt, preds["peer"] != gt)
                    peer_directional = np.logical_and(region, np.logical_or(peer_positive, peer_negative))
                    peer_label = peer_positive[peer_directional].astype(np.uint8)
                    teacher_scores = {
                        "confidence": conf["teacher"][teacher_directional],
                        "student_confidence": -conf["student"][teacher_directional],
                        "confidence_gap": (conf["teacher"] - conf["student"])[teacher_directional],
                        "js": js_st[teacher_directional],
                        "student_entropy": entropy["student"][teacher_directional],
                        "confidence_plus_student": (conf["teacher"] - conf["student"])[teacher_directional],
                    }
                    peer_scores = {
                        "confidence": conf["peer"][peer_directional],
                        "student_confidence": -conf["student"][peer_directional],
                        "confidence_gap": (conf["peer"] - conf["student"])[peer_directional],
                        "js": js_sp[peer_directional],
                        "student_entropy": entropy["student"][peer_directional],
                        "confidence_plus_student": (conf["peer"] - conf["student"])[peer_directional],
                    }
                    row = {
                        "iteration": iteration, "case_id": case["case_id"], "region": region_name,
                        "pixels": int(region.sum()),
                        "student_confidence": _safe_mean(conf["student"], region),
                        "teacher_confidence": _safe_mean(conf["teacher"], region),
                        "peer_confidence": _safe_mean(conf["peer"], region),
                        "student_entropy": _safe_mean(entropy["student"], region),
                        "teacher_entropy": _safe_mean(entropy["teacher"], region),
                        "peer_entropy": _safe_mean(entropy["peer"], region),
                        "student_margin": _safe_mean(margin["student"], region),
                        "teacher_margin": _safe_mean(margin["teacher"], region),
                        "peer_margin": _safe_mean(margin["peer"], region),
                        "student_teacher_js": _safe_mean(js_st, region),
                        "student_peer_js": _safe_mean(js_sp, region),
                        "teacher_peer_js": _safe_mean(js_tp, region),
                        "teacher_student_conf_gap": _safe_mean(conf["teacher"] - conf["student"], region),
                        "peer_student_conf_gap": _safe_mean(conf["peer"] - conf["student"], region),
                        "teacher_direction_positive_pixels": int(teacher_label.sum()),
                        "teacher_direction_negative_pixels": int(teacher_label.size - teacher_label.sum()),
                        "peer_direction_positive_pixels": int(peer_label.sum()),
                        "peer_direction_negative_pixels": int(peer_label.size - peer_label.sum()),
                    }
                    for score_name, score in teacher_scores.items():
                        auc, ap = _binary_metrics(teacher_label, score)
                        row["teacher_%s_auroc" % score_name] = auc
                        row["teacher_%s_auprc" % score_name] = ap
                    for score_name, score in peer_scores.items():
                        auc, ap = _binary_metrics(peer_label, score)
                        row["peer_%s_auroc" % score_name] = auc
                        row["peer_%s_auprc" % score_name] = ap
                    stage_rows.append(row)
            self._append_tsv(self.stage_path, list(stage_rows[0]), stage_rows)

        dense_path = None
        reasons = []
        if iteration in KEY_ITERS:
            reasons.append("key")
        if student_best_refreshed:
            reasons.append("best")
        hashes = {}
        if reasons:
            hashes = {name: state_sha256(model) for name, model in zip(MODEL_NAMES, models)}
            dense_path = self._save_dense(iteration, "+".join(reasons), cases, hashes)
        event = {
            "iteration": iteration, "finite": finite, "rng_unchanged": rng_unchanged,
            "full_audit": full, "dense_path": dense_path, "model_sha256": hashes,
            "student_best_refreshed": bool(student_best_refreshed),
        }
        with open(self.event_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if not finite or not rng_unchanged:
            raise RuntimeError("RGPA observation invariant failed: finite=%s rng_unchanged=%s" %
                               (finite, rng_unchanged))
        return event
