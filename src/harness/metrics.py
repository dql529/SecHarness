"""
metrics.py — Compute all evaluation metrics required by the paper.

Detection metrics (per-class + macro):
  Accuracy, Precision, Recall, F1, FPR

Harness-specific metrics:
  Disagreement Rate, Escalation Rate

Zero-day metrics:
  Zero-Day Detection Rate (recall on held-out attack classes)

Efficiency metrics:
  Average latency, token consumption

Outputs CSV tables to project/results/tables/.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .observability import AuditRecord
from ..agents.consensus import ConsensusResult


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _binary_stats(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute TP, FP, FN, TN from binary labels (1=attack, 0=benign)."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Per-class + macro metrics from AuditRecords
# ---------------------------------------------------------------------------

def _extract_labels(
    records: List[AuditRecord], skip_escalate: bool = True
) -> Tuple[List[str], List[str]]:
    """Extract ground-truth and predicted labels from audit records.

    Ground truth: original label string (e.g. "Normal", "DoS", "Exploits").
    Prediction: "Normal" if benign, else consensus final_attack_type or "Attack".
    Escalated samples are skipped by default.
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    for r in records:
        if r.ground_truth_label is None:
            continue
        if skip_escalate and r.consensus_result.final_verdict == "escalate":
            continue
        y_true.append(r.ground_truth_label)
        if r.consensus_result.final_verdict == "benign":
            y_pred.append("Normal")
        else:
            y_pred.append(r.consensus_result.final_attack_type or "Attack")
    return y_true, y_pred


def compute_binary_metrics(records: List[AuditRecord]) -> Dict[str, float]:
    """Binary classification metrics (attack vs benign)."""
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    for r in records:
        if r.ground_truth_label is None:
            continue
        # Skip escalated samples — they have no definitive prediction
        if r.consensus_result.final_verdict == "escalate":
            continue
        gt_attack = 1 if r.ground_truth_label.lower() not in ("normal", "benign") else 0
        pred_attack = 1 if r.consensus_result.final_verdict == "attack" else 0
        y_true_bin.append(gt_attack)
        y_pred_bin.append(pred_attack)

    if not y_true_bin:
        return {}

    stats = _binary_stats(y_true_bin, y_pred_bin)
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)
    total = tp + fp + fn + tn
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def compute_perclass_metrics(records: List[AuditRecord]) -> Dict[str, Dict[str, float]]:
    """Per-class (one-vs-rest) Precision, Recall, F1.

    Returns dict keyed by class label, plus a "macro" entry.
    """
    y_true, y_pred = _extract_labels(records)
    if not y_true:
        return {}

    classes = sorted(set(y_true) | set(y_pred))
    per_class: Dict[str, Dict[str, float]] = {}

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        support = sum(1 for t in y_true if t == cls)
        per_class[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }

    # Macro average
    n_classes = len(classes)
    per_class["macro"] = {
        "precision": sum(v["precision"] for v in per_class.values()) / n_classes,
        "recall": sum(v["recall"] for v in per_class.values()) / n_classes,
        "f1": sum(v["f1"] for v in per_class.values()) / n_classes,
        "support": len(y_true),
    }
    return per_class


# ---------------------------------------------------------------------------
# Zero-day detection rate
# ---------------------------------------------------------------------------

def compute_zeroday_rate(
    records: List[AuditRecord],
    zeroday_classes: List[str],
) -> Dict[str, float]:
    """Recall on samples whose ground-truth label is in zeroday_classes."""
    zd_set = {c.lower() for c in zeroday_classes}
    zd_records = [
        r for r in records
        if r.ground_truth_label is not None
        and r.ground_truth_label.lower() in zd_set
    ]
    if not zd_records:
        return {"zeroday_total": 0, "zeroday_detected": 0, "zeroday_detection_rate": 0.0}

    detected = sum(1 for r in zd_records if r.consensus_result.final_verdict == "attack")
    return {
        "zeroday_total": len(zd_records),
        "zeroday_detected": detected,
        "zeroday_detection_rate": detected / len(zd_records),
    }


# ---------------------------------------------------------------------------
# Harness-specific metrics
# ---------------------------------------------------------------------------

def compute_harness_metrics(records: List[AuditRecord]) -> Dict[str, float]:
    """Disagreement rate, escalation rate, audit completeness."""
    if not records:
        return {}
    n = len(records)
    ctype_counts = Counter(r.consensus_result.consensus_type for r in records)

    disagree = sum(1 for r in records if r.consensus_result.is_disagreement)
    escalate = sum(1 for r in records if r.consensus_result.final_verdict == "escalate")
    has_gt = sum(1 for r in records if r.ground_truth_label is not None)

    return {
        "disagreement_rate": disagree / n,
        "escalation_rate": escalate / n,
        "audit_completeness": has_gt / n,
        "total_samples": n,
        **{f"consensus_{k}": v for k, v in sorted(ctype_counts.items())},
    }


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def compute_efficiency_metrics(records: List[AuditRecord]) -> Dict[str, float]:
    """Average latency and token consumption."""
    if not records:
        return {}
    n = len(records)
    return {
        "avg_total_latency_ms": sum(r.latency.total_ms for r in records) / n,
        "avg_alpha_latency_ms": sum(r.latency.alpha_ms for r in records) / n,
        "avg_beta_latency_ms": sum(r.latency.beta_ms for r in records) / n,
        "avg_consensus_latency_ms": sum(r.latency.consensus_ms for r in records) / n,
        "avg_alpha_tokens": sum(r.token_usage.alpha_tokens for r in records) / n,
        "avg_beta_tokens": sum(r.token_usage.beta_tokens for r in records) / n,
        "avg_total_tokens": sum(r.token_usage.total_tokens for r in records) / n,
        "p50_total_latency_ms": _percentile([r.latency.total_ms for r in records], 0.5),
        "p95_total_latency_ms": _percentile([r.latency.total_ms for r in records], 0.95),
        "p99_total_latency_ms": _percentile([r.latency.total_ms for r in records], 0.99),
    }


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p)
    idx = min(idx, len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_all_metrics(
    records: List[AuditRecord],
    experiment_id: str,
    dataset: str,
    output_dir: str | Path,
    zeroday_classes: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Compute all metrics and export to CSV files.

    Returns dict of {metric_group: csv_path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_id}_{dataset}"
    exported: Dict[str, Path] = {}

    # 1. Binary metrics
    binary = compute_binary_metrics(records)
    if binary:
        p = output_dir / f"{prefix}_binary_metrics.csv"
        _write_flat_csv(p, binary)
        exported["binary"] = p

    # 2. Per-class metrics
    perclass = compute_perclass_metrics(records)
    if perclass:
        p = output_dir / f"{prefix}_perclass_metrics.csv"
        _write_perclass_csv(p, perclass)
        exported["perclass"] = p

    # 3. Zero-day
    if zeroday_classes:
        zd = compute_zeroday_rate(records, zeroday_classes)
        p = output_dir / f"{prefix}_zeroday_metrics.csv"
        _write_flat_csv(p, zd)
        exported["zeroday"] = p

    # 4. Harness metrics
    harness = compute_harness_metrics(records)
    if harness:
        p = output_dir / f"{prefix}_harness_metrics.csv"
        _write_flat_csv(p, harness)
        exported["harness"] = p

    # 5. Efficiency
    efficiency = compute_efficiency_metrics(records)
    if efficiency:
        p = output_dir / f"{prefix}_efficiency_metrics.csv"
        _write_flat_csv(p, efficiency)
        exported["efficiency"] = p

    return exported


def _write_flat_csv(path: Path, data: Dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in data.items():
            w.writerow([k, v])


def _write_perclass_csv(path: Path, data: Dict[str, Dict]) -> None:
    cols = ["class", "precision", "recall", "f1", "support"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cls, metrics in data.items():
            w.writerow([cls, metrics["precision"], metrics["recall"], metrics["f1"], metrics["support"]])
