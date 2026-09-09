#!/usr/bin/env python3
"""
C12 Part B: Consensus Strategy Optimization

Re-applies different consensus strategies to E5 audit logs (no re-inference).
Strategies:
  1. confidence_weighted: trust the higher-confidence agent on disagreement
  2. adaptive: context-dependent rules per disagreement type
  3. conservative: AND logic (both must agree on attack)

Compares with E4/E5(alpha_priority)/E2(ML-only) baselines.

Usage:
    python evaluate_consensus_strategies.py
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# Paths
PROJECT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT / "results" / "tables"
FIGURES_DIR = PROJECT / "results" / "figures"

# Defaults (UNSW)
DEFAULT_AUDIT_LOG = PROJECT / "logs" / "E5_unsw_audit.jsonl"
DEFAULT_ZERODAY_CLASSES = {"shellcode", "worms"}
DEFAULT_OUTPUT_PREFIX = "unsw"

# Module-level variable set by main()
ZERODAY_CLASSES = DEFAULT_ZERODAY_CLASSES


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_attack_gt(label: str) -> bool:
    return label.lower() not in ("normal", "benign")


def is_zeroday_gt(label: str) -> bool:
    return label.lower() in ZERODAY_CLASSES


def compute_metrics(records: list[dict], final_verdicts: list[str]) -> dict:
    """Compute binary metrics from a list of final verdicts."""
    tp = fp = fn = tn = 0
    zd_total = zd_detected = 0
    escalated = 0

    for r, verdict in zip(records, final_verdicts):
        gt = r["ground_truth_label"]
        if gt is None:
            continue

        gt_attack = is_attack_gt(gt)
        gt_zd = is_zeroday_gt(gt)

        if verdict == "escalate":
            escalated += 1
            # Count zero-day totals even for escalated
            if gt_zd:
                zd_total += 1
            continue

        pred_attack = verdict == "attack"

        if gt_attack and pred_attack:
            tp += 1
        elif not gt_attack and pred_attack:
            fp += 1
        elif gt_attack and not pred_attack:
            fn += 1
        else:
            tn += 1

        if gt_zd:
            zd_total += 1
            if pred_attack:
                zd_detected += 1

    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc = (tp + tn) / total if total > 0 else 0
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "zeroday_rate": zd_rate,
        "zeroday_detected": zd_detected,
        "zeroday_total": zd_total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_evaluated": total,
        "escalated": escalated,
        "escalation_rate": escalated / (total + escalated) if (total + escalated) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def strategy_alpha_priority(records: list[dict]) -> list[str]:
    """E5 original: always trust Alpha on disagreement."""
    verdicts = []
    for r in records:
        ctype = r["consensus_result"]["consensus_type"]
        if ctype == "agreement":
            verdicts.append(r["consensus_result"]["final_verdict"])
        elif ctype == "uncertain":
            verdicts.append("escalate")
        else:
            # Disagreement → Alpha wins
            verdicts.append(r["alpha_verdict"]["verdict"])
    return verdicts


def strategy_beta_priority(records: list[dict]) -> list[str]:
    """Always trust Beta (ML) on disagreement."""
    verdicts = []
    for r in records:
        ctype = r["consensus_result"]["consensus_type"]
        if ctype == "agreement":
            verdicts.append(r["consensus_result"]["final_verdict"])
        elif ctype == "uncertain":
            verdicts.append("escalate")
        else:
            verdicts.append(r["beta_verdict"]["verdict"])
    return verdicts


def strategy_confidence_weighted(records: list[dict]) -> list[str]:
    """Trust the agent with higher confidence on disagreement."""
    verdicts = []
    for r in records:
        ctype = r["consensus_result"]["consensus_type"]
        if ctype == "agreement":
            verdicts.append(r["consensus_result"]["final_verdict"])
        elif ctype == "uncertain":
            verdicts.append("escalate")
        else:
            alpha_conf = r["alpha_verdict"]["confidence"]
            beta_conf = r["beta_verdict"]["confidence"]
            if alpha_conf >= beta_conf:
                verdicts.append(r["alpha_verdict"]["verdict"])
            else:
                verdicts.append(r["beta_verdict"]["verdict"])
    return verdicts


def strategy_adaptive(records: list[dict]) -> list[str]:
    """Context-dependent adaptive strategy.

    - agreement → direct output
    - alpha_only (Alpha=attack, Beta=benign):
        if Alpha confidence > 0.7 → trust Alpha (possible zero-day)
        else → trust Beta (possible LLM hallucination)
    - beta_only (Alpha=benign, Beta=attack):
        always trust Beta (ML pattern matching more reliable)
    - type_conflict: trust Beta's type (ML classification more accurate)
    - uncertain: escalate
    """
    verdicts = []
    for r in records:
        ctype = r["consensus_result"]["consensus_type"]

        if ctype == "agreement":
            verdicts.append(r["consensus_result"]["final_verdict"])
        elif ctype == "uncertain":
            verdicts.append("escalate")
        elif ctype == "alpha_only":
            # Alpha=attack, Beta=benign
            alpha_conf = r["alpha_verdict"]["confidence"]
            if alpha_conf > 0.7:
                verdicts.append("attack")
            else:
                verdicts.append("benign")
        elif ctype == "beta_only":
            # Alpha=benign, Beta=attack → trust Beta
            verdicts.append("attack")
        elif ctype == "type_conflict":
            # Both attack, different type → still attack, trust Beta type
            verdicts.append("attack")
        else:
            verdicts.append(r["consensus_result"]["final_verdict"])
    return verdicts


def strategy_conservative(records: list[dict]) -> list[str]:
    """AND logic: only attack if both agents agree.

    - agreement → direct output
    - disagreement (one says attack, one says benign) → benign (flagged)
    - type_conflict (both attack, different type) → attack
    - uncertain → escalate
    """
    verdicts = []
    for r in records:
        ctype = r["consensus_result"]["consensus_type"]

        if ctype == "agreement":
            verdicts.append(r["consensus_result"]["final_verdict"])
        elif ctype == "uncertain":
            verdicts.append("escalate")
        elif ctype == "type_conflict":
            # Both say attack → attack
            verdicts.append("attack")
        elif ctype in ("alpha_only", "beta_only"):
            # One says attack, one benign → benign (conservative)
            verdicts.append("benign")
        else:
            verdicts.append(r["consensus_result"]["final_verdict"])
    return verdicts


def strategy_escalate_all(records: list[dict]) -> list[str]:
    """Escalate all disagreements for human review."""
    verdicts = []
    for r in records:
        ctype = r["consensus_result"]["consensus_type"]
        if ctype == "agreement":
            verdicts.append(r["consensus_result"]["final_verdict"])
        else:
            verdicts.append("escalate")
    return verdicts


# ---------------------------------------------------------------------------
# Baselines from prior experiments
# ---------------------------------------------------------------------------

BASELINES = {
    "E2_ML_only": {
        "accuracy": 0.9213, "precision": 0.9413, "recall": 0.9391,
        "f1": 0.9402, "fpr": 0.1132, "zeroday_rate": 0.968,
        "escalated": 0, "escalation_rate": 0.0,
    },
    "E4_NoHarness": {
        "accuracy": 0.8880, "precision": 0.9247, "recall": 0.9014,
        "f1": 0.9129, "fpr": 0.1369, "zeroday_rate": 0.824,
        "escalated": 0, "escalation_rate": 0.0,
    },
}


def export_comparison_csv(all_results: dict, output_path: Path):
    """Export strategy comparison to CSV."""
    metrics_keys = ["accuracy", "precision", "recall", "f1", "fpr",
                    "zeroday_rate", "zeroday_detected", "zeroday_total",
                    "tp", "fp", "fn", "tn", "total_evaluated",
                    "escalated", "escalation_rate"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["strategy"] + metrics_keys
        w.writerow(header)
        for name, metrics in all_results.items():
            row = [name] + [metrics.get(k, "") for k in metrics_keys]
            w.writerow(row)
    print(f"  Strategy comparison CSV saved: {output_path}")


def generate_strategy_comparison_chart(all_results: dict, output_path: Path):
    """Bar chart comparing key metrics across strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Select strategies to plot
    strategies = list(all_results.keys())
    metrics_to_plot = ["accuracy", "f1", "fpr", "zeroday_rate"]
    metric_labels = ["Accuracy", "F1 Score", "FPR ↓", "Zero-day Rate"]

    x = np.arange(len(strategies))
    width = 0.18
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        values = [all_results[s].get(metric, 0) for s in strategies]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies],
                        fontsize=9, ha="center")
    ax.set_ylabel("Score", fontsize=12)
    n_samples = sum(next((v for v in all_results.values() if "tp" in v), {}).get(k, 0) for k in ("tp", "fp", "fn", "tn"))
    ax.set_title(f"Consensus Strategy Comparison ({n_samples} samples)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.2)

    # Add escalation rate as text annotation
    for i, s in enumerate(strategies):
        esc = all_results[s].get("escalation_rate", 0)
        if esc > 0:
            ax.text(i, -0.08, f"esc={esc:.1%}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Strategy comparison chart saved: {output_path}")


def main():
    global ZERODAY_CLASSES

    ap = argparse.ArgumentParser(description="C12 Part B: Consensus Strategy Optimization")
    ap.add_argument("--audit-log", type=str, default=None,
                     help="Path to E5 audit JSONL (default: E5_unsw_audit.jsonl)")
    ap.add_argument("--zeroday-classes", type=str, default=None,
                     help="Comma-separated zero-day classes (default: shellcode,worms)")
    ap.add_argument("--output-prefix", type=str, default=None,
                     help="Output file prefix/suffix (default: unsw)")
    ap.add_argument("--baselines-json", type=str, default=None,
                     help="JSON file with baseline metrics (E2/E4)")
    args = ap.parse_args()

    # Resolve parameters
    audit_log = Path(args.audit_log) if args.audit_log else DEFAULT_AUDIT_LOG
    if args.zeroday_classes:
        ZERODAY_CLASSES = set(c.strip().lower() for c in args.zeroday_classes.split(","))
    else:
        ZERODAY_CLASSES = DEFAULT_ZERODAY_CLASSES
    output_prefix = args.output_prefix or DEFAULT_OUTPUT_PREFIX

    print("=" * 60)
    print("C12 Part B: Consensus Strategy Optimization")
    print(f"  Dataset: {output_prefix}")
    print(f"  Zero-day classes: {ZERODAY_CLASSES}")
    print("=" * 60)

    # Load data
    print(f"\nLoading audit log: {audit_log}")
    records = load_records(audit_log)
    print(f"  Total records: {len(records)}")

    # Define strategies
    strategies = {
        "E5_alpha_priority": strategy_alpha_priority,
        "E5_beta_priority": strategy_beta_priority,
        "E5_confidence_weighted": strategy_confidence_weighted,
        "E5_adaptive": strategy_adaptive,
        "E5_conservative": strategy_conservative,
        "E5_escalate_all": strategy_escalate_all,
    }

    # Evaluate each strategy
    all_results = {}

    # Add baselines
    if args.baselines_json and Path(args.baselines_json).exists():
        import json as _json
        with open(args.baselines_json) as _f:
            baselines = _json.load(_f)
        for k, v in baselines.items():
            all_results[k] = v
    else:
        all_results["E2_ML_only"] = BASELINES["E2_ML_only"]
        all_results["E4_NoHarness"] = BASELINES["E4_NoHarness"]

    for name, strategy_fn in strategies.items():
        verdicts = strategy_fn(records)
        metrics = compute_metrics(records, verdicts)
        all_results[name] = metrics

    # Print results
    print("\n" + "=" * 100)
    print(f"{'Strategy':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} "
          f"{'FPR':>7s} {'ZD-Rate':>8s} {'Esc':>6s} {'Esc%':>7s}")
    print("-" * 100)

    for name, m in all_results.items():
        print(f"{name:<25s} "
              f"{m.get('accuracy', 0):>7.4f} "
              f"{m.get('precision', 0):>7.4f} "
              f"{m.get('recall', 0):>7.4f} "
              f"{m.get('f1', 0):>7.4f} "
              f"{m.get('fpr', 0):>7.4f} "
              f"{m.get('zeroday_rate', 0):>8.4f} "
              f"{m.get('escalated', 0):>6d} "
              f"{m.get('escalation_rate', 0):>7.1%}")
    print("=" * 100)

    # Identify best strategy per metric
    print("\n--- Best Strategy per Metric ---")
    for metric, direction in [("accuracy", "max"), ("f1", "max"),
                               ("fpr", "min"), ("zeroday_rate", "max")]:
        if direction == "max":
            best = max(all_results.items(), key=lambda x: x[1].get(metric, -1))
        else:
            # For FPR, exclude strategies with high escalation (unfair comparison)
            candidates = {k: v for k, v in all_results.items()
                          if v.get("escalation_rate", 0) < 0.5}
            best = min(candidates.items(), key=lambda x: x[1].get(metric, 999))
        print(f"  {metric:>12s}: {best[0]} = {best[1].get(metric, 0):.4f}")

    # Export
    print("\n--- Exporting Results ---")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    export_comparison_csv(all_results, TABLES_DIR / f"E5_strategy_comparison_{output_prefix}.csv")
    generate_strategy_comparison_chart(all_results,
                                        FIGURES_DIR / f"RQ3_consensus_strategy_comparison_{output_prefix}.png")

    print("\n[OK] Part B complete.")


if __name__ == "__main__":
    main()
