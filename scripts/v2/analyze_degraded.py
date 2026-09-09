#!/usr/bin/env python3
"""
analyze_degraded.py — E3 vs E3-degraded comparison analysis.

Compares tool usage, chain lengths, self-correction, and per-sample behavior.

Usage:
    python project/scripts/v2/analyze_degraded.py \
        --e3-log project/logs/v2/E3_unsw_sub1000_audit.jsonl \
        --e3deg-log project/logs/v2/E3_degraded_unsw_sub1000_audit.jsonl \
        --output-dir project/results/
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("degraded-analysis")


def load_audit_log(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), path)
    return records


def compute_basic_metrics(records):
    tp = fp = fn = tn = 0
    zd_total = zd_detected = 0
    for r in records:
        ev = r.get("evaluation", {})
        gt = ev.get("binary_gt", "")
        pred = ev.get("binary_pred", "")
        is_zd = r.get("input", {}).get("is_zeroday", False)
        if gt == "attack" and pred == "attack": tp += 1
        elif gt == "benign" and pred == "attack": fp += 1
        elif gt == "attack" and pred == "benign": fn += 1
        else: tn += 1
        if is_zd:
            zd_total += 1
            if pred == "attack": zd_detected += 1
    n = len(records)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    zd_rate = zd_detected / zd_total if zd_total else 0
    return {"n": n, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr, "zd_total": zd_total, "zd_detected": zd_detected, "zd_rate": zd_rate}


def analyze_tool_usage(records):
    """Analyze per-sample tool usage patterns."""
    tool_counts = Counter()
    tool_sample_counts = Counter()
    chain_lengths = []
    behavior_categories = Counter()
    behavior_correct = defaultdict(lambda: [0, 0])  # [correct, total]

    for r in records:
        tc = r.get("tool_chain", [])
        tools_in_chain = [c.get("tool", "") for c in tc]
        unique_tools = set(tools_in_chain) - {"classify"}

        chain_len = len(tc)
        chain_lengths.append(chain_len)

        for t in tools_in_chain:
            tool_counts[t] += 1
        for t in unique_tools:
            tool_sample_counts[t] += 1

        # Behavior classification
        ev = r.get("evaluation", {})
        correct = 1 if ev.get("binary_gt") == ev.get("binary_pred") else 0

        if len(unique_tools) == 0:
            cat = "no-tool"
        elif unique_tools == {"check_anomaly"}:
            cat = "single-tool"
        elif "load_knowledge" in unique_tools:
            cat = "knowledge-augmented"
        elif "lookup_signature" in unique_tools:
            cat = "signature-matched"
        elif len(unique_tools) >= 2:
            cat = "multi-tool"
        else:
            cat = "single-tool"

        behavior_categories[cat] += 1
        behavior_correct[cat][0] += correct
        behavior_correct[cat][1] += 1

    n = len(records)
    return {
        "tool_counts": dict(tool_counts),
        "tool_sample_rates": {t: c / n for t, c in tool_sample_counts.items()},
        "avg_chain_length": np.mean(chain_lengths) if chain_lengths else 0,
        "std_chain_length": np.std(chain_lengths) if chain_lengths else 0,
        "behavior_categories": dict(behavior_categories),
        "behavior_accuracy": {
            cat: (vals[0] / vals[1] if vals[1] > 0 else 0)
            for cat, vals in behavior_correct.items()
        },
    }


def analyze_self_correction(records):
    """Check for self-correction: agent changes classification after tool calls."""
    corrections = 0
    for r in records:
        if r.get("result", {}).get("self_corrected", False):
            corrections += 1
    return corrections, corrections / len(records) if records else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3-log", required=True)
    ap.add_argument("--e3deg-log", required=True)
    ap.add_argument("--output-dir", default="project/results/")
    args = ap.parse_args()

    e3 = load_audit_log(args.e3_log)
    e3d = load_audit_log(args.e3deg_log)

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables" / "v2"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance comparison
    m3 = compute_basic_metrics(e3)
    m3d = compute_basic_metrics(e3d)

    log.info("=== Performance Comparison ===")
    log.info("         E3       E3-degraded  Delta")
    for k in ["accuracy", "f1", "fpr", "zd_rate"]:
        log.info("  %s: %.4f    %.4f    %+.4f", k, m3[k], m3d[k], m3d[k] - m3[k])

    # 2. Tool usage comparison
    tu3 = analyze_tool_usage(e3)
    tu3d = analyze_tool_usage(e3d)

    log.info("=== Tool Usage Comparison ===")
    log.info("E3 avg chain: %.2f ± %.2f", tu3["avg_chain_length"], tu3["std_chain_length"])
    log.info("E3d avg chain: %.2f ± %.2f", tu3d["avg_chain_length"], tu3d["std_chain_length"])

    all_tools = sorted(set(tu3["tool_sample_rates"].keys()) | set(tu3d["tool_sample_rates"].keys()))
    log.info("Tool sample rates:")
    for t in all_tools:
        r3 = tu3["tool_sample_rates"].get(t, 0)
        r3d = tu3d["tool_sample_rates"].get(t, 0)
        log.info("  %s: E3=%.1f%% E3d=%.1f%%", t, r3 * 100, r3d * 100)

    log.info("Behavior categories:")
    all_cats = sorted(set(tu3["behavior_categories"].keys()) | set(tu3d["behavior_categories"].keys()))
    for cat in all_cats:
        c3 = tu3["behavior_categories"].get(cat, 0)
        c3d = tu3d["behavior_categories"].get(cat, 0)
        a3 = tu3["behavior_accuracy"].get(cat, 0)
        a3d = tu3d["behavior_accuracy"].get(cat, 0)
        log.info("  %s: E3=%d(%.1f%% acc) E3d=%d(%.1f%% acc)", cat, c3, a3 * 100, c3d, a3d * 100)

    # 3. Self-correction
    sc3, sr3 = analyze_self_correction(e3)
    sc3d, sr3d = analyze_self_correction(e3d)
    log.info("Self-correction: E3=%d(%.1f%%) E3d=%d(%.1f%%)", sc3, sr3 * 100, sc3d, sr3d * 100)

    # Save comparison CSV
    csv_path = tables_dir / "v2_degraded_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "E3", "E3_degraded", "Delta"])
        for k in ["accuracy", "precision", "recall", "f1", "fpr", "zd_rate"]:
            w.writerow([k, round(m3[k], 4), round(m3d[k], 4), round(m3d[k] - m3[k], 4)])
        w.writerow(["avg_chain_length", round(tu3["avg_chain_length"], 2), round(tu3d["avg_chain_length"], 2),
                     round(tu3d["avg_chain_length"] - tu3["avg_chain_length"], 2)])
        w.writerow(["self_correction_rate", round(sr3, 4), round(sr3d, 4), round(sr3d - sr3, 4)])
        w.writerow([])
        w.writerow(["Tool Sample Rate", "E3", "E3_degraded", "Delta"])
        for t in all_tools:
            r3 = tu3["tool_sample_rates"].get(t, 0)
            r3d = tu3d["tool_sample_rates"].get(t, 0)
            w.writerow([t, round(r3, 4), round(r3d, 4), round(r3d - r3, 4)])
        w.writerow([])
        w.writerow(["Behavior Category", "E3_count", "E3_acc", "E3d_count", "E3d_acc"])
        for cat in all_cats:
            w.writerow([cat,
                        tu3["behavior_categories"].get(cat, 0), round(tu3["behavior_accuracy"].get(cat, 0), 4),
                        tu3d["behavior_categories"].get(cat, 0), round(tu3d["behavior_accuracy"].get(cat, 0), 4)])
    log.info("CSV saved to %s", csv_path)

    # Generate tool usage comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Tool sample rates
    ax = axes[0]
    analysis_tools = [t for t in all_tools if t not in ("classify", "escalate", "log_decision")]
    x = np.arange(len(analysis_tools))
    width = 0.35
    e3_rates = [tu3["tool_sample_rates"].get(t, 0) * 100 for t in analysis_tools]
    e3d_rates = [tu3d["tool_sample_rates"].get(t, 0) * 100 for t in analysis_tools]
    bars1 = ax.bar(x - width / 2, e3_rates, width, label="E3 (full)", color="#2196F3")
    bars2 = ax.bar(x + width / 2, e3d_rates, width, label="E3-degraded", color="#FF9800")
    ax.set_ylabel("% of samples using tool")
    ax.set_title("(a) Tool Usage Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(analysis_tools, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=8)

    # Panel B: Behavior categories
    ax = axes[1]
    x = np.arange(len(all_cats))
    e3_counts = [tu3["behavior_categories"].get(cat, 0) for cat in all_cats]
    e3d_counts = [tu3d["behavior_categories"].get(cat, 0) for cat in all_cats]
    bars1 = ax.bar(x - width / 2, e3_counts, width, label="E3 (full)", color="#2196F3")
    bars2 = ax.bar(x + width / 2, e3d_counts, width, label="E3-degraded", color="#FF9800")
    ax.set_ylabel("Number of samples")
    ax.set_title("(b) Behavior Categories")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    fig_path = figures_dir / "v2_tool_usage_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure saved to %s", fig_path)

    # Also save JSON summary
    json_path = tables_dir / "v2_degraded_comparison.json"
    summary = {
        "e3_metrics": m3,
        "e3d_metrics": m3d,
        "e3_tool_usage": {k: v for k, v in tu3.items() if k != "tool_counts"},
        "e3d_tool_usage": {k: v for k, v in tu3d.items() if k != "tool_counts"},
        "e3_self_correction": {"count": sc3, "rate": sr3},
        "e3d_self_correction": {"count": sc3d, "rate": sr3d},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("JSON saved to %s", json_path)


if __name__ == "__main__":
    main()
