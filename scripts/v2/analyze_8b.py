#!/usr/bin/env python3
"""
analyze_8b.py — 3B vs 8B model scaling comparison.

Compares tool usage behavior, chain lengths, accuracy, and zero-day detection
between 3B (Llama-3.2-3B) and 7B/8B (Qwen2.5-7B) experiments.

Usage:
    python project/scripts/v2/analyze_8b.py \
        --e3-log project/logs/v2/E3_unsw_sub200_audit.jsonl \
        --e3-8b-log project/logs/v2/E3_8B_unsw_sub200_audit.jsonl \
        [--e3deg-log project/logs/v2/E3_degraded_unsw_sub1000_audit.jsonl] \
        [--e3-8b-deg-log project/logs/v2/E3_8B_degraded_unsw_sub200_audit.jsonl] \
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
log = logging.getLogger("8b-analysis")


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
            "fpr": fpr, "zd_total": zd_total, "zd_detected": zd_detected, "zd_rate": zd_rate,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def analyze_tool_usage(records):
    """Analyze per-sample tool usage patterns."""
    tool_counts = Counter()
    tool_sample_counts = Counter()
    chain_lengths = []
    behavior_categories = Counter()
    behavior_correct = defaultdict(lambda: [0, 0])  # [correct, total]
    latencies = []
    termination_counts = Counter()
    parse_fails = 0

    for r in records:
        tc = r.get("tool_chain", [])
        tools_in_chain = [c.get("tool", "") for c in tc]
        unique_tools = set(tools_in_chain) - {"classify", "escalate", "log_decision"}

        chain_len = len(tc)
        chain_lengths.append(chain_len)

        for t in tools_in_chain:
            tool_counts[t] += 1
        for t in unique_tools:
            tool_sample_counts[t] += 1

        # Latency
        lat = r.get("efficiency", {}).get("total_latency_ms", 0)
        latencies.append(lat)

        # Termination
        term = r.get("result", {}).get("termination", "unknown")
        termination_counts[term] += 1
        if term == "parse_fail":
            parse_fails += 1

        # Behavior classification
        ev = r.get("evaluation", {})
        correct = 1 if ev.get("binary_gt") == ev.get("binary_pred") else 0

        if len(unique_tools) == 0:
            cat = "no-tool"
        elif unique_tools == {"check_anomaly"}:
            cat = "single-tool"
        elif "load_knowledge" in unique_tools or "lookup_signature" in unique_tools:
            cat = "multi-tool-knowledge"
        elif len(unique_tools) >= 2:
            cat = "multi-tool"
        else:
            cat = "single-tool"

        behavior_categories[cat] += 1
        behavior_correct[cat][0] += correct
        behavior_correct[cat][1] += 1

    n = len(records)
    # Compute chain length distribution
    chain_dist = Counter(chain_lengths)

    return {
        "tool_counts": dict(tool_counts),
        "tool_sample_rates": {t: c / n for t, c in tool_sample_counts.items()},
        "avg_chain_length": float(np.mean(chain_lengths)) if chain_lengths else 0,
        "std_chain_length": float(np.std(chain_lengths)) if chain_lengths else 0,
        "max_chain_length": max(chain_lengths) if chain_lengths else 0,
        "chain_length_dist": dict(chain_dist),
        "behavior_categories": dict(behavior_categories),
        "behavior_accuracy": {
            cat: (vals[0] / vals[1] if vals[1] > 0 else 0)
            for cat, vals in behavior_correct.items()
        },
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
        "termination_counts": dict(termination_counts),
        "parse_fail_rate": parse_fails / n if n > 0 else 0,
    }


def analyze_self_correction(records):
    corrections = 0
    for r in records:
        if r.get("result", {}).get("self_corrected", False):
            corrections += 1
    return corrections, corrections / len(records) if records else 0


def analyze_multistep_samples(records):
    """Analyze samples where chain_length > 2 (multi-step reasoning)."""
    multistep = []
    singlestep = []
    for r in records:
        tc = r.get("tool_chain", [])
        unique_analysis_tools = set(c.get("tool", "") for c in tc) - {"classify", "escalate", "log_decision"}
        ev = r.get("evaluation", {})
        is_zd = r.get("input", {}).get("is_zeroday", False)
        correct = ev.get("binary_gt") == ev.get("binary_pred")
        entry = {
            "chain_length": len(tc),
            "tools": list(unique_analysis_tools),
            "tool_sequence": [c.get("tool", "") for c in tc],
            "is_zeroday": is_zd,
            "correct": correct,
            "gt": ev.get("binary_gt", ""),
            "pred": ev.get("binary_pred", ""),
        }
        if len(unique_analysis_tools) > 1:
            multistep.append(entry)
        else:
            singlestep.append(entry)

    result = {
        "multistep_count": len(multistep),
        "singlestep_count": len(singlestep),
    }
    if multistep:
        result["multistep_accuracy"] = sum(1 for e in multistep if e["correct"]) / len(multistep)
        result["multistep_zd_rate"] = sum(1 for e in multistep if e["is_zeroday"]) / len(multistep) if multistep else 0
        # Most common tool sequences
        seq_counter = Counter(tuple(e["tool_sequence"]) for e in multistep)
        result["top_sequences"] = seq_counter.most_common(10)
    if singlestep:
        result["singlestep_accuracy"] = sum(1 for e in singlestep if e["correct"]) / len(singlestep)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3-log", required=True, help="E3 3B audit log")
    ap.add_argument("--e3-8b-log", required=True, help="E3-8B audit log")
    ap.add_argument("--e3deg-log", default=None, help="E3-degraded 3B audit log")
    ap.add_argument("--e3-8b-deg-log", default=None, help="E3-8B-degraded audit log")
    ap.add_argument("--output-dir", default="project/results/")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables" / "v2"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load logs
    experiments = {}
    experiments["E3 (3B)"] = load_audit_log(args.e3_log)
    experiments["E3-8B (7B)"] = load_audit_log(args.e3_8b_log)
    if args.e3deg_log:
        experiments["E3-deg (3B)"] = load_audit_log(args.e3deg_log)
    if args.e3_8b_deg_log:
        experiments["E3-8B-deg (7B)"] = load_audit_log(args.e3_8b_deg_log)

    # Compute metrics for all experiments
    all_metrics = {}
    all_tool_usage = {}
    all_multistep = {}
    for name, records in experiments.items():
        all_metrics[name] = compute_basic_metrics(records)
        all_tool_usage[name] = analyze_tool_usage(records)
        all_multistep[name] = analyze_multistep_samples(records)

    # Print comparison table
    log.info("=" * 80)
    log.info("3B vs 7B/8B Model Scaling Comparison")
    log.info("=" * 80)

    header = f"{'Metric':<30}"
    for name in experiments:
        header += f" {name:>15}"
    log.info(header)
    log.info("-" * 80)

    for metric in ["accuracy", "f1", "fpr", "zd_rate"]:
        row = f"{metric:<30}"
        for name in experiments:
            row += f" {all_metrics[name][metric]:>15.4f}"
        log.info(row)

    log.info("-" * 80)
    for metric in ["avg_chain_length", "std_chain_length", "max_chain_length",
                    "parse_fail_rate", "avg_latency_ms"]:
        row = f"{metric:<30}"
        for name in experiments:
            row += f" {all_tool_usage[name][metric]:>15.2f}"
        log.info(row)

    log.info("-" * 80)
    all_tools = sorted(set().union(*(tu["tool_sample_rates"].keys() for tu in all_tool_usage.values())))
    analysis_tools = [t for t in all_tools if t not in ("classify", "escalate", "log_decision")]
    for t in analysis_tools:
        row = f"  tool: {t:<22}"
        for name in experiments:
            rate = all_tool_usage[name]["tool_sample_rates"].get(t, 0)
            row += f" {rate * 100:>14.1f}%"
        log.info(row)

    log.info("-" * 80)
    for name in experiments:
        sc_count, sc_rate = analyze_self_correction(experiments[name])
        log.info("  %s self-correction: %d (%.1f%%)", name, sc_count, sc_rate * 100)

    # Multi-step analysis
    log.info("=" * 80)
    log.info("Multi-step Reasoning Analysis")
    log.info("=" * 80)
    for name in experiments:
        ms = all_multistep[name]
        log.info("  %s: multi-step=%d, single-step=%d", name, ms["multistep_count"], ms["singlestep_count"])
        if ms["multistep_count"] > 0:
            log.info("    multi-step acc=%.1f%%, zd_rate=%.1f%%",
                     ms.get("multistep_accuracy", 0) * 100,
                     ms.get("multistep_zd_rate", 0) * 100)
            log.info("    top sequences: %s", ms.get("top_sequences", [])[:5])
        if ms["singlestep_count"] > 0:
            log.info("    single-step acc=%.1f%%", ms.get("singlestep_accuracy", 0) * 100)

    # Save CSV
    csv_path = tables_dir / "v2_8b_scaling_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        names = list(experiments.keys())
        w.writerow(["Metric"] + names)

        for metric in ["n", "accuracy", "precision", "recall", "f1", "fpr",
                       "zd_total", "zd_detected", "zd_rate"]:
            row = [metric]
            for name in names:
                val = all_metrics[name].get(metric, 0)
                row.append(round(val, 4) if isinstance(val, float) else val)
            w.writerow(row)

        w.writerow([])
        w.writerow(["Tool/Behavior Metric"] + names)
        for metric in ["avg_chain_length", "std_chain_length", "max_chain_length",
                       "parse_fail_rate", "avg_latency_ms"]:
            row = [metric]
            for name in names:
                row.append(round(all_tool_usage[name][metric], 4))
            w.writerow(row)

        for t in analysis_tools:
            row = [f"tool_{t}_rate"]
            for name in names:
                row.append(round(all_tool_usage[name]["tool_sample_rates"].get(t, 0), 4))
            w.writerow(row)

        sc_row = ["self_correction_rate"]
        for name in names:
            _, sr = analyze_self_correction(experiments[name])
            sc_row.append(round(sr, 4))
        w.writerow(sc_row)

        ms_row = ["multistep_sample_count"]
        for name in names:
            ms_row.append(all_multistep[name]["multistep_count"])
        w.writerow(ms_row)

    log.info("CSV saved to %s", csv_path)

    # Save JSON
    json_path = tables_dir / "v2_8b_scaling_comparison.json"
    summary = {}
    for name in experiments:
        key = name.replace(" ", "_").replace("(", "").replace(")", "")
        summary[key] = {
            "metrics": all_metrics[name],
            "tool_usage": {k: v for k, v in all_tool_usage[name].items() if k != "tool_counts"},
            "multistep": all_multistep[name],
            "self_correction": {"count": analyze_self_correction(experiments[name])[0],
                                "rate": analyze_self_correction(experiments[name])[1]},
        }
        # Make top_sequences JSON-serializable
        if "top_sequences" in summary[key]["multistep"]:
            summary[key]["multistep"]["top_sequences"] = [
                {"sequence": list(seq), "count": cnt}
                for seq, cnt in summary[key]["multistep"]["top_sequences"]
            ]
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("JSON saved to %s", json_path)

    # Generate comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names_short = ["3B", "7B"] if len(experiments) == 2 else ["3B", "7B", "3B-deg", "7B-deg"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    # Panel A: Performance metrics
    ax = axes[0]
    perf_metrics = ["accuracy", "f1", "zd_rate"]
    x = np.arange(len(perf_metrics))
    width = 0.8 / len(experiments)
    for i, (name, label) in enumerate(zip(experiments, names_short)):
        vals = [all_metrics[name][m] * 100 for m in perf_metrics]
        bars = ax.bar(x + i * width - width * len(experiments) / 2 + width / 2, vals,
                      width, label=label, color=colors[i])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "F1", "ZD Rate"])
    ax.legend()
    ax.set_ylim(0, 105)

    # Panel B: Tool usage rates
    ax = axes[1]
    x = np.arange(len(analysis_tools))
    width = 0.8 / len(experiments)
    for i, (name, label) in enumerate(zip(experiments, names_short)):
        rates = [all_tool_usage[name]["tool_sample_rates"].get(t, 0) * 100 for t in analysis_tools]
        ax.bar(x + i * width - width * len(experiments) / 2 + width / 2, rates,
               width, label=label, color=colors[i])
    ax.set_ylabel("% of samples using tool")
    ax.set_title("(b) Tool Usage Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(analysis_tools, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 110)

    # Panel C: Chain length distribution
    ax = axes[2]
    all_chain_keys = sorted(set().union(*(tu["chain_length_dist"].keys() for tu in all_tool_usage.values())))
    x = np.arange(len(all_chain_keys))
    width = 0.8 / len(experiments)
    for i, (name, label) in enumerate(zip(experiments, names_short)):
        n_total = all_metrics[name]["n"]
        counts = [all_tool_usage[name]["chain_length_dist"].get(k, 0) / n_total * 100 for k in all_chain_keys]
        ax.bar(x + i * width - width * len(experiments) / 2 + width / 2, counts,
               width, label=label, color=colors[i])
    ax.set_ylabel("% of samples")
    ax.set_title("(c) Chain Length Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in all_chain_keys])
    ax.set_xlabel("Chain length (# tool calls)")
    ax.legend()

    plt.tight_layout()
    fig_path = figures_dir / "v2_8b_tool_usage_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure saved to %s", fig_path)

    # Key findings summary
    log.info("=" * 80)
    log.info("KEY FINDINGS")
    log.info("=" * 80)

    e3_3b = all_tool_usage.get("E3 (3B)", {})
    e3_8b = all_tool_usage.get("E3-8B (7B)", {})

    if e3_3b and e3_8b:
        chain_3b = e3_3b["avg_chain_length"]
        chain_8b = e3_8b["avg_chain_length"]
        if chain_8b > chain_3b + 0.1:
            log.info("  7B shows LONGER chains: %.2f vs %.2f (+%.2f)", chain_8b, chain_3b, chain_8b - chain_3b)
            log.info("  -> Model scaling DOES affect tool usage behavior")
        else:
            log.info("  7B chains similar to 3B: %.2f vs %.2f", chain_8b, chain_3b)
            log.info("  -> 'Intelligent delegation' is model-size-independent")

        # Check for multi-tool usage
        extra_tools_8b = set(e3_8b["tool_sample_rates"].keys()) - set(e3_3b["tool_sample_rates"].keys())
        if extra_tools_8b:
            log.info("  7B uses additional tools: %s", extra_tools_8b)
        else:
            used_3b = set(e3_3b["tool_sample_rates"].keys())
            used_8b = set(e3_8b["tool_sample_rates"].keys())
            if used_3b == used_8b:
                log.info("  Both models use identical tool sets: %s", used_3b)


if __name__ == "__main__":
    main()
