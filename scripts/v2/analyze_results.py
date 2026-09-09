#!/usr/bin/env python3
"""
analyze_results.py — v2 experiment results analysis and visualization.

Reads JSONL audit logs from all experiments, computes metrics, and generates:
- 4 figures (RQ1-RQ4)
- 4 tables (main results, ablation, tool usage, reasoning analysis)

Usage:
    python project/scripts/v2/analyze_results.py --log-dir project/logs/v2/ --output-dir project/results/
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("analyzer")

# Experiment display names
EXP_NAMES = {
    "baseline_unsw": "ML Baseline",
    "E1_unsw": "E1: ZS, No Harness",
    "E2_unsw": "E2: FT, No Harness",
    "E3_unsw": "E3: ZS + Harness",
    "E3_degraded_unsw": "E3-deg: ZS + Harness (degraded)",
    "E4_unsw": "E4: FT + Harness",
    "E4_noTools_unsw": "No Tools",
    "E4_noKnowledge_unsw": "No Knowledge",
    "E4_noObservation_unsw": "No Observation",
    "E4_noPermissions_unsw": "No Permissions",
}

MAIN_EXPERIMENTS = ["baseline_unsw", "E1_unsw", "E2_unsw", "E3_unsw", "E3_degraded_unsw", "E4_unsw"]
ABLATION_EXPERIMENTS = ["E4_unsw", "E4_noTools_unsw", "E4_noKnowledge_unsw", "E4_noObservation_unsw", "E4_noPermissions_unsw"]


def load_records(log_dir: Path) -> dict[str, list[dict]]:
    """Load all JSONL audit logs from a directory.

    When multiple files map to the same experiment ID (e.g., sub200 and sub1000),
    keeps the one with more records.
    """
    all_records = {}
    for f in sorted(log_dir.glob("*_audit.jsonl")):
        # Extract experiment ID: e.g., E3_unsw_sub200_audit.jsonl -> E3_unsw
        name = f.stem.replace("_audit", "")
        # Remove _sub200, _pilot10 etc suffixes
        for suffix in ["_sub2000", "_sub1000", "_sub500", "_sub200", "_sub100", "_pilot20", "_pilot10"]:
            name = name.replace(suffix, "")
        records = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            # Keep the file with the most records when there are duplicates
            if name in all_records and len(all_records[name]) >= len(records):
                log.info("Skipping %s (%d records, already have %d) from %s",
                         name, len(records), len(all_records[name]), f.name)
                continue
            all_records[name] = records
            log.info("Loaded %s: %d records from %s", name, len(records), f.name)
    return all_records


def compute_experiment_metrics(records: list[dict]) -> dict:
    """Compute all metrics from a list of audit records."""
    tp = fp = fn = tn = 0
    zd_total = zd_detected = 0
    total_latency = 0.0
    total_tokens = 0
    tool_counts = defaultdict(int)
    tool_sample_counts = defaultdict(int)  # samples that used each tool
    chain_lengths = []
    llm_call_counts = []
    self_corrected = 0
    escalated = 0
    parse_fails = 0
    termination_counts = defaultdict(int)

    for r in records:
        ev = r.get("evaluation", {})
        gt = ev.get("binary_gt", "")
        pred = ev.get("binary_pred", "")
        is_zd = r.get("input", {}).get("is_zeroday", False)
        eff = r.get("efficiency", {})
        res = r.get("result", {})

        if gt == "attack" and pred == "attack":
            tp += 1
        elif gt == "benign" and pred == "attack":
            fp += 1
        elif gt == "attack" and pred == "benign":
            fn += 1
        elif gt == "benign" and pred == "benign":
            tn += 1

        if is_zd:
            zd_total += 1
            if pred == "attack":
                zd_detected += 1

        total_latency += eff.get("total_latency_ms", 0)
        total_tokens += eff.get("total_tokens", 0)

        tc = r.get("tool_chain", [])
        chain_lengths.append(len(tc))
        llm_call_counts.append(eff.get("num_llm_calls", 1))

        tools_used_set = set()
        for call in tc:
            tool_name = call.get("tool", "")
            tool_counts[tool_name] += 1
            tools_used_set.add(tool_name)
        for t in tools_used_set:
            tool_sample_counts[t] += 1

        if res.get("self_corrected"):
            self_corrected += 1
        if res.get("escalated"):
            escalated += 1
        term = res.get("termination", "")
        termination_counts[term] += 1
        if term == "parse_fail":
            parse_fails += 1

    n = len(records)
    acc = (tp + tn) / n if n > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0

    return {
        "n": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr_val,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "zd_total": zd_total,
        "zd_detected": zd_detected,
        "zd_rate": zd_rate,
        "avg_latency_ms": total_latency / n if n > 0 else 0,
        "avg_tokens": total_tokens / n if n > 0 else 0,
        "avg_chain_length": np.mean(chain_lengths) if chain_lengths else 0,
        "avg_llm_calls": np.mean(llm_call_counts) if llm_call_counts else 0,
        "self_correction_rate": self_corrected / n if n > 0 else 0,
        "escalation_rate": escalated / n if n > 0 else 0,
        "parse_fail_rate": parse_fails / n if n > 0 else 0,
        "tool_counts": dict(tool_counts),
        "tool_sample_rates": {t: c / n for t, c in tool_sample_counts.items()} if n > 0 else {},
        "termination_counts": dict(termination_counts),
    }


def plot_rq1_main_comparison(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """RQ1: Bar chart comparing E1/E2/E3/E4/Baseline on Acc, F1, ZD."""
    exps = [e for e in MAIN_EXPERIMENTS if e in metrics_map]
    if not exps:
        log.warning("No main experiments found for RQ1 plot")
        return

    labels = [EXP_NAMES.get(e, e) for e in exps]
    acc = [metrics_map[e]["accuracy"] for e in exps]
    f1_scores = [metrics_map[e]["f1"] for e in exps]
    zd = [metrics_map[e]["zd_rate"] for e in exps]

    x = np.arange(len(exps))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, acc, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x, f1_scores, width, label="F1 Score", color="#DD8452")
    bars3 = ax.bar(x + width, zd, width, label="Zero-day Rate", color="#55A868")

    ax.set_ylabel("Score")
    ax.set_title(f"RQ1: Main Experiment Comparison ({dataset.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = output_dir / "figures" / f"RQ1_main_comparison_{dataset}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_rq2_harness_vs_finetune(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """RQ2: Direct comparison of E2 (FT only) vs E3 (Harness only)."""
    needed = ["E2_unsw", "E3_unsw"]
    if not all(e in metrics_map for e in needed):
        log.warning("Missing E2 or E3 for RQ2 plot")
        return

    metrics_names = ["accuracy", "f1", "recall", "precision", "fpr", "zd_rate"]
    display_names = ["Accuracy", "F1", "Recall", "Precision", "FPR", "ZD Rate"]

    e2 = [metrics_map["E2_unsw"][m] for m in metrics_names]
    e3 = [metrics_map["E3_unsw"][m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, e2, width, label="E2: Fine-tuned Only", color="#DD8452")
    ax.bar(x + width/2, e3, width, label="E3: Harness Only (ZS)", color="#55A868")

    ax.set_ylabel("Score")
    ax.set_title(f"RQ2: Fine-tuning vs Harness ({dataset.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i in range(len(metrics_names)):
        ax.annotate(f'{e2[i]:.2f}', xy=(x[i] - width/2, e2[i]), xytext=(0, 3),
                   textcoords="offset points", ha='center', fontsize=8)
        ax.annotate(f'{e3[i]:.2f}', xy=(x[i] + width/2, e3[i]), xytext=(0, 3),
                   textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    path = output_dir / "figures" / f"RQ2_harness_vs_finetune_{dataset}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_rq3_ablation(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """RQ3: Ablation study — E4 vs 4 ablation conditions."""
    exps = [e for e in ABLATION_EXPERIMENTS if e in metrics_map]
    if len(exps) < 2:
        log.warning("Not enough ablation experiments for RQ3 plot")
        return

    labels = [EXP_NAMES.get(e, e) for e in exps]
    acc = [metrics_map[e]["accuracy"] for e in exps]
    f1_scores = [metrics_map[e]["f1"] for e in exps]

    x = np.arange(len(exps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, acc, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, f1_scores, width, label="F1 Score", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title(f"RQ3: Ablation Study ({dataset.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Highlight E4 (full) bar
    for bars in ax.containers:
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(2)

    # Add value labels
    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    path = output_dir / "figures" / f"RQ3_ablation_{dataset}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_rq4_reasoning_chain(metrics_map: dict, all_records: dict, output_dir: Path, dataset: str = "unsw"):
    """RQ4: Reasoning chain analysis — step distribution + tool frequency."""
    # Only harness experiments
    harness_exps = [e for e in ["E3_unsw", "E4_unsw"] if e in all_records]
    if not harness_exps:
        log.warning("No harness experiments for RQ4 plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: chain length distribution
    ax1 = axes[0]
    for exp_id in harness_exps:
        lengths = [len(r.get("tool_chain", [])) for r in all_records[exp_id]]
        bins = range(0, max(lengths) + 2) if lengths else range(0, 6)
        ax1.hist(lengths, bins=bins, alpha=0.6, label=EXP_NAMES.get(exp_id, exp_id), edgecolor="black")
    ax1.set_xlabel("Tool Chain Length (steps)")
    ax1.set_ylabel("Sample Count")
    ax1.set_title("Reasoning Chain Length Distribution")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: tool usage frequency
    ax2 = axes[1]
    tool_data = {}
    for exp_id in harness_exps:
        m = metrics_map.get(exp_id, {})
        rates = m.get("tool_sample_rates", {})
        for tool, rate in rates.items():
            if tool not in ("classify", "escalate", "log_decision"):
                if tool not in tool_data:
                    tool_data[tool] = {}
                tool_data[tool][exp_id] = rate

    if tool_data:
        tools = sorted(tool_data.keys())
        x = np.arange(len(tools))
        width = 0.35
        for i, exp_id in enumerate(harness_exps):
            vals = [tool_data.get(t, {}).get(exp_id, 0) for t in tools]
            offset = (i - len(harness_exps)/2 + 0.5) * width
            ax2.bar(x + offset, vals, width, label=EXP_NAMES.get(exp_id, exp_id))
        ax2.set_xlabel("Tool")
        ax2.set_ylabel("Usage Rate (fraction of samples)")
        ax2.set_title("Tool Usage Frequency")
        ax2.set_xticks(x)
        ax2.set_xticklabels(tools, rotation=30, ha="right", fontsize=8)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "figures" / f"RQ4_reasoning_chain_{dataset}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def export_main_results_table(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """Export main experiment results as CSV."""
    exps = [e for e in MAIN_EXPERIMENTS if e in metrics_map]
    if not exps:
        return

    fields = ["experiment", "n", "accuracy", "precision", "recall", "f1", "fpr",
              "zd_rate", "avg_latency_ms", "avg_tokens", "avg_chain_length",
              "self_correction_rate", "escalation_rate", "parse_fail_rate"]

    path = output_dir / "tables" / "v2" / f"v2_main_results_{dataset}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in exps:
            m = metrics_map[e]
            row = {
                "experiment": EXP_NAMES.get(e, e),
                "n": m["n"],
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "fpr": round(m["fpr"], 4),
                "zd_rate": round(m["zd_rate"], 4),
                "avg_latency_ms": round(m["avg_latency_ms"], 1),
                "avg_tokens": round(m["avg_tokens"], 0),
                "avg_chain_length": round(m["avg_chain_length"], 2),
                "self_correction_rate": round(m["self_correction_rate"], 4),
                "escalation_rate": round(m["escalation_rate"], 4),
                "parse_fail_rate": round(m["parse_fail_rate"], 4),
            }
            w.writerow(row)
    log.info("Main results table → %s", path)


def export_ablation_table(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """Export ablation study results."""
    exps = [e for e in ABLATION_EXPERIMENTS if e in metrics_map]
    if len(exps) < 2:
        return

    fields = ["experiment", "n", "accuracy", "f1", "fpr", "zd_rate",
              "avg_chain_length", "avg_tokens", "avg_latency_ms",
              "acc_delta", "f1_delta"]

    base = metrics_map.get("E4_unsw", {})

    path = output_dir / "tables" / "v2" / f"v2_ablation_results_{dataset}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in exps:
            m = metrics_map[e]
            row = {
                "experiment": EXP_NAMES.get(e, e),
                "n": m["n"],
                "accuracy": round(m["accuracy"], 4),
                "f1": round(m["f1"], 4),
                "fpr": round(m["fpr"], 4),
                "zd_rate": round(m["zd_rate"], 4),
                "avg_chain_length": round(m["avg_chain_length"], 2),
                "avg_tokens": round(m["avg_tokens"], 0),
                "avg_latency_ms": round(m["avg_latency_ms"], 1),
                "acc_delta": round(m["accuracy"] - base.get("accuracy", 0), 4) if base else "",
                "f1_delta": round(m["f1"] - base.get("f1", 0), 4) if base else "",
            }
            w.writerow(row)
    log.info("Ablation table → %s", path)


def export_tool_usage_table(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """Export tool usage statistics."""
    harness_exps = [e for e in metrics_map if metrics_map[e].get("tool_counts")]
    if not harness_exps:
        return

    # Collect all tool names
    all_tools = set()
    for e in harness_exps:
        all_tools.update(metrics_map[e]["tool_counts"].keys())
    all_tools = sorted(all_tools)

    fields = ["experiment"] + [f"{t}_count" for t in all_tools] + [f"{t}_rate" for t in all_tools]

    path = output_dir / "tables" / "v2" / f"v2_tool_usage_{dataset}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in harness_exps:
            m = metrics_map[e]
            row = {"experiment": EXP_NAMES.get(e, e)}
            for t in all_tools:
                row[f"{t}_count"] = m["tool_counts"].get(t, 0)
                row[f"{t}_rate"] = round(m["tool_sample_rates"].get(t, 0), 4)
            w.writerow(row)
    log.info("Tool usage table → %s", path)


def export_reasoning_table(metrics_map: dict, output_dir: Path, dataset: str = "unsw"):
    """Export reasoning chain analysis."""
    exps = [e for e in metrics_map]
    if not exps:
        return

    fields = ["experiment", "avg_llm_calls", "avg_chain_length", "self_correction_rate",
              "escalation_rate", "parse_fail_rate", "avg_tokens", "avg_latency_ms"]

    path = output_dir / "tables" / "v2" / f"v2_reasoning_analysis_{dataset}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in exps:
            m = metrics_map[e]
            row = {
                "experiment": EXP_NAMES.get(e, e),
                "avg_llm_calls": round(m["avg_llm_calls"], 2),
                "avg_chain_length": round(m["avg_chain_length"], 2),
                "self_correction_rate": round(m["self_correction_rate"], 4),
                "escalation_rate": round(m["escalation_rate"], 4),
                "parse_fail_rate": round(m["parse_fail_rate"], 4),
                "avg_tokens": round(m["avg_tokens"], 0),
                "avg_latency_ms": round(m["avg_latency_ms"], 1),
            }
            w.writerow(row)
    log.info("Reasoning analysis table → %s", path)


def main():
    ap = argparse.ArgumentParser(description="Analyze v2 experiment results")
    ap.add_argument("--log-dir", default="project/logs/v2/", help="Directory with JSONL audit logs")
    ap.add_argument("--output-dir", default="project/results/", help="Output directory for figures and tables")
    ap.add_argument("--dataset", default="unsw", help="Dataset name for plot titles")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    # Load all records
    all_records = load_records(log_dir)
    if not all_records:
        log.error("No audit logs found in %s", log_dir)
        sys.exit(1)

    # Compute metrics for each experiment
    metrics_map = {}
    for exp_id, records in all_records.items():
        metrics_map[exp_id] = compute_experiment_metrics(records)
        m = metrics_map[exp_id]
        log.info("  %s: Acc=%.4f F1=%.4f FPR=%.4f ZD=%.4f Lat=%.0fms",
                 exp_id, m["accuracy"], m["f1"], m["fpr"], m["zd_rate"], m["avg_latency_ms"])

    # Generate figures
    log.info("=== Generating figures ===")
    plot_rq1_main_comparison(metrics_map, output_dir, args.dataset)
    plot_rq2_harness_vs_finetune(metrics_map, output_dir, args.dataset)
    plot_rq3_ablation(metrics_map, output_dir, args.dataset)
    plot_rq4_reasoning_chain(metrics_map, all_records, output_dir, args.dataset)

    # Generate tables
    log.info("=== Generating tables ===")
    export_main_results_table(metrics_map, output_dir, args.dataset)
    export_ablation_table(metrics_map, output_dir, args.dataset)
    export_tool_usage_table(metrics_map, output_dir, args.dataset)
    export_reasoning_table(metrics_map, output_dir, args.dataset)

    log.info("=== Analysis complete ===")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<30} {'Acc':>6} {'F1':>6} {'FPR':>6} {'ZD':>6} {'Lat(ms)':>8} {'Tokens':>7}")
    print("-" * 80)
    for exp_id in MAIN_EXPERIMENTS + [e for e in ABLATION_EXPERIMENTS if e != "E4_unsw"]:
        if exp_id in metrics_map:
            m = metrics_map[exp_id]
            print(f"{EXP_NAMES.get(exp_id, exp_id):<30} "
                  f"{m['accuracy']:>6.4f} {m['f1']:>6.4f} {m['fpr']:>6.4f} "
                  f"{m['zd_rate']:>6.4f} {m['avg_latency_ms']:>8.1f} {m['avg_tokens']:>7.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
