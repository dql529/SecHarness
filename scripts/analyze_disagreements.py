#!/usr/bin/env python3
"""
C12 Part A: Disagreement Signal Deep Analysis

Analyzes E5 audit logs to understand:
1. Zero-day correlation: disagreement rate in known vs zero-day samples
2. Disagreement type breakdown: ground truth distribution per consensus_type
3. Confidence analysis: disagreement vs agreement confidence distributions
4. Visualizations: heatmap + confidence scatter plot

Usage:
    python analyze_disagreements.py
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

# Paths
PROJECT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT / "results" / "tables"
FIGURES_DIR = PROJECT / "results" / "figures"

# Defaults (UNSW)
DEFAULT_AUDIT_LOG = PROJECT / "logs" / "E5_unsw_audit.jsonl"
DEFAULT_ZERODAY_CLASSES = {"shellcode", "worms"}
DEFAULT_OUTPUT_PREFIX = "unsw"
NORMAL_CLASS = "normal"

# Module-level variable set by main() or caller
ZERODAY_CLASSES = DEFAULT_ZERODAY_CLASSES


def load_audit_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_sample(gt_label: str) -> dict:
    """Classify a sample as normal/known_attack/zeroday."""
    gt = gt_label.lower()
    if gt in (NORMAL_CLASS, "benign"):
        return {"category": "normal", "is_attack": False, "is_zeroday": False}
    elif gt in ZERODAY_CLASSES:
        return {"category": "zeroday", "is_attack": True, "is_zeroday": True}
    else:
        return {"category": "known_attack", "is_attack": True, "is_zeroday": False}


def analyze_zeroday_correlation(records: list[dict]) -> dict:
    """Part A.1: Zero-day correlation analysis with chi-square test."""
    # Count disagreements by sample category
    groups = {"normal": {"disagree": 0, "agree": 0},
              "known_attack": {"disagree": 0, "agree": 0},
              "zeroday": {"disagree": 0, "agree": 0}}

    for r in records:
        gt = r["ground_truth_label"]
        if gt is None:
            continue
        cat = classify_sample(gt)["category"]
        ctype = r["consensus_result"]["consensus_type"]
        is_disagree = ctype in ("alpha_only", "beta_only", "type_conflict", "uncertain")
        if is_disagree:
            groups[cat]["disagree"] += 1
        else:
            groups[cat]["agree"] += 1

    # Disagreement rates
    rates = {}
    for cat, counts in groups.items():
        total = counts["disagree"] + counts["agree"]
        rate = counts["disagree"] / total if total > 0 else 0
        rates[cat] = {"disagree": counts["disagree"], "agree": counts["agree"],
                      "total": total, "disagreement_rate": rate}

    # Chi-square test: known_attack vs zeroday
    known = groups["known_attack"]
    zd = groups["zeroday"]
    contingency = np.array([
        [known["disagree"], known["agree"]],
        [zd["disagree"], zd["agree"]]
    ])
    if contingency.min() >= 0 and contingency.sum() > 0:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    else:
        chi2, p_value, dof = 0, 1.0, 1

    # Disagreement as zero-day predictor
    # Among all disagreement samples, how many are actually zero-day?
    disagree_samples = [r for r in records
                        if r["consensus_result"]["consensus_type"]
                        in ("alpha_only", "beta_only", "type_conflict", "uncertain")]
    zd_in_disagree = sum(1 for r in disagree_samples
                         if r["ground_truth_label"] and
                         r["ground_truth_label"].lower() in ZERODAY_CLASSES)
    total_zd = rates["zeroday"]["total"]
    zd_detected_via_disagree = rates["zeroday"]["disagree"]

    predictor = {
        "precision": zd_in_disagree / len(disagree_samples) if disagree_samples else 0,
        "recall": zd_detected_via_disagree / total_zd if total_zd > 0 else 0,
        "total_disagreements": len(disagree_samples),
        "zeroday_in_disagreements": zd_in_disagree,
    }

    return {
        "rates": rates,
        "chi2_test": {"chi2": chi2, "p_value": p_value, "dof": dof,
                      "contingency": contingency.tolist()},
        "zeroday_predictor": predictor,
    }


def analyze_disagreement_types(records: list[dict]) -> dict:
    """Part A.2: Breakdown by consensus_type with ground truth distribution."""
    type_gt = defaultdict(lambda: Counter())
    type_correct = defaultdict(lambda: {"correct": 0, "incorrect": 0})

    for r in records:
        gt = r["ground_truth_label"]
        if gt is None:
            continue
        ctype = r["consensus_result"]["consensus_type"]
        cat_info = classify_sample(gt)

        # Map to broader category for analysis
        if cat_info["is_zeroday"]:
            gt_cat = f"zeroday:{gt}"
        elif cat_info["is_attack"]:
            gt_cat = f"known:{gt}"
        else:
            gt_cat = "Normal"

        type_gt[ctype][gt_cat] += 1

        if r.get("detection_correct", False):
            type_correct[ctype]["correct"] += 1
        else:
            type_correct[ctype]["incorrect"] += 1

    # Compute error rates per type
    result = {}
    for ctype in ["agreement", "alpha_only", "beta_only", "type_conflict", "uncertain"]:
        gt_dist = dict(type_gt.get(ctype, {}))
        corr = type_correct.get(ctype, {"correct": 0, "incorrect": 0})
        total = corr["correct"] + corr["incorrect"]
        result[ctype] = {
            "ground_truth_distribution": gt_dist,
            "correct": corr["correct"],
            "incorrect": corr["incorrect"],
            "total": total,
            "error_rate": corr["incorrect"] / total if total > 0 else 0,
        }

    return result


def analyze_confidence(records: list[dict]) -> dict:
    """Part A.3: Confidence distribution analysis."""
    agree_alpha_conf = []
    agree_beta_conf = []
    disagree_alpha_conf = []
    disagree_beta_conf = []

    correct_alpha_conf = []
    correct_beta_conf = []
    incorrect_alpha_conf = []
    incorrect_beta_conf = []

    for r in records:
        alpha_conf = r["alpha_verdict"]["confidence"]
        beta_conf = r["beta_verdict"]["confidence"]
        ctype = r["consensus_result"]["consensus_type"]
        is_disagree = ctype in ("alpha_only", "beta_only", "type_conflict", "uncertain")

        if is_disagree:
            disagree_alpha_conf.append(alpha_conf)
            disagree_beta_conf.append(beta_conf)
        else:
            agree_alpha_conf.append(alpha_conf)
            agree_beta_conf.append(beta_conf)

        if r.get("detection_correct", False):
            correct_alpha_conf.append(alpha_conf)
            correct_beta_conf.append(beta_conf)
        else:
            incorrect_alpha_conf.append(alpha_conf)
            incorrect_beta_conf.append(beta_conf)

    def conf_stats(vals):
        if not vals:
            return {"mean": 0, "std": 0, "median": 0, "n": 0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "median": float(np.median(arr)), "n": len(arr)}

    # Low confidence error rate analysis
    low_thresh = 0.5
    low_conf_samples = [r for r in records
                        if r["consensus_result"]["combined_confidence"] < low_thresh]
    high_conf_samples = [r for r in records
                         if r["consensus_result"]["combined_confidence"] >= low_thresh]
    low_errors = sum(1 for r in low_conf_samples if not r.get("detection_correct", False))
    high_errors = sum(1 for r in high_conf_samples if not r.get("detection_correct", False))

    return {
        "agreement": {"alpha": conf_stats(agree_alpha_conf),
                      "beta": conf_stats(agree_beta_conf)},
        "disagreement": {"alpha": conf_stats(disagree_alpha_conf),
                         "beta": conf_stats(disagree_beta_conf)},
        "correct": {"alpha": conf_stats(correct_alpha_conf),
                    "beta": conf_stats(correct_beta_conf)},
        "incorrect": {"alpha": conf_stats(incorrect_alpha_conf),
                      "beta": conf_stats(incorrect_beta_conf)},
        "low_confidence_analysis": {
            "threshold": low_thresh,
            "low_conf_total": len(low_conf_samples),
            "low_conf_errors": low_errors,
            "low_conf_error_rate": low_errors / len(low_conf_samples) if low_conf_samples else 0,
            "high_conf_total": len(high_conf_samples),
            "high_conf_errors": high_errors,
            "high_conf_error_rate": high_errors / len(high_conf_samples) if high_conf_samples else 0,
        }
    }


def generate_heatmap(records: list[dict], output_path: Path):
    """Part A.4a: Heatmap of consensus_type × attack_category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Build matrix
    ctypes = ["agreement", "alpha_only", "beta_only", "type_conflict"]
    categories_raw = defaultdict(lambda: defaultdict(int))

    for r in records:
        gt = r["ground_truth_label"]
        if gt is None:
            continue
        ctype = r["consensus_result"]["consensus_type"]
        if ctype == "uncertain":
            continue
        categories_raw[ctype][gt] += 1

    # Get all attack categories sorted
    all_cats = sorted(set(gt for r in records if r["ground_truth_label"]
                          for gt in [r["ground_truth_label"]]))

    matrix = np.zeros((len(ctypes), len(all_cats)))
    for i, ct in enumerate(ctypes):
        for j, cat in enumerate(all_cats):
            matrix[i, j] = categories_raw[ct].get(cat, 0)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Normalize rows for better visualization
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100

    im = ax.imshow(matrix_pct, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(all_cats)))
    ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(ctypes)))
    ax.set_yticklabels([t.replace("_", " ").title() for t in ctypes], fontsize=10)

    # Annotate cells with count and percentage
    for i in range(len(ctypes)):
        for j in range(len(all_cats)):
            count = int(matrix[i, j])
            pct = matrix_pct[i, j]
            if count > 0:
                color = "white" if pct > 50 else "black"
                ax.text(j, i, f"{count}\n({pct:.0f}%)",
                        ha="center", va="center", fontsize=8, color=color)

    # Mark zero-day columns
    for j, cat in enumerate(all_cats):
        if cat.lower() in ZERODAY_CLASSES:
            ax.text(j, -0.6, "ZD", ha="center", va="center",
                    fontsize=9, color="red", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Row %", shrink=0.8)
    ax.set_title("Disagreement Signal Distribution: Consensus Type × Attack Category",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Ground Truth Category")
    ax.set_ylabel("Consensus Type")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def generate_confidence_scatter(records: list[dict], output_path: Path):
    """Part A.4b: Alpha vs Beta confidence scatter, colored by correct/incorrect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha_confs = []
    beta_confs = []
    colors = []
    markers = []

    for r in records:
        alpha_confs.append(r["alpha_verdict"]["confidence"])
        beta_confs.append(r["beta_verdict"]["confidence"])
        correct = r.get("detection_correct", False)
        ctype = r["consensus_result"]["consensus_type"]
        is_disagree = ctype in ("alpha_only", "beta_only", "type_conflict", "uncertain")

        if correct and not is_disagree:
            colors.append("#2196F3")  # Blue: correct agreement
        elif correct and is_disagree:
            colors.append("#4CAF50")  # Green: correct disagreement
        elif not correct and not is_disagree:
            colors.append("#FF9800")  # Orange: incorrect agreement
        else:
            colors.append("#F44336")  # Red: incorrect disagreement

    fig, ax = plt.subplots(figsize=(9, 8))

    alpha_arr = np.array(alpha_confs)
    beta_arr = np.array(beta_confs)
    color_arr = np.array(colors)

    # Plot each category separately for legend
    categories = [
        ("#2196F3", "Correct + Agreement", "o"),
        ("#4CAF50", "Correct + Disagreement", "^"),
        ("#FF9800", "Incorrect + Agreement", "s"),
        ("#F44336", "Incorrect + Disagreement", "D"),
    ]

    for color, label, marker in categories:
        mask = color_arr == color
        if mask.sum() > 0:
            ax.scatter(alpha_arr[mask], beta_arr[mask], c=color, label=f"{label} (n={mask.sum()})",
                       alpha=0.4, s=15, marker=marker, edgecolors="none")

    ax.set_xlabel("Alpha Agent Confidence", fontsize=12)
    ax.set_ylabel("Beta Agent (ML) Confidence", fontsize=12)
    ax.set_title("Agent Confidence Distribution: Alpha vs Beta",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)

    # Add confidence threshold lines
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved: {output_path}")


def export_analysis_csv(zd_result: dict, type_result: dict, conf_result: dict, output_path: Path):
    """Export all analysis results to CSV."""
    rows = []

    # Section 1: Zero-day correlation
    rows.append(["=== Zero-day Correlation ===", "", ""])
    rows.append(["Category", "Disagreement Rate", "N"])
    for cat in ["normal", "known_attack", "zeroday"]:
        r = zd_result["rates"][cat]
        rows.append([cat, f"{r['disagreement_rate']:.4f}", r["total"]])
    rows.append(["chi2_statistic", f"{zd_result['chi2_test']['chi2']:.4f}", ""])
    rows.append(["chi2_p_value", f"{zd_result['chi2_test']['p_value']:.6f}", ""])
    rows.append(["zd_predictor_precision", f"{zd_result['zeroday_predictor']['precision']:.4f}", ""])
    rows.append(["zd_predictor_recall", f"{zd_result['zeroday_predictor']['recall']:.4f}", ""])

    # Section 2: Disagreement type breakdown
    rows.append([])
    rows.append(["=== Disagreement Type Analysis ===", "", ""])
    rows.append(["Consensus Type", "Error Rate", "Total", "Correct", "Incorrect"])
    for ctype in ["agreement", "alpha_only", "beta_only", "type_conflict"]:
        t = type_result.get(ctype, {})
        rows.append([ctype, f"{t.get('error_rate', 0):.4f}",
                      t.get("total", 0), t.get("correct", 0), t.get("incorrect", 0)])

    # Section 3: Ground truth distribution per type
    rows.append([])
    rows.append(["=== Ground Truth per Consensus Type ===", "", ""])
    for ctype in ["agreement", "alpha_only", "beta_only", "type_conflict"]:
        t = type_result.get(ctype, {})
        gt_dist = t.get("ground_truth_distribution", {})
        rows.append([f"--- {ctype} ---", "", ""])
        for gt_cat, count in sorted(gt_dist.items()):
            rows.append(["", gt_cat, count])

    # Section 4: Confidence analysis
    rows.append([])
    rows.append(["=== Confidence Analysis ===", "", ""])
    for group in ["agreement", "disagreement", "correct", "incorrect"]:
        rows.append([f"--- {group} ---", "", ""])
        for agent in ["alpha", "beta"]:
            s = conf_result[group][agent]
            rows.append([f"{agent}_mean", f"{s['mean']:.4f}", f"n={s['n']}"])
            rows.append([f"{agent}_std", f"{s['std']:.4f}", ""])

    lca = conf_result["low_confidence_analysis"]
    rows.append([f"--- Low confidence (<{lca['threshold']}) ---", "", ""])
    rows.append(["low_conf_error_rate", f"{lca['low_conf_error_rate']:.4f}", f"n={lca['low_conf_total']}"])
    rows.append(["high_conf_error_rate", f"{lca['high_conf_error_rate']:.4f}", f"n={lca['high_conf_total']}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    print(f"  Analysis CSV saved: {output_path}")


def main():
    global ZERODAY_CLASSES

    ap = argparse.ArgumentParser(description="C12 Part A: Disagreement Signal Deep Analysis")
    ap.add_argument("--audit-log", type=str, default=None,
                     help="Path to E5 audit JSONL (default: E5_unsw_audit.jsonl)")
    ap.add_argument("--zeroday-classes", type=str, default=None,
                     help="Comma-separated zero-day classes (default: shellcode,worms)")
    ap.add_argument("--output-prefix", type=str, default=None,
                     help="Output file prefix/suffix (default: unsw)")
    args = ap.parse_args()

    # Resolve parameters
    audit_log = Path(args.audit_log) if args.audit_log else DEFAULT_AUDIT_LOG
    if args.zeroday_classes:
        ZERODAY_CLASSES = set(c.strip().lower() for c in args.zeroday_classes.split(","))
    else:
        ZERODAY_CLASSES = DEFAULT_ZERODAY_CLASSES
    output_prefix = args.output_prefix or DEFAULT_OUTPUT_PREFIX

    print("=" * 60)
    print("C12 Part A: Disagreement Signal Deep Analysis")
    print(f"  Dataset: {output_prefix}")
    print(f"  Zero-day classes: {ZERODAY_CLASSES}")
    print("=" * 60)

    # Load data
    print(f"\nLoading audit log: {audit_log}")
    records = load_audit_records(audit_log)
    print(f"  Total records: {len(records)}")

    # 1. Zero-day correlation
    print("\n--- 1. Zero-day Correlation Analysis ---")
    zd_result = analyze_zeroday_correlation(records)
    for cat in ["normal", "known_attack", "zeroday"]:
        r = zd_result["rates"][cat]
        print(f"  {cat:15s}: disagree_rate={r['disagreement_rate']:.3f} "
              f"({r['disagree']}/{r['total']})")

    chi = zd_result["chi2_test"]
    print(f"\n  Chi-square test (known vs zeroday):")
    print(f"    χ²={chi['chi2']:.4f}, p={chi['p_value']:.6f}, dof={chi['dof']}")
    if chi["p_value"] < 0.05:
        print(f"    [OK] SIGNIFICANT: Zero-day samples have significantly different disagreement rate")
    else:
        print(f"    [--] NOT significant at alpha=0.05")

    pred = zd_result["zeroday_predictor"]
    print(f"\n  Disagreement as zero-day predictor:")
    print(f"    Precision: {pred['precision']:.4f} ({pred['zeroday_in_disagreements']}/{pred['total_disagreements']})")
    print(f"    Recall:    {pred['recall']:.4f}")

    # 2. Disagreement type analysis
    print("\n--- 2. Disagreement Type Analysis ---")
    type_result = analyze_disagreement_types(records)
    for ctype in ["agreement", "alpha_only", "beta_only", "type_conflict"]:
        t = type_result.get(ctype, {})
        print(f"  {ctype:15s}: error_rate={t.get('error_rate', 0):.3f} "
              f"(total={t.get('total', 0)}, errors={t.get('incorrect', 0)})")
        gt_dist = t.get("ground_truth_distribution", {})
        for gt_cat, count in sorted(gt_dist.items(), key=lambda x: -x[1])[:5]:
            print(f"    {gt_cat}: {count}")

    # 3. Confidence analysis
    print("\n--- 3. Confidence Analysis ---")
    conf_result = analyze_confidence(records)
    for group in ["agreement", "disagreement"]:
        print(f"  {group}:")
        for agent in ["alpha", "beta"]:
            s = conf_result[group][agent]
            print(f"    {agent}: mean={s['mean']:.3f} ±{s['std']:.3f} (n={s['n']})")

    lca = conf_result["low_confidence_analysis"]
    print(f"\n  Low confidence (<{lca['threshold']}) error rate: "
          f"{lca['low_conf_error_rate']:.3f} (n={lca['low_conf_total']})")
    print(f"  High confidence (≥{lca['threshold']}) error rate: "
          f"{lca['high_conf_error_rate']:.3f} (n={lca['high_conf_total']})")

    # 4. Export
    print("\n--- 4. Exporting Results ---")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    export_analysis_csv(zd_result, type_result, conf_result,
                        TABLES_DIR / f"C12_disagreement_analysis_{output_prefix}.csv")

    generate_heatmap(records, FIGURES_DIR / f"RQ3_disagreement_heatmap_{output_prefix}.png")
    generate_confidence_scatter(records, FIGURES_DIR / f"RQ3_confidence_scatter_{output_prefix}.png")

    print("\n[OK] Part A complete.")


if __name__ == "__main__":
    main()
