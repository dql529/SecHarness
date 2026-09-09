#!/usr/bin/env python3
"""
Generate cross-dataset comparison table: UNSW-NB15 vs CIC-IDS2017.

Reads E2/E4/E5 results + C12 analysis from both datasets and produces
a unified comparison CSV.

Usage:
    python cross_dataset_comparison.py
"""

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT / "results" / "tables"


def read_binary_metrics(prefix: str) -> dict:
    """Read binary metrics CSV (metric,value format)."""
    path = TABLES_DIR / f"{prefix}_binary_metrics.csv"
    if not path.exists():
        return {}
    result = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) >= 2:
                result[row[0]] = float(row[1])
    return result


def read_zeroday_metrics(prefix: str) -> dict:
    path = TABLES_DIR / f"{prefix}_zeroday_metrics.csv"
    if not path.exists():
        return {}
    result = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            result[row[0]] = float(row[1]) if row[1] else 0
    return result


def read_strategy_comparison(prefix: str) -> dict:
    """Read strategy comparison CSV."""
    path = TABLES_DIR / f"E5_strategy_comparison_{prefix}.csv"
    if not path.exists():
        return {}
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["strategy"]] = {k: float(v) if v else 0
                                        for k, v in row.items() if k != "strategy"}
    return result


def read_disagreement_analysis(prefix: str) -> dict:
    """Read C12 disagreement analysis CSV — extract key metrics."""
    path = TABLES_DIR / f"C12_disagreement_analysis_{prefix}.csv"
    if not path.exists():
        return {}
    result = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                val = row[1].strip()
                if key in ("chi2_statistic", "chi2_p_value",
                           "zd_predictor_precision", "zd_predictor_recall"):
                    result[key] = float(val)
                elif key == "zeroday":
                    result["zeroday_disagreement_rate"] = float(val)
                elif key == "known_attack":
                    result["known_disagreement_rate"] = float(val)
    return result


def main():
    rows = []

    # Define experiments for each dataset
    datasets = {
        "UNSW-NB15": {
            "E2_prefix": "E2_unsw_ml_only",
            "E4_prefix": "E4_unsw",
            "E5_prefix": "E5_unsw",
            "strategy_prefix": "unsw",
            "c12_prefix": "unsw",
        },
        "CIC-IDS2017": {
            "E2_prefix": "E2_cic_full_ml_only",
            "E4_prefix": "E4_cic_full",
            "E5_prefix": "E5_cic_full",
            "strategy_prefix": "cic_full",
            "c12_prefix": "cic_full",
        },
    }

    comparison = []

    for ds_name, prefixes in datasets.items():
        e2 = read_binary_metrics(prefixes["E2_prefix"])
        e2_zd = read_zeroday_metrics(prefixes["E2_prefix"])
        e4 = read_binary_metrics(prefixes["E4_prefix"])
        e4_zd = read_zeroday_metrics(prefixes["E4_prefix"])
        e5 = read_binary_metrics(prefixes["E5_prefix"])
        e5_zd = read_zeroday_metrics(prefixes["E5_prefix"])
        strategies = read_strategy_comparison(prefixes["strategy_prefix"])
        c12 = read_disagreement_analysis(prefixes["c12_prefix"])

        adaptive = strategies.get("E5_adaptive", {})

        comparison.append({
            "dataset": ds_name,
            "E2_ML_Acc": e2.get("accuracy", ""),
            "E2_ML_F1": e2.get("f1", ""),
            "E2_ML_FPR": e2.get("fpr", ""),
            "E2_ML_ZD": e2_zd.get("zeroday_detection_rate", ""),
            "E4_NoHarness_Acc": e4.get("accuracy", ""),
            "E4_NoHarness_F1": e4.get("f1", ""),
            "E4_NoHarness_FPR": e4.get("fpr", ""),
            "E4_NoHarness_ZD": e4_zd.get("zeroday_detection_rate", ""),
            "E5_alpha_priority_Acc": e5.get("accuracy", ""),
            "E5_alpha_priority_F1": e5.get("f1", ""),
            "E5_alpha_priority_FPR": e5.get("fpr", ""),
            "E5_alpha_priority_ZD": e5_zd.get("zeroday_detection_rate", ""),
            "E5_adaptive_Acc": adaptive.get("accuracy", ""),
            "E5_adaptive_F1": adaptive.get("f1", ""),
            "E5_adaptive_FPR": adaptive.get("fpr", ""),
            "E5_adaptive_ZD": adaptive.get("zeroday_rate", ""),
            "zeroday_disagreement_rate": c12.get("zeroday_disagreement_rate", ""),
            "known_disagreement_rate": c12.get("known_disagreement_rate", ""),
            "chi2": c12.get("chi2_statistic", ""),
            "chi2_p_value": c12.get("chi2_p_value", ""),
        })

    # Write CSV
    output_path = TABLES_DIR / "cross_dataset_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(comparison[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in comparison:
            w.writerow(row)

    # Print summary
    print("=" * 80)
    print("Cross-Dataset Comparison: UNSW-NB15 vs CIC-IDS2017")
    print("=" * 80)

    header = f"{'Metric':<35s} {'UNSW-NB15':>15s} {'CIC-IDS2017':>15s}"
    print(header)
    print("-" * 65)

    metric_labels = [
        ("E2_ML_Acc", "E2 ML-only Accuracy"),
        ("E2_ML_F1", "E2 ML-only F1"),
        ("E2_ML_ZD", "E2 ML-only Zero-day"),
        ("E4_NoHarness_Acc", "E4 NoHarness Accuracy"),
        ("E4_NoHarness_F1", "E4 NoHarness F1"),
        ("E4_NoHarness_ZD", "E4 NoHarness Zero-day"),
        ("E5_alpha_priority_Acc", "E5 alpha_priority Accuracy"),
        ("E5_alpha_priority_F1", "E5 alpha_priority F1"),
        ("E5_alpha_priority_ZD", "E5 alpha_priority Zero-day"),
        ("E5_adaptive_Acc", "E5 adaptive Accuracy"),
        ("E5_adaptive_F1", "E5 adaptive F1"),
        ("E5_adaptive_ZD", "E5 adaptive Zero-day"),
        ("zeroday_disagreement_rate", "Zero-day Disagreement Rate"),
        ("known_disagreement_rate", "Known Disagreement Rate"),
        ("chi2", "Chi-square Statistic"),
    ]

    unsw = comparison[0]
    cic = comparison[1]
    for key, label in metric_labels:
        uval = unsw.get(key, "")
        cval = cic.get(key, "")
        ufmt = f"{float(uval):.4f}" if uval != "" else "—"
        cfmt = f"{float(cval):.4f}" if cval != "" else "—"
        print(f"  {label:<33s} {ufmt:>15s} {cfmt:>15s}")

    print("=" * 80)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
