#!/usr/bin/env python3
"""
statistical_tests.py — Statistical significance tests for v2 experiments.

Compares E2 vs E3 (and optionally other pairs) using:
1. McNemar's test (paired sample comparison)
2. Bootstrap 95% CI for Acc, F1, ZD
3. Fisher's exact test on zero-day detection

Usage:
    python project/scripts/v2/statistical_tests.py \
        --e2-log project/logs/v2/E2_unsw_sub1000_audit.jsonl \
        --e3-log project/logs/v2/E3_unsw_sub1000_audit.jsonl \
        --output project/results/tables/v2/v2_statistical_tests.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("stat-tests")

SEED = 42
N_BOOTSTRAP = 10000


def load_audit_log(path: str) -> list[dict]:
    """Load JSONL audit log."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), path)
    return records


def extract_per_sample(records: list[dict]) -> dict:
    """Extract per-sample binary correctness, zero-day status, and binary predictions."""
    correct = []
    is_zeroday = []
    binary_pred = []
    binary_gt = []
    for r in records:
        ev = r.get("evaluation", {})
        gt = ev.get("binary_gt", "")
        pred = ev.get("binary_pred", "")
        correct.append(1 if gt == pred else 0)
        is_zeroday.append(r.get("input", {}).get("is_zeroday", False))
        binary_pred.append(pred)
        binary_gt.append(gt)
    return {
        "correct": np.array(correct),
        "is_zeroday": np.array(is_zeroday),
        "binary_pred": np.array(binary_pred),
        "binary_gt": np.array(binary_gt),
    }


def compute_metrics_from_arrays(correct, binary_pred, binary_gt, is_zeroday):
    """Compute Acc, F1, ZD from arrays."""
    n = len(correct)
    acc = correct.mean()

    tp = np.sum((binary_gt == "attack") & (binary_pred == "attack"))
    fp = np.sum((binary_gt == "benign") & (binary_pred == "attack"))
    fn = np.sum((binary_gt == "attack") & (binary_pred == "benign"))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    zd_mask = is_zeroday
    zd_total = zd_mask.sum()
    zd_detected = np.sum(zd_mask & (binary_pred == "attack"))
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0

    return acc, f1, zd_rate


def mcnemar_test(correct_a, correct_b):
    """McNemar's test comparing two classifiers on paired samples.

    Returns: chi2, p_value, odds_ratio, contingency_table (2x2)
    """
    # Contingency table:
    # [both_correct, a_correct_b_wrong]
    # [a_wrong_b_correct, both_wrong]
    both_correct = np.sum((correct_a == 1) & (correct_b == 1))
    a_only = np.sum((correct_a == 1) & (correct_b == 0))
    b_only = np.sum((correct_a == 0) & (correct_b == 1))
    both_wrong = np.sum((correct_a == 0) & (correct_b == 0))

    table = np.array([[both_correct, a_only], [b_only, both_wrong]])

    log.info("McNemar contingency table:")
    log.info("  Both correct: %d | E2 only correct: %d", both_correct, a_only)
    log.info("  E3 only correct: %d | Both wrong: %d", b_only, both_wrong)

    # McNemar's test (discordant cells: a_only, b_only)
    n_discord = a_only + b_only
    if n_discord == 0:
        log.warning("No discordant pairs — McNemar's test not applicable")
        return 0.0, 1.0, 1.0, table

    # Exact McNemar's test using binomial
    if n_discord < 25:
        # Use exact binomial test
        result = stats.binomtest(b_only, n_discord, 0.5)
        p_value = result.pvalue
        chi2 = (a_only - b_only) ** 2 / n_discord
    else:
        # Use chi-squared approximation with continuity correction
        chi2 = (abs(a_only - b_only) - 1) ** 2 / n_discord
        p_value = stats.chi2.sf(chi2, df=1)

    odds_ratio = b_only / a_only if a_only > 0 else float("inf")

    return chi2, p_value, odds_ratio, table


def bootstrap_ci(data_a, data_b, metric_fn, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap confidence intervals for a metric and the difference.

    Args:
        data_a, data_b: tuples of (correct, binary_pred, binary_gt, is_zeroday) for each experiment
        metric_fn: function(correct, binary_pred, binary_gt, is_zeroday) -> scalar
        n_bootstrap: number of bootstrap resamples
        ci: confidence level

    Returns:
        dict with ci_a, ci_b, ci_diff (each as (lower, upper))
    """
    rng = np.random.RandomState(seed)
    n = len(data_a[0])
    alpha = (1 - ci) / 2

    vals_a = []
    vals_b = []
    vals_diff = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        va = metric_fn(data_a[0][idx], data_a[1][idx], data_a[2][idx], data_a[3][idx])
        vb = metric_fn(data_b[0][idx], data_b[1][idx], data_b[2][idx], data_b[3][idx])
        vals_a.append(va)
        vals_b.append(vb)
        vals_diff.append(vb - va)  # E3 - E2

    vals_a = np.array(vals_a)
    vals_b = np.array(vals_b)
    vals_diff = np.array(vals_diff)

    return {
        "ci_a": (np.percentile(vals_a, alpha * 100), np.percentile(vals_a, (1 - alpha) * 100)),
        "ci_b": (np.percentile(vals_b, alpha * 100), np.percentile(vals_b, (1 - alpha) * 100)),
        "ci_diff": (np.percentile(vals_diff, alpha * 100), np.percentile(vals_diff, (1 - alpha) * 100)),
        "mean_a": vals_a.mean(),
        "mean_b": vals_b.mean(),
        "mean_diff": vals_diff.mean(),
    }


def fisher_zeroday_test(pred_a, pred_b, is_zeroday):
    """Fisher's exact test comparing zero-day detection rates."""
    zd_mask = is_zeroday
    if zd_mask.sum() == 0:
        return 0.0, 1.0, np.array([[0, 0], [0, 0]])

    zd_pred_a = pred_a[zd_mask]
    zd_pred_b = pred_b[zd_mask]

    a_detected = np.sum(zd_pred_a == "attack")
    a_missed = np.sum(zd_pred_a != "attack")
    b_detected = np.sum(zd_pred_b == "attack")
    b_missed = np.sum(zd_pred_b != "attack")

    table = np.array([[a_detected, a_missed], [b_detected, b_missed]])

    log.info("Fisher's exact test (zero-day):")
    log.info("  E2: detected=%d, missed=%d (rate=%.1f%%)", a_detected, a_missed, 100 * a_detected / (a_detected + a_missed) if (a_detected + a_missed) > 0 else 0)
    log.info("  E3: detected=%d, missed=%d (rate=%.1f%%)", b_detected, b_missed, 100 * b_detected / (b_detected + b_missed) if (b_detected + b_missed) > 0 else 0)

    odds_ratio, p_value = stats.fisher_exact(table)
    return odds_ratio, p_value, table


def main():
    ap = argparse.ArgumentParser(description="Statistical significance tests for v2 experiments")
    ap.add_argument("--e2-log", required=True, help="Path to E2 audit JSONL")
    ap.add_argument("--e3-log", required=True, help="Path to E3 audit JSONL")
    ap.add_argument("--output", default="project/results/tables/v2/v2_statistical_tests.csv")
    ap.add_argument("--baseline-log", default=None, help="Optional baseline audit JSONL for additional comparison")
    args = ap.parse_args()

    # Load data
    e2_records = load_audit_log(args.e2_log)
    e3_records = load_audit_log(args.e3_log)

    assert len(e2_records) == len(e3_records), (
        f"Sample count mismatch: E2={len(e2_records)}, E3={len(e3_records)}"
    )
    n = len(e2_records)
    log.info("Comparing %d paired samples", n)

    e2 = extract_per_sample(e2_records)
    e3 = extract_per_sample(e3_records)

    # Point estimates
    e2_acc, e2_f1, e2_zd = compute_metrics_from_arrays(e2["correct"], e2["binary_pred"], e2["binary_gt"], e2["is_zeroday"])
    e3_acc, e3_f1, e3_zd = compute_metrics_from_arrays(e3["correct"], e3["binary_pred"], e3["binary_gt"], e3["is_zeroday"])

    log.info("Point estimates:")
    log.info("  E2: Acc=%.4f  F1=%.4f  ZD=%.4f", e2_acc, e2_f1, e2_zd)
    log.info("  E3: Acc=%.4f  F1=%.4f  ZD=%.4f", e3_acc, e3_f1, e3_zd)

    # 1. McNemar's test
    log.info("=" * 60)
    log.info("1. McNemar's Test (E2 vs E3)")
    chi2, p_mcnemar, odds, table = mcnemar_test(e2["correct"], e3["correct"])
    log.info("  χ²=%.4f, p=%.6f, odds_ratio=%.3f", chi2, p_mcnemar, odds)
    sig_mcnemar = "Yes" if p_mcnemar < 0.05 else "No"
    log.info("  Significant at α=0.05? %s", sig_mcnemar)

    # 2. Bootstrap CIs
    log.info("=" * 60)
    log.info("2. Bootstrap 95%% CIs (n_bootstrap=%d)", N_BOOTSTRAP)

    def acc_fn(c, p, g, z):
        return c.mean()

    def f1_fn(c, p, g, z):
        tp = np.sum((g == "attack") & (p == "attack"))
        fp = np.sum((g == "benign") & (p == "attack"))
        fn = np.sum((g == "attack") & (p == "benign"))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    def zd_fn(c, p, g, z):
        zm = z.astype(bool)
        if zm.sum() == 0:
            return 0
        return np.sum(zm & (p == "attack")) / zm.sum()

    data_e2 = (e2["correct"], e2["binary_pred"], e2["binary_gt"], e2["is_zeroday"])
    data_e3 = (e3["correct"], e3["binary_pred"], e3["binary_gt"], e3["is_zeroday"])

    results = []
    for name, fn in [("Accuracy", acc_fn), ("F1", f1_fn), ("Zero-day Rate", zd_fn)]:
        bs = bootstrap_ci(data_e2, data_e3, fn)
        diff_sig = "Yes" if (bs["ci_diff"][0] > 0 or bs["ci_diff"][1] < 0) else "No"
        log.info("  %s:", name)
        log.info("    E2: %.4f [%.4f, %.4f]", bs["mean_a"], bs["ci_a"][0], bs["ci_a"][1])
        log.info("    E3: %.4f [%.4f, %.4f]", bs["mean_b"], bs["ci_b"][0], bs["ci_b"][1])
        log.info("    Diff (E3-E2): %.4f [%.4f, %.4f] — Significant? %s",
                 bs["mean_diff"], bs["ci_diff"][0], bs["ci_diff"][1], diff_sig)
        results.append({
            "metric": name,
            "e2_mean": round(bs["mean_a"], 4),
            "e2_ci_lo": round(bs["ci_a"][0], 4),
            "e2_ci_hi": round(bs["ci_a"][1], 4),
            "e3_mean": round(bs["mean_b"], 4),
            "e3_ci_lo": round(bs["ci_b"][0], 4),
            "e3_ci_hi": round(bs["ci_b"][1], 4),
            "diff_mean": round(bs["mean_diff"], 4),
            "diff_ci_lo": round(bs["ci_diff"][0], 4),
            "diff_ci_hi": round(bs["ci_diff"][1], 4),
            "significant": diff_sig,
        })

    # 3. Fisher's exact test (zero-day)
    log.info("=" * 60)
    log.info("3. Fisher's Exact Test (Zero-day Detection)")
    fisher_or, p_fisher, fisher_table = fisher_zeroday_test(
        e2["binary_pred"], e3["binary_pred"], e2["is_zeroday"]
    )
    log.info("  Odds ratio=%.4f, p=%.6f", fisher_or, p_fisher)
    sig_fisher = "Yes" if p_fisher < 0.05 else "No"
    log.info("  Significant at α=0.05? %s", sig_fisher)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Statistic", "Value"])
        writer.writerow(["McNemar", "chi2", round(chi2, 4)])
        writer.writerow(["McNemar", "p_value", round(p_mcnemar, 6)])
        writer.writerow(["McNemar", "odds_ratio", round(odds, 4)])
        writer.writerow(["McNemar", "significant_0.05", sig_mcnemar])
        writer.writerow(["McNemar", "n_discordant", int(table[0, 1] + table[1, 0])])
        writer.writerow(["McNemar", "e2_only_correct", int(table[0, 1])])
        writer.writerow(["McNemar", "e3_only_correct", int(table[1, 0])])
        writer.writerow([])

        for r in results:
            prefix = f"Bootstrap_{r['metric']}"
            writer.writerow([prefix, "e2_mean", r["e2_mean"]])
            writer.writerow([prefix, "e2_95ci", f"[{r['e2_ci_lo']}, {r['e2_ci_hi']}]"])
            writer.writerow([prefix, "e3_mean", r["e3_mean"]])
            writer.writerow([prefix, "e3_95ci", f"[{r['e3_ci_lo']}, {r['e3_ci_hi']}]"])
            writer.writerow([prefix, "diff_mean", r["diff_mean"]])
            writer.writerow([prefix, "diff_95ci", f"[{r['diff_ci_lo']}, {r['diff_ci_hi']}]"])
            writer.writerow([prefix, "significant_0.05", r["significant"]])
            writer.writerow([])

        writer.writerow(["Fisher_ZeroDay", "odds_ratio", round(fisher_or, 4)])
        writer.writerow(["Fisher_ZeroDay", "p_value", round(p_fisher, 6)])
        writer.writerow(["Fisher_ZeroDay", "significant_0.05", sig_fisher])

    log.info("Results saved to %s", output_path)

    # Also save JSON with full details
    json_path = output_path.with_suffix(".json")
    summary = {
        "n_samples": n,
        "mcnemar": {
            "chi2": round(chi2, 4),
            "p_value": round(p_mcnemar, 6),
            "odds_ratio": round(odds, 4),
            "significant": sig_mcnemar == "Yes",
            "contingency_table": table.tolist(),
        },
        "bootstrap": results,
        "fisher_zeroday": {
            "odds_ratio": round(fisher_or, 4),
            "p_value": round(p_fisher, 6),
            "significant": sig_fisher == "Yes",
            "contingency_table": fisher_table.tolist(),
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("JSON summary saved to %s", json_path)


if __name__ == "__main__":
    main()
