#!/usr/bin/env python3
"""C3 — supplementary statistics for Table 8: Spearman, confidence intervals, and a
multiplicity statement.

Table 8 currently reports Pearson r only, over n=11 (CIC) and n=9 (UNSW) folds, on a
bounded outcome with mass at 0 and 1, across three detector families and two datasets.
That is 6 simultaneous tests. This script adds, for each of the 6 cells:
  - Pearson r with BCa-free percentile bootstrap CI (paired resampling of folds)
  - Spearman rho with the same bootstrap CI
  - Holm-Bonferroni adjusted p across the 6 tests, reported per correlation type
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

TAB = Path.home() / "Paper_project/SecHarness/project/results/tables/candidate_b"
N_BOOT = 10_000
SEED = 42
FAMILIES = {"zd_logreg": "Logistic reg. (linear)",
            "zd_histgb": "Grad. boosting (boosting)",
            "zd_rf": "Random forest (bagging)"}


def boot_ci(x: np.ndarray, y: np.ndarray, stat, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over folds. Degenerate resamples (no variance) are dropped."""
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        vals.append(stat(xb, yb)[0])
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            len(vals), float(np.mean(vals)))


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    df = pd.read_csv(TAB / "multidetector_validation.csv")
    rows = []
    for ds, g in df.groupby("dataset"):
        cov = g["cov_knn_attack"].to_numpy(dtype=float)
        for col, label in FAMILIES.items():
            zd = g[col].to_numpy(dtype=float)
            pr, pp = pearsonr(cov, zd)
            sr, sp = spearmanr(cov, zd)
            plo, phi, pn, pmean = boot_ci(cov, zd, pearsonr)
            slo, shi, sn, smean = boot_ci(cov, zd, spearmanr)
            rows.append({
                "dataset": ds, "family": label, "n": len(g),
                "pearson_r": pr, "pearson_p": pp,
                "pearson_ci_lo": plo, "pearson_ci_hi": phi, "pearson_boot_mean": pmean,
                "spearman_r": sr, "spearman_p": sp,
                "spearman_ci_lo": slo, "spearman_ci_hi": shi, "spearman_boot_mean": smean,
                "n_boot_valid": min(pn, sn),
            })

    out = pd.DataFrame(rows)
    # Multiplicity: 6 simultaneous tests per correlation type (3 families x 2 datasets)
    out["pearson_p_holm"] = holm(out["pearson_p"].to_numpy())
    out["spearman_p_holm"] = holm(out["spearman_p"].to_numpy())
    out["pearson_sig_holm_05"] = out["pearson_p_holm"] < 0.05
    out["spearman_sig_holm_05"] = out["spearman_p_holm"] < 0.05
    out["ci_excludes_zero_pearson"] = (out["pearson_ci_lo"] > 0) | (out["pearson_ci_hi"] < 0)
    out["ci_excludes_zero_spearman"] = (out["spearman_ci_lo"] > 0) | (out["spearman_ci_hi"] < 0)

    out.to_csv(TAB / "table8_spearman_ci_holm.csv", index=False)

    print("=== Table 8 revised: Pearson + Spearman, 95% percentile bootstrap CI, "
          "Holm-adjusted over 6 tests ===\n")
    hdr = (f"{'dataset':13s} {'family':26s} {'n':>3} "
           f"{'Pearson r [95% CI]':>26} {'p_holm':>9} "
           f"{'Spearman r [95% CI]':>26} {'p_holm':>9}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in out.iterrows():
        pc = f"{r.pearson_r:+.2f} [{r.pearson_ci_lo:+.2f},{r.pearson_ci_hi:+.2f}]"
        sc = f"{r.spearman_r:+.2f} [{r.spearman_ci_lo:+.2f},{r.spearman_ci_hi:+.2f}]"
        pm = "*" if r.pearson_sig_holm_05 else " "
        sm = "*" if r.spearman_sig_holm_05 else " "
        print(f"{r.dataset:13s} {r.family:26s} {r.n:>3} {pc:>26} {r.pearson_p_holm:8.3f}{pm} "
              f"{sc:>26} {r.spearman_p_holm:8.3f}{sm}")

    print("\n* = survives Holm-Bonferroni at alpha=0.05 across the 6 tests")
    print("\n--- what changes vs the submitted Table 8 ---")
    for _, r in out.iterrows():
        if r.pearson_sig_holm_05 and not r.spearman_sig_holm_05:
            print(f"  {r.dataset}/{r.family}: Pearson survives but SPEARMAN DOES NOT "
                  f"(rho={r.spearman_r:+.2f}, Holm p={r.spearman_p_holm:.3f}) "
                  f"-> relationship is driven by extreme folds, not by rank order")
        if not r.pearson_sig_holm_05 and r.pearson_p < 0.05:
            print(f"  {r.dataset}/{r.family}: Pearson p={r.pearson_p:.4f} raw but "
                  f"{r.pearson_p_holm:.3f} after Holm -> loses significance under multiplicity")

    with open(TAB / "table8_spearman_ci_holm_summary.json", "w") as fh:
        json.dump({"n_boot": N_BOOT, "seed": SEED, "n_tests_per_type": len(out),
                   "correction": "Holm-Bonferroni",
                   "rows": out.to_dict(orient="records")}, fh, indent=2, default=str)
    print(f"\n=== DONE -> {TAB/'table8_spearman_ci_holm.csv'} ===")


if __name__ == "__main__":
    main()
