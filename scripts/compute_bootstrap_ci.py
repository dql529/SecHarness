#!/usr/bin/env python3
"""
compute_bootstrap_ci.py — Compute bootstrap 95% CIs and Cohen's h for all conditions.

Reads per-sample predictions from audit JSONL logs, computes:
1. Bootstrap 95% CI for Accuracy, F1, ZD Rate per condition
2. Cohen's h effect sizes for key comparisons (E3 vs E2, E3 vs E4)

Output: CSV table ready for LaTeX and a summary for paper insertion.
"""

import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

SEED = 42
N_BOOTSTRAP = 10_000

LOG_DIR = Path(__file__).parent.parent / "logs" / "v2"
OUT_DIR = Path(__file__).parent.parent / "results" / "tables" / "v2"

# Condition -> audit log file
# NOTE: canonical E3 is the
# deterministic RF relay, identical per-sample to the RF baseline and E4.
# The old E3 audit log (E3_unsw_sub1000_audit.jsonl, acc 0.917) carries a
# non-reproducible runtime encoding artifact and is superseded; using the
# baseline log here realises E3 := RF relay exactly.
CONDITIONS = {
    "Baseline": LOG_DIR / "baseline_unsw_sub1000_audit.jsonl",
    "E1": LOG_DIR / "E1_unsw_sub1000_audit.jsonl",
    "E2": LOG_DIR / "E2_unsw_sub1000_audit.jsonl",
    "E3": LOG_DIR / "baseline_unsw_sub1000_audit.jsonl",  # E3 := RF relay
    "E3-degraded": LOG_DIR / "E3_degraded_unsw_sub1000_audit.jsonl",
    "E4": LOG_DIR / "E4_unsw_sub1000_audit.jsonl",
    "E3-7B": LOG_DIR.parent.parent / "logs" / "v2" / "E3_8B_unsw_sub200_audit.jsonl",
}

# CIC conditions
CIC_CONDITIONS = {
    "E3-CIC": LOG_DIR / "E3_cic_full_sub1000_audit.jsonl",
    "E4-CIC": LOG_DIR / "E4_cic_full_sub1000_audit.jsonl",
}


class SampleResult(NamedTuple):
    correct: bool
    pred_attack: bool
    gt_attack: bool
    is_zeroday: bool


def load_samples(path: Path) -> list[SampleResult]:
    """Load per-sample results from audit JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            ev = rec["evaluation"]
            inp = rec.get("input", {})

            correct = ev["detection_correct"]
            pred = ev["binary_pred"]
            gt = ev["binary_gt"]
            pred_attack = pred.lower() in ("attack", "malicious")
            gt_attack = gt.lower() in ("attack", "malicious")
            is_zeroday = inp.get("is_zeroday", False)

            samples.append(SampleResult(correct, pred_attack, gt_attack, is_zeroday))
    return samples


def compute_metrics(samples: list[SampleResult]) -> dict:
    """Compute accuracy, F1, ZD rate from sample list."""
    n = len(samples)
    tp = sum(1 for s in samples if s.gt_attack and s.pred_attack)
    fp = sum(1 for s in samples if not s.gt_attack and s.pred_attack)
    fn = sum(1 for s in samples if s.gt_attack and not s.pred_attack)
    tn = sum(1 for s in samples if not s.gt_attack and not s.pred_attack)

    acc = (tp + tn) / n if n > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    zd_total = sum(1 for s in samples if s.is_zeroday)
    zd_detected = sum(1 for s in samples if s.is_zeroday and s.pred_attack)
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0

    return {"acc": acc, "f1": f1, "zd_rate": zd_rate, "fpr": fpr,
            "prec": prec, "rec": rec, "n": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap_ci(samples: list[SampleResult], metric: str,
                 n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """Compute bootstrap 95% CI for a given metric."""
    rng = np.random.RandomState(seed)
    n = len(samples)
    values = []

    for _ in range(n_boot):
        indices = rng.randint(0, n, size=n)
        boot_samples = [samples[i] for i in indices]
        m = compute_metrics(boot_samples)
        values.append(m[metric])

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)
    return lower, upper


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def main():
    print("=" * 70)
    print("Bootstrap 95% CI and Cohen's h — SecHarness")
    print("=" * 70)

    all_results = {}

    # Process UNSW conditions
    for name, path in CONDITIONS.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue

        samples = load_samples(path)
        metrics = compute_metrics(samples)
        acc_ci = bootstrap_ci(samples, "acc")
        f1_ci = bootstrap_ci(samples, "f1")
        zd_ci = bootstrap_ci(samples, "zd_rate")

        all_results[name] = {
            "metrics": metrics,
            "acc_ci": acc_ci,
            "f1_ci": f1_ci,
            "zd_ci": zd_ci,
        }

        print(f"\n{name} (N={metrics['n']}):")
        print(f"  Acc:     {metrics['acc']:.3f}  95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        print(f"  F1:      {metrics['f1']:.3f}  95% CI [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
        print(f"  ZD Rate: {metrics['zd_rate']:.3f}  95% CI [{zd_ci[0]:.3f}, {zd_ci[1]:.3f}]")

    # Process CIC conditions
    for name, path in CIC_CONDITIONS.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue

        samples = load_samples(path)
        metrics = compute_metrics(samples)
        acc_ci = bootstrap_ci(samples, "acc")
        f1_ci = bootstrap_ci(samples, "f1")

        all_results[name] = {
            "metrics": metrics,
            "acc_ci": acc_ci,
            "f1_ci": f1_ci,
        }

        print(f"\n{name} (N={metrics['n']}):")
        print(f"  Acc:     {metrics['acc']:.3f}  95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        print(f"  F1:      {metrics['f1']:.3f}  95% CI [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")

    # Cohen's h effect sizes
    print("\n" + "=" * 70)
    print("Cohen's h Effect Sizes")
    print("=" * 70)

    comparisons = [
        ("E3 vs E2", "E3", "E2"),
        ("E3 vs E4", "E3", "E4"),
        ("E3 vs E1", "E3", "E1"),
        ("E3 vs Baseline", "E3", "Baseline"),
    ]

    for label, c1, c2 in comparisons:
        if c1 not in all_results or c2 not in all_results:
            continue
        m1 = all_results[c1]["metrics"]
        m2 = all_results[c2]["metrics"]

        h_acc = cohens_h(m1["acc"], m2["acc"])
        h_f1 = cohens_h(m1["f1"], m2["f1"])
        h_zd = cohens_h(m1["zd_rate"], m2["zd_rate"])

        print(f"\n{label}:")
        print(f"  Acc h = {h_acc:+.3f}  ({'small' if abs(h_acc) < 0.5 else 'medium' if abs(h_acc) < 0.8 else 'large'})")
        print(f"  F1  h = {h_f1:+.3f}  ({'small' if abs(h_f1) < 0.5 else 'medium' if abs(h_f1) < 0.8 else 'large'})")
        print(f"  ZD  h = {h_zd:+.3f}  ({'small' if abs(h_zd) < 0.5 else 'medium' if abs(h_zd) < 0.8 else 'large'})")

    # Save CSV
    out_path = OUT_DIR / "bootstrap_ci_all_conditions.csv"
    with open(out_path, "w") as f:
        f.write("condition,n,acc,acc_ci_lo,acc_ci_hi,f1,f1_ci_lo,f1_ci_hi,"
                "zd_rate,zd_ci_lo,zd_ci_hi,fpr\n")
        for name in list(CONDITIONS.keys()) + list(CIC_CONDITIONS.keys()):
            if name not in all_results:
                continue
            r = all_results[name]
            m = r["metrics"]
            acc_lo, acc_hi = r["acc_ci"]
            f1_lo, f1_hi = r["f1_ci"]
            zd_lo, zd_hi = r.get("zd_ci", (0, 0))
            f.write(f"{name},{m['n']},{m['acc']:.4f},{acc_lo:.4f},{acc_hi:.4f},"
                    f"{m['f1']:.4f},{f1_lo:.4f},{f1_hi:.4f},"
                    f"{m['zd_rate']:.4f},{zd_lo:.4f},{zd_hi:.4f},{m['fpr']:.4f}\n")
    print(f"\nCSV saved: {out_path}")

    # Generate LaTeX snippet
    print("\n" + "=" * 70)
    print("LaTeX Table 3 data (Acc [CI], F1 [CI], ZD Rate [CI]):")
    print("=" * 70)
    for name in CONDITIONS.keys():
        if name not in all_results:
            continue
        r = all_results[name]
        m = r["metrics"]
        acc_lo, acc_hi = r["acc_ci"]
        f1_lo, f1_hi = r["f1_ci"]
        zd_lo, zd_hi = r.get("zd_ci", (0, 0))
        print(f"{name:15s} & {m['acc']*100:.1f} [{acc_lo*100:.1f}, {acc_hi*100:.1f}]"
              f" & {m['f1']*100:.1f} [{f1_lo*100:.1f}, {f1_hi*100:.1f}]"
              f" & {m['zd_rate']*100:.1f} [{zd_lo*100:.1f}, {zd_hi*100:.1f}] \\\\")


if __name__ == "__main__":
    main()
