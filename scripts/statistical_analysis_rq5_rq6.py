#!/usr/bin/env python3
"""Statistical analysis for RQ5 (E3-noRF) and RQ6 (Scaling + Harness Delta).

Outputs:
  - results/tables/v2/rq5_rq6_statistical_tests.json
  - results/tables/v2/rq5_rq6_statistical_tests.csv
  - Console summary
"""

import json
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results" / "tables" / "v2"
LOGS = BASE / "logs" / "v2"
OUT_JSON = RESULTS / "rq5_rq6_statistical_tests.json"
OUT_CSV = RESULTS / "rq5_rq6_statistical_tests.csv"


def load_json(name: str) -> dict:
    with open(RESULTS / name) as f:
        return json.load(f)


def load_audit_verdicts(log_file: str) -> list[int]:
    """Load per-sample correct/incorrect (1/0) from audit log."""
    verdicts = []
    path = LOGS / log_file
    if not path.exists():
        return []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            # v2 audit format: evaluation.detection_correct
            eval_data = entry.get("evaluation", {})
            correct = int(eval_data.get("detection_correct", False))
            verdicts.append(correct)
    return verdicts


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (pooled). Returns (z_stat, p_value)."""
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val


def mcnemar_test(v1: list[int], v2: list[int]) -> tuple[float, float]:
    """McNemar's test on paired binary outcomes. Returns (chi2, p_value)."""
    assert len(v1) == len(v2), f"Length mismatch: {len(v1)} vs {len(v2)}"
    b = sum(1 for a, b_ in zip(v1, v2) if a == 1 and b_ == 0)  # v1 correct, v2 wrong
    c = sum(1 for a, b_ in zip(v1, v2) if a == 0 and b_ == 1)  # v1 wrong, v2 correct
    if b + c == 0:
        return 0.0, 1.0
    # Edwards correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) >= 25 else None
    if chi2 is not None:
        p_val = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        # Exact binomial for small samples
        p_val = stats.binom_test(b, b + c, 0.5) if hasattr(stats, 'binom_test') else stats.binomtest(b, b + c, 0.5).pvalue
    return chi2 if chi2 is not None else float('nan'), p_val


def bootstrap_acc_diff(v1: list[int], v2: list[int], n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap 95% CI for accuracy difference (v1 - v2). Returns (mean_diff, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    n = len(v1_arr)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(v1_arr[idx].mean() - v2_arr[idx].mean())
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ============================================================
# Load all experiment data
# ============================================================

experiments: dict[str, dict] = {}
file_map = {
    "E1_3B": ("E1_unsw_results.json", "E1_unsw_sub1000_audit.jsonl"),
    "E1_RF_3B": ("E1_rf_context_unsw_results.json", "E1_rf_context_unsw_sub1000_audit.jsonl"),
    "E1_RF_7B": ("E1_rf_context_7B_unsw_results.json", "E1_rf_context_7B_unsw_sub200_audit.jsonl"),
    "E1_RF_8B": ("E1_rf_context_8B_ollama_unsw_results.json", "E1_rf_context_8B_ollama_unsw_sub200_audit.jsonl"),
    "E1_RF_12B": ("E1_rf_context_12B_unsw_results.json", "E1_rf_context_12B_unsw_sub200_audit.jsonl"),
    "E1_RF_14B": ("E1_rf_context_14B_unsw_results.json", "E1_rf_context_14B_unsw_sub200_audit.jsonl"),
    "E1_RF_31B": ("E1_rf_context_31B_unsw_results.json", "E1_rf_context_31B_unsw_sub200_audit.jsonl"),
    "E1_RF_32B": ("E1_rf_context_32B_local_unsw_results.json", "E1_rf_context_32B_local_unsw_sub200_audit.jsonl"),
    "E1_RF_70B": ("E1_rf_context_70B_unsw_results.json", "E1_rf_context_70B_unsw_sub200_audit.jsonl"),
    "E3_3B": ("E3_unsw_results.json", "E3_unsw_sub200_audit.jsonl"),
    "E3_8B": ("E3_8B_unsw_results.json", "E3_8B_unsw_sub200_audit.jsonl"),
    "E3_14B": ("E3_14B_unsw_results.json", "E3_14B_unsw_sub200_audit.jsonl"),
    "E3_32B": ("E3_32B_local_unsw_results.json", "E3_32B_local_unsw_sub200_audit.jsonl"),
    "E3_70B": ("E3_70B_unsw_results.json", "E3_70B_unsw_sub200_audit.jsonl"),
    "E3_noRF_8B": ("E3_noRF_8B_unsw_results.json", "E3_noRF_8B_unsw_sub200_audit.jsonl"),
    "E3_noRF_14B": ("E3_noRF_14B_unsw_results.json", "E3_noRF_14B_unsw_sub200_audit.jsonl"),
    "E3_noRF_32B": ("E3_noRF_32B_unsw_results.json", "E3_noRF_32B_unsw_sub200_audit.jsonl"),
    "E3_noRF_70B": ("E3_noRF_70B_unsw_results.json", None),  # No audit log (remote)
}

for name, (json_file, audit_file) in file_map.items():
    d = load_json(json_file)
    verdicts = load_audit_verdicts(audit_file) if audit_file else []
    experiments[name] = {**d, "verdicts": verdicts, "name": name}

all_tests: list[dict[str, Any]] = []


def add_test(comparison: str, rq: str, test_name: str, stat: float, p_value: float,
             effect_size: float, effect_type: str, significant: bool, notes: str = ""):
    all_tests.append({
        "comparison": comparison,
        "rq": rq,
        "test": test_name,
        "statistic": round(stat, 4) if not math.isnan(stat) else None,
        "p_value": round(p_value, 6),
        "effect_size": round(effect_size, 4),
        "effect_type": effect_type,
        "significant": significant,
        "notes": notes,
    })


# ============================================================
# RQ5: E3-noRF vs E1 (HE framework independent contribution)
# ============================================================
print("=" * 70)
print("RQ5: E3-noRF vs E1 (HE Framework Independent Contribution)")
print("=" * 70)

e1 = experiments["E1_3B"]
e1_correct = e1["tp"] + e1["tn"]

for noRF_name in ["E3_noRF_8B", "E3_noRF_14B", "E3_noRF_32B", "E3_noRF_70B"]:
    exp = experiments[noRF_name]
    exp_correct = exp["tp"] + exp["tn"]

    # Two-proportion z-test (different N: 200 vs 1000)
    z, p = two_proportion_z_test(exp_correct, exp["n_samples"], e1_correct, e1["n_samples"])
    h = cohens_h(exp["accuracy"], e1["accuracy"])
    sig = p < 0.05

    label = f"{noRF_name} vs E1_3B"
    print(f"\n{label}:")
    print(f"  Acc: {exp['accuracy']:.3f} vs {e1['accuracy']:.3f} (Δ={exp['accuracy']-e1['accuracy']:+.3f})")
    print(f"  z={z:.3f}, p={p:.2e}, Cohen's h={h:.3f}, {'PASS' if sig else 'FAIL'}")

    add_test(label, "RQ5", "two-proportion z-test (Acc)",
             z, p, h, "Cohen's h", sig,
             f"N={exp['n_samples']} vs N={e1['n_samples']}, independent samples")

    # FPR comparison
    z_fpr, p_fpr = two_proportion_z_test(exp["fp"], exp["fp"] + exp["tn"],
                                          e1["fp"], e1["fp"] + e1["tn"])
    h_fpr = cohens_h(exp["fpr"], e1["fpr"])
    print(f"  FPR: {exp['fpr']:.3f} vs {e1['fpr']:.3f}, z={z_fpr:.3f}, p={p_fpr:.2e}")
    add_test(f"{noRF_name} vs E1_3B (FPR)", "RQ5", "two-proportion z-test (FPR)",
             z_fpr, p_fpr, h_fpr, "Cohen's h", p_fpr < 0.05)

# Cross-family comparison within E3-noRF
print("\n--- Cross-family within E3-noRF ---")
for pair in [("E3_noRF_8B", "E3_noRF_14B"), ("E3_noRF_8B", "E3_noRF_32B"),
             ("E3_noRF_70B", "E3_noRF_32B")]:
    a, b = experiments[pair[0]], experiments[pair[1]]
    a_correct = a["tp"] + a["tn"]
    b_correct = b["tp"] + b["tn"]
    z, p = two_proportion_z_test(a_correct, a["n_samples"], b_correct, b["n_samples"])
    h = cohens_h(a["accuracy"], b["accuracy"])
    label = f"{pair[0]} vs {pair[1]}"
    print(f"  {label}: Δ Acc={a['accuracy']-b['accuracy']:+.3f}, z={z:.3f}, p={p:.2e}, h={h:.3f}")
    add_test(label, "RQ5-family", "two-proportion z-test",
             z, p, h, "Cohen's h", p < 0.05,
             "Cross-family/scale comparison within E3-noRF")


# ============================================================
# RQ6: E1-RF Scaling (model family and size effects)
# ============================================================
print("\n" + "=" * 70)
print("RQ6: E1-RF Scaling — Model Family and Size Effects")
print("=" * 70)

# Pairwise: each E1-RF model vs E1-RF-3B (baseline)
rf_3b = experiments["E1_RF_3B"]
rf_3b_correct = rf_3b["tp"] + rf_3b["tn"]

for model_name in ["E1_RF_7B", "E1_RF_8B", "E1_RF_12B", "E1_RF_14B",
                    "E1_RF_31B", "E1_RF_32B", "E1_RF_70B"]:
    exp = experiments[model_name]
    exp_correct = exp["tp"] + exp["tn"]
    z, p = two_proportion_z_test(exp_correct, exp["n_samples"],
                                  rf_3b_correct, rf_3b["n_samples"])
    h = cohens_h(exp["accuracy"], rf_3b["accuracy"])
    sig = p < 0.05
    label = f"{model_name} vs E1_RF_3B"
    print(f"  {label}: Acc {exp['accuracy']:.3f} vs {rf_3b['accuracy']:.3f}, "
          f"z={z:.3f}, p={p:.2e}, h={h:.3f} {'PASS' if sig else 'FAIL'}")
    add_test(label, "RQ6-scaling", "two-proportion z-test (Acc)",
             z, p, h, "Cohen's h", sig,
             f"N={exp['n_samples']} vs N={rf_3b['n_samples']}")

# Within-family scaling tests (Llama: 3B→8B→70B, Qwen: 7B→14B→32B)
print("\n--- Within-family scaling ---")
llama_models = ["E1_RF_3B", "E1_RF_8B", "E1_RF_70B"]
qwen_models = ["E1_RF_7B", "E1_RF_14B", "E1_RF_32B"]

for family, models in [("Llama", llama_models), ("Qwen", qwen_models)]:
    accs = [experiments[m]["accuracy"] for m in models]
    sizes = [int(m.split("_")[-1].replace("B", "")) for m in models]
    # Spearman correlation: accuracy vs size
    rho, p_rho = stats.spearmanr(sizes, accs)
    print(f"  {family} scaling: sizes={sizes}, accs={[f'{a:.3f}' for a in accs]}")
    print(f"    Spearman rho={rho:.3f}, p={p_rho:.3f}")
    add_test(f"{family} size→acc trend", "RQ6-scaling", "Spearman correlation",
             rho, p_rho, abs(rho), "Spearman rho", p_rho < 0.05,
             f"Models: {', '.join(models)}")

# Cross-family at similar scale: Llama-8B vs Qwen-7B
print("\n--- Cross-family at similar scale ---")
for pair in [("E1_RF_8B", "E1_RF_7B"), ("E1_RF_8B", "E1_RF_12B")]:
    a, b = experiments[pair[0]], experiments[pair[1]]
    v_a = load_audit_verdicts(file_map[pair[0]][1])
    v_b = load_audit_verdicts(file_map[pair[1]][1])
    if v_a and v_b and len(v_a) == len(v_b):
        chi2, p = mcnemar_test(v_a, v_b)
        h = cohens_h(a["accuracy"], b["accuracy"])
        print(f"  {pair[0]} vs {pair[1]}: McNemar chi2={chi2}, p={p:.4f}, h={h:.3f}")
        add_test(f"{pair[0]} vs {pair[1]}", "RQ6-family", "McNemar (paired)",
                 chi2 if chi2 else 0, p, h, "Cohen's h", p < 0.05,
                 "Same 200-sample subset, paired test")
    else:
        a_correct = a["tp"] + a["tn"]
        b_correct = b["tp"] + b["tn"]
        z, p = two_proportion_z_test(a_correct, a["n_samples"], b_correct, b["n_samples"])
        h = cohens_h(a["accuracy"], b["accuracy"])
        print(f"  {pair[0]} vs {pair[1]}: z={z:.3f}, p={p:.2e}, h={h:.3f}")
        add_test(f"{pair[0]} vs {pair[1]}", "RQ6-family", "two-proportion z-test",
                 z, p, h, "Cohen's h", p < 0.05)


# ============================================================
# RQ6: Harness Delta (E3 vs E1-RF at each scale)
# ============================================================
print("\n" + "=" * 70)
print("RQ6: Harness Delta — E3 vs E1-RF at Each Scale")
print("=" * 70)

harness_pairs = [
    ("E3_3B", "E1_RF_3B", "3B"),
    ("E3_8B", "E1_RF_7B", "7B"),  # E3 uses Qwen-7B with harness
    ("E3_14B", "E1_RF_14B", "14B"),
    ("E3_32B", "E1_RF_32B", "32B"),
    ("E3_70B", "E1_RF_70B", "70B"),
]

for e3_name, e1rf_name, size_label in harness_pairs:
    e3 = experiments[e3_name]
    e1rf = experiments[e1rf_name]

    # Try paired McNemar if audit logs available
    v_e3 = load_audit_verdicts(file_map[e3_name][1]) if file_map[e3_name][1] else []
    v_e1rf = load_audit_verdicts(file_map[e1rf_name][1]) if file_map[e1rf_name][1] else []

    if v_e3 and v_e1rf and len(v_e3) == len(v_e1rf):
        chi2, p = mcnemar_test(v_e3, v_e1rf)
        mean_d, ci_lo, ci_hi = bootstrap_acc_diff(v_e3, v_e1rf)
        test_used = "McNemar (paired)"
        stat_val = chi2 if chi2 else 0
        note = f"Paired N={len(v_e3)}, bootstrap Δ acc: {mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]"
    else:
        # Fall back to proportion test
        e3_correct = e3["tp"] + e3["tn"]
        e1rf_correct = e1rf["tp"] + e1rf["tn"]
        stat_val, p = two_proportion_z_test(e3_correct, e3["n_samples"],
                                             e1rf_correct, e1rf["n_samples"])
        test_used = "two-proportion z-test"
        note = f"N={e3['n_samples']} vs N={e1rf['n_samples']}"

    h = cohens_h(e3["accuracy"], e1rf["accuracy"])
    delta_acc = e3["accuracy"] - e1rf["accuracy"]
    delta_fpr = e3["fpr"] - e1rf["fpr"]

    print(f"\n  {size_label}: {e3_name} vs {e1rf_name}")
    print(f"    Δ Acc: {delta_acc:+.3f}, Δ FPR: {delta_fpr:+.3f}")
    print(f"    {test_used}: stat={stat_val}, p={p:.4f}, h={h:.3f}")
    if "bootstrap" in note:
        print(f"    {note}")

    add_test(f"Harness delta {size_label}: {e3_name} vs {e1rf_name}", "RQ6-delta",
             test_used, stat_val, p, h, "Cohen's h", p < 0.05, note)

    # FPR delta test
    e3_neg = e3["fp"] + e3["tn"]
    e1rf_neg = e1rf["fp"] + e1rf["tn"]
    if e3_neg > 0 and e1rf_neg > 0:
        z_fpr, p_fpr = two_proportion_z_test(e3["fp"], e3_neg, e1rf["fp"], e1rf_neg)
        h_fpr = cohens_h(e3["fpr"], e1rf["fpr"])
        add_test(f"Harness delta FPR {size_label}", "RQ6-delta", "two-proportion z-test (FPR)",
                 z_fpr, p_fpr, h_fpr, "Cohen's h", p_fpr < 0.05,
                 f"FPR: {e3['fpr']:.3f} vs {e1rf['fpr']:.3f}")


# ============================================================
# Summary: Kruskal-Wallis across all E1-RF models (excluding 31B failure)
# ============================================================
print("\n" + "=" * 70)
print("Kruskal-Wallis: E1-RF accuracy across model sizes (excl. 31B)")
print("=" * 70)

rf_models_no31b = ["E1_RF_7B", "E1_RF_8B", "E1_RF_12B", "E1_RF_14B", "E1_RF_32B", "E1_RF_70B"]
# Per-sample verdicts for KW test
groups = []
for m in rf_models_no31b:
    v = load_audit_verdicts(file_map[m][1])
    if v:
        groups.append(v)
        print(f"  {m}: N={len(v)}, acc={sum(v)/len(v):.3f}")

if len(groups) >= 3:
    H, p_kw = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis H={H:.3f}, p={p_kw:.4f}")
    add_test("E1-RF all models (excl 31B)", "RQ6-scaling", "Kruskal-Wallis",
             H, p_kw, 0, "H statistic", p_kw < 0.05,
             "6 models, per-sample verdicts")


# ============================================================
# Save results
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return super().default(obj)

# Sanitize nan/numpy types in all_tests
for t in all_tests:
    for k, v in t.items():
        if isinstance(v, float) and math.isnan(v):
            t[k] = None
        elif isinstance(v, (np.bool_, np.integer)):
            t[k] = int(v)
        elif isinstance(v, np.floating):
            t[k] = float(v)

with open(OUT_JSON, "w") as f:
    json.dump(all_tests, f, indent=2, cls=NumpyEncoder)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["comparison", "rq", "test", "statistic",
                                            "p_value", "effect_size", "effect_type",
                                            "significant", "notes"])
    writer.writeheader()
    writer.writerows(all_tests)

print(f"\n[OK] Saved {len(all_tests)} tests to:")
print(f"  {OUT_JSON}")
print(f"  {OUT_CSV}")
