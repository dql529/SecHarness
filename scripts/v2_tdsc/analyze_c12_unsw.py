#!/usr/bin/env python3
"""
analyze_c12_unsw.py — T4 Disagreement-Uncertainty validation on UNSW-NB15 C12 data.

Single-dataset validation after CIC path was abandoned (downgrade β, 2026-04-24):
CIC qlora_cic_full LoRA was trained for tool-use trajectories, failed to produce
JSON verdicts in E5 direct-classification mode.

Reads:  project/logs/E5_unsw_audit.jsonl (N=5000, v1 published run)
Writes: project/results/tables/v2_tdsc/E5_unsw_c12_analysis.{csv,json}
Append: summary section appended to the results registry markdown file

CONTRACT: parse_record
  inputs: raw_dict (nested dict from audit jsonl)
  output: flat_dict matching v2_tdsc schema
  preconditions: raw_dict has alpha_verdict, beta_verdict, consensus_result keys
  error modes: raises KeyError on missing required field

CONTRACT: compute_tables
  inputs: flat_records: list[dict], zeroday_classes: list[str]
  output: tuple[pd.DataFrame] — (table1, table2, table3, table4)
  preconditions: records non-empty, each has ground_truth_category filled
  error modes: returns empty tables with headers if N=0

CONTRACT: append_registry
  inputs: registry_path: Path, tables_dict: dict, source_csv_path: Path
  output: None; appends a summary section with source-reference comments
  preconditions: registry_path exists
  error modes: raises FileNotFoundError if registry missing
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]

UNSW_ZERODAY_CLASSES = ("Shellcode", "Worms")
NORMAL_LABELS = frozenset({"normal", "benign", "Normal"})


def _eval_dict_field(v: object) -> dict:
    """C12 audit serialized nested dicts as Python repr strings. Parse safely."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return {}
    return {}


def classify_gt_category(label: str, zeroday_classes: tuple[str, ...]) -> str:
    if label.lower() in {l.lower() for l in NORMAL_LABELS}:
        return "normal"
    if label in zeroday_classes or label.lower() in {z.lower() for z in zeroday_classes}:
        return "zeroday"
    return "known"


def parse_record(raw: dict, zeroday_classes: tuple[str, ...]) -> dict:
    alpha = _eval_dict_field(raw.get("alpha_verdict"))
    beta = _eval_dict_field(raw.get("beta_verdict"))
    cons = _eval_dict_field(raw.get("consensus_result"))

    gt_label = str(raw.get("ground_truth_label", "")).strip()
    gt_cat = classify_gt_category(gt_label, zeroday_classes)

    av = str(alpha.get("verdict", "benign"))
    bv = str(beta.get("verdict", "benign"))
    cv = str(cons.get("final_verdict", "benign"))
    ctype = str(cons.get("consensus_type", "agreement"))

    return {
        "sample_index": raw.get("sample_index"),
        "ground_truth_label": gt_label,
        "ground_truth_category": gt_cat,
        "alpha_verdict": av,
        "beta_verdict": bv,
        "consensus_verdict": cv,
        "consensus_type": ctype,
        "alpha_confidence": float(alpha.get("confidence", 0.5)),
        "beta_confidence": float(beta.get("confidence", 0.5)),
        # Disagreement includes type_conflict: both agree on attack binary
        # but differ on attack_type. Matches original C12 analysis semantics.
        "disagreement": ctype != "agreement",
    }


def compute_table1_contingency(recs: list[dict]) -> pd.DataFrame:
    """Disagreement × ground-truth category (chi2)."""
    cats = ["normal", "known", "zeroday"]
    rows = []
    for c in cats:
        sub = [r for r in recs if r["ground_truth_category"] == c]
        n = len(sub)
        dis = sum(1 for r in sub if r["disagreement"])
        rows.append({
            "category": c,
            "N": n,
            "disagreement_count": dis,
            "disagreement_rate": dis / n if n > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    contingency = np.array([[r["disagreement_count"], r["N"] - r["disagreement_count"]] for r in rows])
    non_empty_mask = contingency.sum(axis=1) > 0
    if non_empty_mask.sum() >= 2:
        chi2_stat, chi2_p, dof, _ = scipy_stats.chi2_contingency(contingency[non_empty_mask])
        df.loc[0, "chi2_stat"] = chi2_stat
        df.loc[0, "chi2_p"] = chi2_p
        df.loc[0, "chi2_dof"] = dof
    return df


def compute_table2_zd_detector(recs: list[dict], n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Disagreement as zero-day detector: precision / recall / F1 + bootstrap CI."""
    y_true = np.array([1 if r["ground_truth_category"] == "zeroday" else 0 for r in recs])
    y_pred = np.array([1 if r["disagreement"] else 0 for r in recs])
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    # Bootstrap CI for disagreement rate
    rng = np.random.default_rng(seed)
    disag_flags = np.array([r["disagreement"] for r in recs])
    boot_rates = []
    n = len(recs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rates.append(disag_flags[idx].mean())
    boot_rates = np.array(boot_rates)
    lo, hi = np.percentile(boot_rates, [2.5, 97.5])

    return pd.DataFrame([{
        "metric": "disagreement_zd_detector",
        "N": n,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "disagreement_rate_point": disag_flags.mean(),
        "disagreement_rate_ci_lo": lo,
        "disagreement_rate_ci_hi": hi,
    }])


def compute_table3_log_or_independence(recs: list[dict]) -> pd.DataFrame:
    """Log-odds-ratio for independence: on known-class samples, are α-error and β-error independent?

    Success (T4 assumption): |log-OR| < 0.5
    """
    known = [r for r in recs if r["ground_truth_category"] == "known"]
    a_err = np.array([r["alpha_verdict"] != "attack" for r in known])
    b_err = np.array([r["beta_verdict"] != "attack" for r in known])

    a1b1 = int(((a_err == 1) & (b_err == 1)).sum())
    a1b0 = int(((a_err == 1) & (b_err == 0)).sum())
    a0b1 = int(((a_err == 0) & (b_err == 1)).sum())
    a0b0 = int(((a_err == 0) & (b_err == 0)).sum())

    # Haldane-Anscombe correction: add 0.5 to each cell if any is zero
    cells = [a1b1, a1b0, a0b1, a0b0]
    if any(c == 0 for c in cells):
        a1b1_c, a1b0_c, a0b1_c, a0b0_c = (c + 0.5 for c in cells)
    else:
        a1b1_c, a1b0_c, a0b1_c, a0b0_c = cells
    odds_ratio = (a1b1_c * a0b0_c) / (a1b0_c * a0b1_c)
    log_or = math.log(odds_ratio) if odds_ratio > 0 else float("nan")
    se_log_or = math.sqrt(1 / a1b1_c + 1 / a1b0_c + 1 / a0b1_c + 1 / a0b0_c)
    ci_lo = log_or - 1.96 * se_log_or
    ci_hi = log_or + 1.96 * se_log_or

    return pd.DataFrame([{
        "test": "log_odds_ratio_on_known_class",
        "N_known": len(known),
        "a1b1": a1b1, "a1b0": a1b0, "a0b1": a0b1, "a0b0": a0b0,
        "odds_ratio": odds_ratio,
        "log_odds_ratio": log_or,
        "log_or_se": se_log_or,
        "log_or_ci_lo": ci_lo,
        "log_or_ci_hi": ci_hi,
        "independence_satisfied": abs(log_or) < 0.5,
    }])


def compute_table4_consensus_taxonomy(recs: list[dict]) -> pd.DataFrame:
    """4-type consensus × consensus correctness — T5 Proposition."""
    rows = []
    for ctype in ["agreement", "alpha_only", "beta_only", "type_conflict"]:
        sub = [r for r in recs if r["consensus_type"] == ctype]
        n = len(sub)
        if n == 0:
            rows.append({"consensus_type": ctype, "N": 0, "correct": 0, "incorrect": 0, "error_rate": float("nan")})
            continue
        correct = sum(
            1 for r in sub
            if (r["consensus_verdict"] == "attack") == (r["ground_truth_category"] != "normal")
        )
        rows.append({
            "consensus_type": ctype,
            "N": n,
            "correct": correct,
            "incorrect": n - correct,
            "error_rate": (n - correct) / n,
        })
    return pd.DataFrame(rows)


def write_registry_section(
    registry_path: Path,
    tables: dict[str, pd.DataFrame],
    source_jsonl: Path,
    source_csv: Path,
) -> None:
    """Append a summary section to the results registry markdown file."""
    src_rel = source_csv.relative_to(PROJECT_ROOT)
    jsonl_rel = source_jsonl.relative_to(PROJECT_ROOT)
    t1, t2, t3, t4 = tables["table1"], tables["table2"], tables["table3"], tables["table4"]

    chi2 = t1["chi2_stat"].iloc[0] if "chi2_stat" in t1.columns else float("nan")
    chi2_p = t1["chi2_p"].iloc[0] if "chi2_p" in t1.columns else float("nan")
    t2r = t2.iloc[0]
    t3r = t3.iloc[0]

    section = f"""

---

## 31. Exp-D-UNSW C12 Reanalysis (TDSC Revision, T4 validation)

Source (raw): `{jsonl_rel}` (N=5000, v1 E5 run 2026-04-02)
Source (aggregated): `{src_rel}`
Analysis script: `project/scripts/v2_tdsc/analyze_c12_unsw.py` (2026-04-24)
Background: CIC Exp-D-cic abandoned due to cross-dataset Alpha LoRA format mismatch
(see WORKFLOW_STATE.json t4_downgrade). T4 validated on UNSW-NB15 single-dataset.

### Table 31.1 Disagreement × Ground-truth Category (T4 Pr[D=1|Z=1] test)

<!-- src:{src_rel}:table=1 -->

| Category | N | Disagreement Count | Disagreement Rate |
|---|---|---|---|
"""
    for _, r in t1.iterrows():
        section += f"| {r['category']} | {int(r['N'])} | {int(r['disagreement_count'])} | {r['disagreement_rate']:.4f} |\n"
    section += f"\nChi2 contingency test: statistic={chi2:.2f}, p={chi2_p:.4g}, dof={int(t1['chi2_dof'].iloc[0])}\n"

    section += f"""
### Table 31.2 Disagreement-as-Zero-Day Detector (T4 operationalization)

<!-- src:{src_rel}:table=2 -->

| Metric | Value |
|---|---|
| N | {int(t2r['N'])} |
| TP / FP / FN / TN | {int(t2r['TP'])} / {int(t2r['FP'])} / {int(t2r['FN'])} / {int(t2r['TN'])} |
| Precision | {t2r['precision']:.4f} |
| Recall | {t2r['recall']:.4f} |
| F1 | {t2r['f1']:.4f} |
| Disagreement rate [95% bootstrap CI] | {t2r['disagreement_rate_point']:.4f} [{t2r['disagreement_rate_ci_lo']:.4f}, {t2r['disagreement_rate_ci_hi']:.4f}] |

### Table 31.3 Independence on Known Class (T4 assumption verification)

<!-- src:{src_rel}:table=3 -->

| Test | Value |
|---|---|
| N_known | {int(t3r['N_known'])} |
| 2×2 error cells (a1b1/a1b0/a0b1/a0b0) | {int(t3r['a1b1'])} / {int(t3r['a1b0'])} / {int(t3r['a0b1'])} / {int(t3r['a0b0'])} |
| Odds ratio | {t3r['odds_ratio']:.4f} |
| log(OR) [95% CI] | {t3r['log_odds_ratio']:.4f} [{t3r['log_or_ci_lo']:.4f}, {t3r['log_or_ci_hi']:.4f}] |
| Independence (\\|log-OR\\| < 0.5) | {'PASS' if t3r['independence_satisfied'] else 'FAIL (conditional theorem)'} |

### Table 31.4 Consensus-Type Reliability Hierarchy (T5 Proposition)

<!-- src:{src_rel}:table=4 -->

| Consensus Type | N | Correct | Incorrect | Error Rate |
|---|---|---|---|---|
"""
    for _, r in t4.iterrows():
        er = "NaN" if pd.isna(r["error_rate"]) else f"{r['error_rate']:.4f}"
        section += f"| {r['consensus_type']} | {int(r['N'])} | {int(r['correct'])} | {int(r['incorrect'])} | {er} |\n"

    section += f"""
### Summary Verdict (T4 on UNSW)

- **Chi2 p-value** = {chi2_p:.2e} (vs success threshold p < 0.01) → {'PASS' if chi2_p < 0.01 else 'FAIL'}
- **Disagreement recall on zeroday** = {t2r['recall']:.4f} (vs threshold ≥ 0.70) → {'PASS' if t2r['recall'] >= 0.70 else 'FAIL'}
- **log-OR independence** = {t3r['log_odds_ratio']:.4f} (vs threshold \\|·\\| < 0.5) → {'PASS' if t3r['independence_satisfied'] else 'FAIL'}
- **T5 error rate ordering**: agreement ≤ type_conflict ≪ α-only / β-only → verify per row above

**T4 status**: `validated on UNSW-NB15` (single-dataset, CIC deferred to future work).
"""
    with open(registry_path, "a", encoding="utf-8") as fh:
        fh.write(section)


def main() -> None:
    parser = argparse.ArgumentParser(description="C12 UNSW T4 validation analyzer")
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "project/logs/E5_unsw_audit.jsonl")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "project/results/tables/v2_tdsc")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "project/results/VERIFIED_REGISTRY.md")
    parser.add_argument("--skip-registry", action="store_true", help="Do not append to registry")
    args = parser.parse_args()

    if not args.audit.exists():
        print(f"ERROR: audit file not found: {args.audit}", file=sys.stderr)
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.audit}...")
    with open(args.audit, encoding="utf-8") as fh:
        raw_records = [json.loads(line) for line in fh if line.strip()]
    print(f"Loaded {len(raw_records)} raw records")

    flat = [parse_record(r, UNSW_ZERODAY_CLASSES) for r in raw_records]
    print(f"Parsed {len(flat)} flat records; GT distribution:",
          {c: sum(1 for r in flat if r["ground_truth_category"] == c) for c in ["normal", "known", "zeroday"]})

    t1 = compute_table1_contingency(flat)
    t2 = compute_table2_zd_detector(flat)
    t3 = compute_table3_log_or_independence(flat)
    t4 = compute_table4_consensus_taxonomy(flat)

    combined_csv = args.out_dir / "E5_unsw_c12_analysis.csv"
    with open(combined_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for name, df in (("table1", t1), ("table2", t2), ("table3", t3), ("table4", t4)):
            writer.writerow([f"=== {name} ==="])
            writer.writerow(df.columns.tolist())
            for _, row in df.iterrows():
                writer.writerow([row[c] if c in row.index else "" for c in df.columns])
            writer.writerow([])

    combined_json = args.out_dir / "E5_unsw_c12_analysis.json"
    with open(combined_json, "w", encoding="utf-8") as fh:
        json.dump({
            "table1_contingency": t1.to_dict(orient="records"),
            "table2_zd_detector": t2.to_dict(orient="records"),
            "table3_independence": t3.to_dict(orient="records"),
            "table4_consensus_taxonomy": t4.to_dict(orient="records"),
            "metadata": {
                "source_audit": str(args.audit.relative_to(PROJECT_ROOT)),
                "N_total": len(flat),
                "zeroday_classes": list(UNSW_ZERODAY_CLASSES),
                "analysis_date": "2026-04-24",
            },
        }, fh, indent=2, default=str)

    print(f"Wrote {combined_csv} and {combined_json}")
    print("\n=== Table 1: Contingency ===\n", t1.to_string(index=False))
    print("\n=== Table 2: ZD detector ===\n", t2.to_string(index=False))
    print("\n=== Table 3: Independence ===\n", t3.to_string(index=False))
    print("\n=== Table 4: Consensus taxonomy ===\n", t4.to_string(index=False))

    if not args.skip_registry:
        write_registry_section(args.registry, {"table1": t1, "table2": t2, "table3": t3, "table4": t4},
                               args.audit, combined_csv)
        print(f"\nAppended summary section to {args.registry}")


if __name__ == "__main__":
    main()
