#!/usr/bin/env python3
"""
analyze_disagreement_cic.py — T4 Disagreement-Uncertainty statistical analysis.

Reads audit JSONL across seeds and emits 4 tables testing the T4 hypothesis:
  Pr[D=1 | Z=1] >= 1 - sum Pr[A_alpha=y|Z=1]*Pr[A_beta=y|Z=1]
  + independence: |log-OR(alpha-err, beta-err | Z=0)| < 0.5

Outputs:
  - Table 1: per-seed disagreement counts by (category x disagreement) with chi2
  - Table 2: 3-seed aggregated disagreement rate [95% CI bootstrap], zeroday
              detection via disagreement P/R/F1, chi2 combined (Fisher's method)
  - Table 3: log-odds-ratio(alpha-err, beta-err | Z=0) per seed + aggregated
  - Table 4: 5-type consensus breakdown matching C12 format

CONTRACT: load_audit_jsonl
  inputs: path:Path — JSONL audit log
  output: list[dict]
  preconditions: file exists
  error modes: raises FileNotFoundError if missing

CONTRACT: compute_table1
  inputs: records:list[dict], seed:int
  output: dict — {seed, category_counts, chi2_stat, chi2_p, chi2_df}
  preconditions: records non-empty
  error modes: returns NaN stats on insufficient data

CONTRACT: compute_table2
  inputs: per_seed_records:dict[int, list[dict]], seeds:list[int], n_bootstrap:int
  output: dict — {disagree_rate_mean, disagree_rate_ci, zeroday_precision,
                  zeroday_recall, zeroday_f1, fisher_chi2, fisher_p}
  preconditions: at least 1 seed has records
  error modes: NaN on degenerate inputs

CONTRACT: compute_table3
  inputs: per_seed_records:dict[int, list[dict]], seeds:list[int]
  output: dict — per-seed and aggregated log-OR with 95% CI
  preconditions: records contain alpha/beta verdict fields
  error modes: NaN if no normal (Z=0) samples

CONTRACT: compute_table4
  inputs: all_records:list[dict]
  output: dict — per consensus_type counts and error rates
  preconditions: records contain consensus_type field
  error modes: empty categories return 0 counts

CONTRACT: main
  inputs: CLI argv
  output: None; writes CSV+JSON and appends to the results registry markdown file
  error modes: SystemExit(1) on missing input files
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_disagreement_cic")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NORMAL_LABELS: frozenset[str] = frozenset({"normal", "benign", "BENIGN"})
CONSENSUS_TYPES: tuple[str, ...] = (
    "agreement", "alpha_only", "beta_only", "type_conflict", "uncertain"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_audit_jsonl(path: Path) -> list[dict]:
    """Load audit records from a JSONL file.

    CONTRACT: see module docstring.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Table 1: per-seed disagreement counts by category x disagreement
# ---------------------------------------------------------------------------

def compute_table1(records: list[dict], seed: int) -> dict:
    """Per-seed disagree counts by (category x disagreement) with chi2 test.

    CONTRACT: see module docstring.
    """
    categories = ("zeroday", "known", "normal")
    counts: dict[str, dict[str, int]] = {
        cat: {"disagree": 0, "agree": 0, "total": 0}
        for cat in categories
    }

    for r in records:
        cat = r.get("ground_truth_category", "known")
        if cat not in counts:
            cat = "known"
        d = bool(r["disagreement"])
        counts[cat]["total"] += 1
        if d:
            counts[cat]["disagree"] += 1
        else:
            counts[cat]["agree"] += 1

    # Chi2 contingency: disagreement x (zeroday vs not-zeroday)
    zd_dis = counts["zeroday"]["disagree"]
    zd_agr = counts["zeroday"]["agree"]
    nzd_dis = counts["known"]["disagree"] + counts["normal"]["disagree"]
    nzd_agr = counts["known"]["agree"] + counts["normal"]["agree"]

    chi2_stat: float = float("nan")
    chi2_p: float = float("nan")
    chi2_df: int = 1

    if (zd_dis + zd_agr) > 0 and (nzd_dis + nzd_agr) > 0:
        try:
            contingency = [[zd_dis, zd_agr], [nzd_dis, nzd_agr]]
            res = scipy_stats.chi2_contingency(contingency, correction=False)
            chi2_stat = float(res.statistic)
            chi2_p = float(res.pvalue)
        except Exception as exc:
            log.warning("Table1 chi2 failed (seed=%d): %s", seed, exc)

    return {
        "seed": seed,
        "category_counts": counts,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "chi2_df": chi2_df,
        "total": len(records),
    }


# ---------------------------------------------------------------------------
# Table 2: 3-seed aggregated stats
# ---------------------------------------------------------------------------

def compute_table2(
    per_seed_records: dict[int, list[dict]],
    seeds: list[int],
    n_bootstrap: int = 10000,
) -> dict:
    """3-seed aggregated disagreement rate + bootstrap CI + Fisher chi2.

    CONTRACT: see module docstring.
    """
    all_records = [r for s in seeds for r in per_seed_records.get(s, [])]
    if not all_records:
        return {
            "disagree_rate_mean": float("nan"),
            "disagree_rate_ci_lo": float("nan"),
            "disagree_rate_ci_hi": float("nan"),
            "zeroday_precision": float("nan"),
            "zeroday_recall": float("nan"),
            "zeroday_f1": float("nan"),
            "fisher_chi2": float("nan"),
            "fisher_p": float("nan"),
            "N_total": 0,
        }

    n_total = len(all_records)
    disagree_flags = np.array([1 if r["disagreement"] else 0 for r in all_records])

    # Bootstrap 95% CI on disagreement rate
    rng = np.random.default_rng(42)
    boot_rates = np.array([
        rng.choice(disagree_flags, size=len(disagree_flags), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo = float(np.percentile(boot_rates, 2.5))
    ci_hi = float(np.percentile(boot_rates, 97.5))
    disagree_rate_mean = float(disagree_flags.mean())

    # Zeroday detection via disagreement: treat D=1 as "predicted zeroday"
    # Precision = P(Z=1 | D=1), Recall = P(D=1 | Z=1), F1
    zd_and_d = sum(1 for r in all_records if r["disagreement"] and r.get("ground_truth_category") == "zeroday")
    total_d = sum(1 for r in all_records if r["disagreement"])
    total_zd = sum(1 for r in all_records if r.get("ground_truth_category") == "zeroday")

    precision = zd_and_d / total_d if total_d > 0 else float("nan")
    recall = zd_and_d / total_zd if total_zd > 0 else float("nan")
    if not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    # Fisher's method to combine per-seed chi2 p-values
    chi2_pvals: list[float] = []
    fisher_degenerate = False  # set True if any p==0 (chi2→inf, combined p→0)
    for seed in seeds:
        recs = per_seed_records.get(seed, [])
        if recs:
            t1 = compute_table1(recs, seed)
            p = t1["chi2_p"]
            if math.isnan(p):
                continue
            if p == 0.0:
                # p=0 makes log(p) undefined; Fisher stat diverges to +inf,
                # so combined p-value is exactly 0 — record and short-circuit.
                fisher_degenerate = True
                break
            chi2_pvals.append(p)

    fisher_chi2: float = float("nan")
    fisher_p: float = float("nan")
    if fisher_degenerate:
        fisher_chi2 = float("inf")
        fisher_p = 0.0
    elif chi2_pvals:
        fisher_stat = -2.0 * sum(math.log(p) for p in chi2_pvals)
        fisher_df = 2 * len(chi2_pvals)
        fisher_chi2 = fisher_stat
        fisher_p = float(1.0 - scipy_stats.chi2.cdf(fisher_stat, df=fisher_df))

    return {
        "disagree_rate_mean": disagree_rate_mean,
        "disagree_rate_ci_lo": ci_lo,
        "disagree_rate_ci_hi": ci_hi,
        "zeroday_precision": precision,
        "zeroday_recall": recall,
        "zeroday_f1": f1,
        "fisher_chi2": fisher_chi2,
        "fisher_p": fisher_p,
        "N_total": n_total,
    }


# ---------------------------------------------------------------------------
# Table 3: log-odds-ratio(alpha-err, beta-err | Z=0) — independence test
# ---------------------------------------------------------------------------

def _log_or_with_ci(
    a: int, b: int, c: int, d: int
) -> tuple[float, float, float]:
    """Compute log odds ratio and 95% CI (Woolf method with Haldane correction).

    Returns (log_or, ci_lo, ci_hi). Uses 0.5 correction for zero cells.
    """
    # Haldane-Anscombe correction for zero cells
    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log(ac * dc / (bc * cc))
    se = math.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)
    ci_lo = log_or - 1.96 * se
    ci_hi = log_or + 1.96 * se
    return log_or, ci_lo, ci_hi


def compute_table3(
    per_seed_records: dict[int, list[dict]],
    seeds: list[int],
) -> dict:
    """Log-OR(alpha-err, beta-err | Z=0) per seed and aggregated.

    Filters to Z=0 (normal/known) samples only. Alpha-error = alpha predicted
    attack but ground-truth is not attack. Beta-error similarly.

    CONTRACT: see module docstring.
    """
    results: dict[str, object] = {"per_seed": {}, "aggregated": {}}
    all_normal_records: list[dict] = []

    for seed in seeds:
        recs = per_seed_records.get(seed, [])
        # Z=0: normal samples (not zero-day)
        z0_recs = [
            r for r in recs
            if r.get("ground_truth_category") in ("normal", "known")
        ]
        all_normal_records.extend(z0_recs)

        if not z0_recs:
            results["per_seed"][seed] = {  # type: ignore[index]
                "N_z0": 0, "log_or": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
            }
            continue

        # Alpha-error: alpha says attack but Z=0 (false positive from alpha)
        # Beta-error: beta says attack but Z=0
        alpha_err = [1 if r["alpha_verdict"] == "attack" else 0 for r in z0_recs]
        beta_err = [1 if r["beta_verdict"] == "attack" else 0 for r in z0_recs]

        # 2x2 contingency: alpha-err x beta-err
        a = sum(1 for ae, be in zip(alpha_err, beta_err) if ae == 1 and be == 1)
        b = sum(1 for ae, be in zip(alpha_err, beta_err) if ae == 1 and be == 0)
        c = sum(1 for ae, be in zip(alpha_err, beta_err) if ae == 0 and be == 1)
        d = sum(1 for ae, be in zip(alpha_err, beta_err) if ae == 0 and be == 0)

        log_or, ci_lo, ci_hi = _log_or_with_ci(a, b, c, d)

        results["per_seed"][seed] = {  # type: ignore[index]
            "N_z0": len(z0_recs),
            "alpha_err_rate": sum(alpha_err) / len(alpha_err),
            "beta_err_rate": sum(beta_err) / len(beta_err),
            "contingency": {"a": a, "b": b, "c": c, "d": d},
            "log_or": log_or,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "independence_pass": abs(log_or) < 0.5,
        }

    # Aggregated across seeds
    if all_normal_records:
        alpha_err_all = [1 if r["alpha_verdict"] == "attack" else 0 for r in all_normal_records]
        beta_err_all = [1 if r["beta_verdict"] == "attack" else 0 for r in all_normal_records]
        a = sum(1 for ae, be in zip(alpha_err_all, beta_err_all) if ae == 1 and be == 1)
        b = sum(1 for ae, be in zip(alpha_err_all, beta_err_all) if ae == 1 and be == 0)
        c = sum(1 for ae, be in zip(alpha_err_all, beta_err_all) if ae == 0 and be == 1)
        d = sum(1 for ae, be in zip(alpha_err_all, beta_err_all) if ae == 0 and be == 0)
        log_or, ci_lo, ci_hi = _log_or_with_ci(a, b, c, d)
        results["aggregated"] = {
            "N_z0": len(all_normal_records),
            "log_or": log_or,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "independence_pass": abs(log_or) < 0.5,
        }
    else:
        results["aggregated"] = {
            "N_z0": 0, "log_or": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "independence_pass": False,
        }

    return results


# ---------------------------------------------------------------------------
# Table 4: consensus breakdown matching C12 format
# ---------------------------------------------------------------------------

def compute_table4(all_records: list[dict]) -> dict:
    """5-type consensus breakdown with per-type consensus-level error rate.

    Uses consensus_verdict (the actual final consensus decision) for correctness,
    with backward-compatible fallback to alpha_verdict for older audit logs that
    predate the consensus_verdict field.

    CONTRACT: see module docstring.
    """
    type_stats: dict[str, dict[str, int]] = {
        ct: {"count": 0, "correct": 0, "incorrect": 0}
        for ct in CONSENSUS_TYPES
    }

    for r in all_records:
        ctype = r.get("consensus_type", "agreement")
        if ctype not in type_stats:
            ctype = "agreement"
        type_stats[ctype]["count"] += 1

        # Determine correctness: consensus-level error rate, not alpha-only.
        # consensus_verdict is the actual final decision (may be "escalate" under
        # escalate_all strategy, or alpha/beta verdict under priority strategies).
        # Falls back to alpha_verdict for audit logs written before this field existed.
        gt_cat = r.get("ground_truth_category", "known")
        gt_is_attack = gt_cat != "normal"
        pred_verdict = r.get("consensus_verdict", r.get("alpha_verdict", "benign"))
        pred_is_attack = pred_verdict == "attack"

        if gt_is_attack == pred_is_attack:
            type_stats[ctype]["correct"] += 1
        else:
            type_stats[ctype]["incorrect"] += 1

    for ctype in CONSENSUS_TYPES:
        stats = type_stats[ctype]
        total = stats["count"]
        stats["error_rate"] = (  # type: ignore[assignment]
            stats["incorrect"] / total if total > 0 else float("nan")
        )

    return {"by_type": type_stats, "N_total": len(all_records)}


# ---------------------------------------------------------------------------
# Registry append
# ---------------------------------------------------------------------------

def append_to_registry(
    registry_path: Path,
    table1_list: list[dict],
    table2: dict,
    table3: dict,
    table4: dict,
    analysis_csv: Path,
    analysis_json: Path,
) -> None:
    """Append a summary section to the results registry markdown file.

    CONTRACT: appends new section; does not overwrite existing content.
    """
    section = f"""
## 14. Exp-D-cic (TDSC Revision)

**Source**: `{analysis_csv.relative_to(PROJECT_ROOT)}`
**Analysis script**: `project/scripts/v2_tdsc/analyze_disagreement_cic.py`

### Table 1: Per-seed disagreement by category × disagreement (chi2 test)

| Seed | N | ZD_disagree | ZD_agree | nonZD_disagree | nonZD_agree | chi2_stat | chi2_p |
|------|---|-------------|----------|----------------|-------------|-----------|--------|
"""
    for t1 in table1_list:
        cats = t1["category_counts"]
        zd_dis = cats.get("zeroday", {}).get("disagree", 0)
        zd_agr = cats.get("zeroday", {}).get("agree", 0)
        nzd_dis = cats.get("known", {}).get("disagree", 0) + cats.get("normal", {}).get("disagree", 0)
        nzd_agr = cats.get("known", {}).get("agree", 0) + cats.get("normal", {}).get("agree", 0)
        chi2s = f"{t1['chi2_stat']:.4f}" if not math.isnan(t1["chi2_stat"]) else "nan"
        chi2p = f"{t1['chi2_p']:.4f}" if not math.isnan(t1["chi2_p"]) else "nan"
        section += f"| {t1['seed']} | {t1['total']} | {zd_dis} | {zd_agr} | {nzd_dis} | {nzd_agr} | {chi2s} | {chi2p} |\n"

    section += f"""<!-- src:{analysis_csv.relative_to(PROJECT_ROOT)}:row=table1,col=chi2_p -->

### Table 2: 3-seed aggregated disagreement + zeroday detection via disagreement

| Metric | Value | 95% CI |
|--------|-------|--------|
| Disagreement rate | {table2['disagree_rate_mean']:.4f} | [{table2['disagree_rate_ci_lo']:.4f}, {table2['disagree_rate_ci_hi']:.4f}] |
| Zeroday precision (D=1 → Z=1) | {table2['zeroday_precision']:.4f} | — |
| Zeroday recall (Z=1 → D=1) | {table2['zeroday_recall']:.4f} | — |
| Zeroday F1 via disagreement | {table2['zeroday_f1']:.4f} | — |
| Fisher combined chi2 | {table2['fisher_chi2']:.4f} (p={table2['fisher_p']:.4f}) | — |
| N total | {table2['N_total']} | — |
<!-- src:{analysis_csv.relative_to(PROJECT_ROOT)}:row=table2,col=zeroday_f1 -->

### Table 3: Log-odds-ratio(alpha-err, beta-err | Z=0) — Independence Test

| Seed | N_z0 | log_OR | 95% CI | |log_OR|<0.5 |
|------|------|--------|--------|------------|
"""
    per_seed = table3.get("per_seed", {})
    for seed, sd in per_seed.items():
        if isinstance(sd, dict) and sd.get("N_z0", 0) > 0:
            lor = f"{sd['log_or']:.4f}" if not math.isnan(sd["log_or"]) else "nan"
            ci = f"[{sd['ci_lo']:.4f}, {sd['ci_hi']:.4f}]" if not math.isnan(sd.get("ci_lo", float("nan"))) else "—"
            ipass = "PASS" if sd.get("independence_pass") else "FAIL"
            section += f"| {seed} | {sd['N_z0']} | {lor} | {ci} | {ipass} |\n"

    agg = table3.get("aggregated", {})
    if isinstance(agg, dict) and agg.get("N_z0", 0) > 0:
        lor = f"{agg['log_or']:.4f}" if not math.isnan(agg["log_or"]) else "nan"
        ci = f"[{agg['ci_lo']:.4f}, {agg['ci_hi']:.4f}]" if not math.isnan(agg.get("ci_lo", float("nan"))) else "—"
        ipass = "PASS" if agg.get("independence_pass") else "FAIL"
        section += f"| **aggregated** | {agg['N_z0']} | {lor} | {ci} | {ipass} |\n"

    section += f"""<!-- src:{analysis_csv.relative_to(PROJECT_ROOT)}:row=table3,col=log_or -->

### Table 4: Consensus type breakdown (C12 format)

| consensus_type | count | correct | incorrect | error_rate |
|----------------|-------|---------|-----------|------------|
"""
    by_type = table4.get("by_type", {})
    for ct in CONSENSUS_TYPES:
        s = by_type.get(ct, {"count": 0, "correct": 0, "incorrect": 0, "error_rate": float("nan")})
        er = f"{s.get('error_rate', float('nan')):.4f}" if not math.isnan(s.get("error_rate", float("nan"))) else "nan"
        section += f"| {ct} | {s['count']} | {s['correct']} | {s['incorrect']} | {er} |\n"

    section += f"""<!-- src:{analysis_csv.relative_to(PROJECT_ROOT)}:row=table4,col=error_rate -->
"""

    # Append to registry
    with open(registry_path, "a", encoding="utf-8") as fout:
        fout.write(section)
    log.info("Registry section 14 appended to %s", registry_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze Exp-D-cic disagreement logs and emit 4 T4 tables"
    )
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=PROJECT_ROOT / "project/logs/v2_tdsc",
        help="Directory containing audit_E5_cic_tdsc_seed*.jsonl files",
    )
    ap.add_argument(
        "--prefix",
        default="E5_cic_tdsc",
        help="Audit log prefix (default: E5_cic_tdsc)",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated seeds to load (default: 42,123,456)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "project/results/tables/v2_tdsc",
        help="Output directory for CSV/JSON",
    )
    ap.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "project/results/VERIFIED_REGISTRY.md",
        help="Path to the results registry markdown file to append a summary section to",
    )
    ap.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip registry append (useful for testing)",
    )
    args = ap.parse_args()

    seeds: list[int] = [int(s.strip()) for s in args.seeds.split(",")]
    logs_dir: Path = args.logs_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-seed records
    per_seed_records: dict[int, list[dict]] = {}
    for seed in seeds:
        audit_path = logs_dir / f"audit_{args.prefix}_seed{seed}.jsonl"
        if not audit_path.exists():
            log.error("Audit log not found: %s", audit_path)
            sys.exit(1)
        recs = load_audit_jsonl(audit_path)
        per_seed_records[seed] = recs
        log.info("Loaded seed=%d: %d records from %s", seed, len(recs), audit_path)

    all_records: list[dict] = [r for s in seeds for r in per_seed_records.get(s, [])]
    log.info("Total records: %d", len(all_records))

    # Compute tables
    table1_list = [compute_table1(per_seed_records[s], s) for s in seeds]
    table2 = compute_table2(per_seed_records, seeds)
    table3 = compute_table3(per_seed_records, seeds)
    table4 = compute_table4(all_records)

    # Log key results
    log.info("Table 2 — disagree_rate=%.4f [%.4f, %.4f], zeroday_recall=%.4f, fisher_p=%.4f",
             table2["disagree_rate_mean"], table2["disagree_rate_ci_lo"], table2["disagree_rate_ci_hi"],
             table2["zeroday_recall"], table2["fisher_p"])
    agg = table3.get("aggregated", {})
    if isinstance(agg, dict) and agg.get("N_z0", 0) > 0:
        log.info("Table 3 — aggregated log_OR=%.4f, |log_OR|<0.5: %s",
                 agg["log_or"], agg.get("independence_pass"))

    # Write CSV (flat summary)
    analysis_csv = output_dir / f"{args.prefix}_analysis.csv"
    csv_rows: list[dict] = []
    for t1 in table1_list:
        cats = t1["category_counts"]
        csv_rows.append({
            "table": "1_per_seed_chi2",
            "seed": t1["seed"],
            "total": t1["total"],
            "zeroday_disagree": cats.get("zeroday", {}).get("disagree", 0),
            "zeroday_agree": cats.get("zeroday", {}).get("agree", 0),
            "known_disagree": cats.get("known", {}).get("disagree", 0),
            "normal_disagree": cats.get("normal", {}).get("disagree", 0),
            "chi2_stat": t1["chi2_stat"],
            "chi2_p": t1["chi2_p"],
        })
    csv_rows.append({
        "table": "2_aggregated",
        "seed": "all",
        "total": table2["N_total"],
        "disagree_rate_mean": table2["disagree_rate_mean"],
        "disagree_rate_ci_lo": table2["disagree_rate_ci_lo"],
        "disagree_rate_ci_hi": table2["disagree_rate_ci_hi"],
        "zeroday_precision": table2["zeroday_precision"],
        "zeroday_recall": table2["zeroday_recall"],
        "zeroday_f1": table2["zeroday_f1"],
        "fisher_chi2": table2["fisher_chi2"],
        "fisher_p": table2["fisher_p"],
    })
    if isinstance(agg, dict) and agg.get("N_z0", 0) > 0:
        csv_rows.append({
            "table": "3_log_or_aggregated",
            "seed": "all",
            "N_z0": agg["N_z0"],
            "log_or": agg["log_or"],
            "ci_lo": agg["ci_lo"],
            "ci_hi": agg["ci_hi"],
            "independence_pass": agg["independence_pass"],
        })

    import csv
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in csv_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(analysis_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("CSV written: %s", analysis_csv)

    # Write JSON (full structured output)
    analysis_json = output_dir / f"{args.prefix}_analysis.json"
    result_obj = {
        "prefix": args.prefix,
        "seeds": seeds,
        "table1": table1_list,
        "table2": table2,
        "table3": table3,
        "table4": table4,
    }
    with open(analysis_json, "w", encoding="utf-8") as fout:
        json.dump(result_obj, fout, indent=2, ensure_ascii=False)
    log.info("JSON written: %s", analysis_json)

    # Registry append
    if not args.no_registry:
        if not args.registry.exists():
            log.warning("Registry file not found: %s — skipping append", args.registry)
        else:
            append_to_registry(
                registry_path=args.registry,
                table1_list=table1_list,
                table2=table2,
                table3=table3,
                table4=table4,
                analysis_csv=analysis_csv,
                analysis_json=analysis_json,
            )

    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
