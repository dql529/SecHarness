"""
consolidate_results.py — Candidate B consolidation (pure pandas, no LLM).

Produces the final numbers for the paper:
  1. Multi-seed robustness: harness-ZD vs coverage Pearson across seeds 42/123/7
     -> mean +/- std (kills the single-seed criticism).
  2. Capacity scaling: coverage-vs-harness-ZD correlation at 3B/7B/14B
     -> shows predictive power declines as the agent shifts relay -> exploration.
  3. Selective reframed as coverage-driven HUMAN escalation triage (from the
     seed-42 selective per-sample data): escalating the lowest-local-coverage
     traffic recovers the zero-days the relay misses, far more efficiently than
     random escalation.

Output: results/tables/candidate_b/CONSOLIDATED_summary.json (+ console).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TAB = Path(__file__).resolve().parents[2] / "results" / "tables" / "candidate_b"


def multiseed() -> dict:
    rows = {}
    for seed, sub in [(42, ""), (123, "seed_123"), (7, "seed_7")]:
        p = TAB / sub / "harness_zd_3b_summary.json"
        d = json.load(open(p))
        rows[seed] = {g: d[g]["pearson_r"] for g in ("pooled", "CIC-IDS2017", "UNSW-NB15")}
    out = {"per_seed": rows}
    for g in ("pooled", "CIC-IDS2017", "UNSW-NB15"):
        vals = [rows[s][g] for s in rows]
        out[f"{g}_mean"] = float(np.mean(vals))
        out[f"{g}_std"] = float(np.std(vals, ddof=1))
    return out


def scaling() -> dict:
    out = {}
    for tag, name in [("3b", "Llama-3.2-3B"), ("qwen7b", "Qwen2.5-7B"),
                      ("qwen14b", "Qwen2.5-14B")]:
        d = json.load(open(TAB / f"harness_zd_{tag}_summary.json"))
        zd = pd.read_csv(TAB / f"harness_zd_{tag}.csv")["harness_zd_rate"]
        out[name] = {"pooled_r": d["pooled"]["pearson_r"],
                     "CIC_r": d["CIC-IDS2017"]["pearson_r"],
                     "UNSW_r": d["UNSW-NB15"]["pearson_r"],
                     "zd_mean": float(zd.mean()), "zd_std": float(zd.std())}
    return out


def escalation_triage() -> dict:
    """Coverage-driven human-escalation: escalate lowest-local_cov traffic to a
    (perfect) human; recover missed zero-days. Compare coverage-sort vs random."""
    fp = TAB / "selective_3b_samples.csv"
    if not fp.exists():
        return {"skipped": "selective_3b_samples.csv not pulled yet"}
    df = pd.read_csv(fp)
    atk = df[df.sample_type == "attack"].copy()
    n_atk = len(atk)
    # relay baseline recall
    base_recall = float((atk.relay_verdict == "attack").mean())
    # an attack is "recovered" by escalation if relay missed it (relay==benign)
    # and it gets escalated (human catches it).
    all_df = df.copy()
    all_df["esc_order"] = all_df["local_cov"]  # escalate ascending cov first
    n_all = len(all_df)

    def recall_at_budget(frac: float, by: str) -> float:
        k = int(round(frac * n_all))
        if by == "cov":
            esc = set(all_df.nsmallest(k, "esc_order").index)
        else:  # random (seed fixed)
            esc = set(all_df.sample(n=k, random_state=42).index)
        a = atk
        caught = ((a.relay_verdict == "attack") | (a.index.isin(esc))).sum()
        return float(caught / n_atk)

    budgets = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    curve = []
    for f in budgets:
        curve.append({"escalation_budget": f,
                      "recall_coverage_sorted": recall_at_budget(f, "cov"),
                      "recall_random": recall_at_budget(f, "rand")})
    return {"relay_base_recall": base_recall, "n_attack": n_atk, "n_total": n_all,
            "triage_curve": curve}


def main() -> None:
    res = {"multiseed_3b_harnessZD_vs_coverage": multiseed(),
           "capacity_scaling": scaling(),
           "selective_escalation_triage": escalation_triage()}
    json.dump(res, open(TAB / "CONSOLIDATED_summary.json", "w"), indent=2)

    ms = res["multiseed_3b_harnessZD_vs_coverage"]
    print("=== MULTI-SEED (3B harness-ZD vs coverage Pearson r) ===")
    for s in (42, 123, 7):
        r = ms["per_seed"][s]
        print(f"  seed {s:>3}: pooled={r['pooled']:.3f} CIC={r['CIC-IDS2017']:.3f} UNSW={r['UNSW-NB15']:.3f}")
    print(f"  MEAN+/-STD: pooled={ms['pooled_mean']:.3f}+/-{ms['pooled_std']:.3f}  "
          f"CIC={ms['CIC-IDS2017_mean']:.3f}+/-{ms['CIC-IDS2017_std']:.3f}  "
          f"UNSW={ms['UNSW-NB15_mean']:.3f}+/-{ms['UNSW-NB15_std']:.3f}")

    print("\n=== CAPACITY SCALING (coverage predictive power vs model size) ===")
    for name, d in res["capacity_scaling"].items():
        print(f"  {name:14}: pooled r={d['pooled_r']:+.3f}  (ZD mean={d['zd_mean']:.3f} std={d['zd_std']:.3f})")

    print("\n=== SELECTIVE -> coverage-driven escalation triage ===")
    t = res["selective_escalation_triage"]
    if "skipped" in t:
        print(f"  [skipped] {t['skipped']}")
        print(f"\n-> {TAB/'CONSOLIDATED_summary.json'}")
        return
    print(f"  relay base recall={t['relay_base_recall']:.3f}  (n_attack={t['n_attack']})")
    for p in t["triage_curve"]:
        print(f"  escalate {p['escalation_budget']*100:4.0f}%: "
              f"cov-sorted recall={p['recall_coverage_sorted']:.3f}  "
              f"random recall={p['recall_random']:.3f}")
    print(f"\n-> {TAB/'CONSOLIDATED_summary.json'}")


if __name__ == "__main__":
    main()
