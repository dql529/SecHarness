#!/usr/bin/env python3
"""C5 — sensitivity of cov(c) to the number of candidate-class samples |S_c|.

Determines how few samples the coverage statistic still needs: it is computed from samples
of the very attack class whose blind spot it forecasts.

Design
------
The paper's primary metric (cov_knn_attack, v3) already subsamples the held-out class to
EVAL_CAP=200 before the nearest-neighbour lookup. This script sweeps that budget over
|S_c| in {10,20,50,100,200,500,all} x 5 seeds and recorrelates against the *frozen*
zero-day detection rates (RF detector, and the deployed 3B agent) that are already on disk.
No detector is retrained and no LLM is run: only cov(c) is recomputed.

Correctness gate
----------------
--verify reproduces the published seed-42 / |S_c|=200 values in coverage_variants.csv
(column cov_knn_attack) bit-for-bit. The sweep refuses to run unless the gate passes,
so a fast v3-only path cannot silently drift from the original implementation.

Standardisation is cached per (dataset, held_class): it depends only on the _cap(SEED)
known-pool draw, never on the sweep's rng or |S_c|.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.candidate_b.analyze_coverage_variants import (  # noqa: E402
    KNN_REF_PER_GROUP,
    _feat,
)
from scripts.candidate_b.run_coverage_zd_stageA import (  # noqa: E402
    ATTACK_CAP,
    BENIGN_CAP,
    EVAL_CAP,
    OUT_DIR,
    SEED,
    _cap,
    load_cic,
    load_unsw,
)

SWEEP = [10, 20, 50, 100, 200, 500, None]  # None = every available sample
SEEDS = [42, 123, 7, 2024, 31337]


def standardised_fold(known_benign, known_attacks, held, num_cols, cat_cols):
    """Reproduce coverage_variants' feature + standardisation path verbatim (rng-free)."""
    if cat_cols:
        known_all = pd.concat([known_benign] + list(known_attacks.values()), ignore_index=True)
        oh_cols = list(pd.get_dummies(known_all[cat_cols].astype(str), columns=cat_cols).columns)
    else:
        oh_cols = []

    Xb = _feat(known_benign, num_cols, cat_cols, oh_cols)
    Xa = {n: _feat(a, num_cols, cat_cols, oh_cols) for n, a in known_attacks.items()}
    Xh = _feat(held, num_cols, cat_cols, oh_cols)

    pool = np.vstack([Xb] + list(Xa.values()))
    col_med = np.nanmedian(pool, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)

    def clean(X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_med, idx[1])
        return X

    pool_imp = clean(pool)
    mu = pool_imp.mean(axis=0)
    sd = pool_imp.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    def std(X):
        return (clean(X) - mu) / sd

    return std(Xb), np.vstack([std(x) for x in Xa.values()]), std(Xh)


def v3_knn_attack(Xb_s, atk_stack, Xh_s, rng, n_sc) -> Tuple[float, int]:
    """cov_knn_attack with |S_c| = n_sc. rng consumption order matches the original:
    benign reference draw, attack reference draw, then the held-class draw."""
    n_per = min(KNN_REF_PER_GROUP, len(Xb_s))
    ref_b = Xb_s[rng.choice(len(Xb_s), n_per, replace=False)]
    n_a = min(KNN_REF_PER_GROUP, len(atk_stack))
    ref_a = atk_stack[rng.choice(len(atk_stack), n_a, replace=False)]
    ref = np.vstack([ref_b, ref_a])
    ref_is_attack = np.array([0] * len(ref_b) + [1] * len(ref_a))

    cap = len(Xh_s) if n_sc is None else n_sc
    n_eval = min(cap, len(Xh_s))
    Xh_eval = Xh_s[rng.choice(len(Xh_s), n_eval, replace=False)] if len(Xh_s) > n_eval else Xh_s

    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    _, ind = nn.kneighbors(Xh_eval)
    return float(ref_is_attack[ind[:, 0]].mean()), len(Xh_eval)


def build_folds(name: str, loader) -> List[Dict]:
    """Cache the rng-independent standardised arrays for every eligible held-out class."""
    df, num_cols, cat_cols, benign, eligible = loader()
    benign_all = df[df["is_benign"]]
    folds = []
    for c in eligible:
        held_full = df[df["cls"] == c]
        if len(_cap(held_full, EVAL_CAP, SEED)) < 10:
            continue
        known_benign = _cap(benign_all, BENIGN_CAP, SEED)
        known_attacks = {a: _cap(df[df["cls"] == a], ATTACK_CAP, SEED)
                         for a in eligible if a != c}
        Xb_s, atk_stack, Xh_s = standardised_fold(
            known_benign, known_attacks, held_full, num_cols, cat_cols)
        folds.append({"dataset": name, "held_class": c, "n_available": len(Xh_s),
                      "Xb_s": Xb_s, "atk_stack": atk_stack, "Xh_s": Xh_s})
        print(f"  cached {name:12s} {c:18s} n_available={len(Xh_s)}", flush=True)
    return folds


def verify(folds: List[Dict]) -> bool:
    """Gate: reproduce the published seed-42, |S_c|=200 cov_knn_attack values."""
    published = pd.read_csv(OUT_DIR / "coverage_variants.csv")
    ref = {(r.dataset, r.held_class): r.cov_knn_attack for r in published.itertuples()}
    ok, checked = True, 0
    for ds in ("CIC-IDS2017", "UNSW-NB15"):
        rng = np.random.default_rng(SEED)  # dataset-level rng, shared across folds
        for f in [x for x in folds if x["dataset"] == ds]:
            got, _ = v3_knn_attack(f["Xb_s"], f["atk_stack"], f["Xh_s"], rng, EVAL_CAP)
            want = ref.get((ds, f["held_class"]))
            if want is None:
                continue
            checked += 1
            if abs(got - want) > 1e-12:
                print(f"  MISMATCH {ds}/{f['held_class']}: got {got} want {want}")
                ok = False
            else:
                print(f"  match    {ds}/{f['held_class']:18s} {got:.3f}")
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'} ({checked} folds compared bit-for-bit)")
    return ok and checked > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only run the correctness gate")
    ap.add_argument("--out", default=str(OUT_DIR / "cov_sample_sensitivity"))
    args = ap.parse_args()

    print("=== caching standardised folds ===", flush=True)
    folds = build_folds("CIC-IDS2017", load_cic) + build_folds("UNSW-NB15", load_unsw)

    print("\n=== correctness gate (seed 42, |S_c|=200 vs published) ===", flush=True)
    gate = verify(folds)
    if args.verify:
        sys.exit(0 if gate else 1)
    if not gate:
        print("BLOCK: v3 fast path does not reproduce published values; sweep aborted.")
        sys.exit(1)

    # Frozen outcome variables: nothing is retrained, nothing is re-inferred.
    stage_a = pd.read_csv(OUT_DIR / "coverage_zd_stageA.csv")
    zd_rf = {(r.dataset, r.held_class): r.zd_rate for r in stage_a.itertuples()}
    agent = pd.read_csv(OUT_DIR / "harness_zd_3b.csv")
    zd_col = "harness_zd_rate" if "harness_zd_rate" in agent.columns else "zd_rate"
    zd_agent = {(r.dataset, r.held_class): getattr(r, zd_col) for r in agent.itertuples()}

    print("\n=== sweeping |S_c| ===", flush=True)
    rows = []
    for n_sc in SWEEP:
        for seed in SEEDS:
            for ds in ("CIC-IDS2017", "UNSW-NB15"):
                rng = np.random.default_rng(seed)
                for f in [x for x in folds if x["dataset"] == ds]:
                    cov, n_used = v3_knn_attack(
                        f["Xb_s"], f["atk_stack"], f["Xh_s"], rng, n_sc)
                    key = (ds, f["held_class"])
                    rows.append({
                        "n_sc_requested": -1 if n_sc is None else n_sc,
                        "n_sc_used": n_used, "seed": seed, "dataset": ds,
                        "held_class": f["held_class"], "cov_knn_attack": cov,
                        "zd_rf": zd_rf.get(key), "zd_agent_3b": zd_agent.get(key),
                    })
        print(f"  |S_c|={'all' if n_sc is None else n_sc}: done", flush=True)

    per_fold = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(f"{out}_perfold.csv", index=False)

    # Correlation per (|S_c|, seed, dataset), then mean +/- std across seeds.
    corr_rows = []
    for (n_req, seed, ds), g in per_fold.groupby(["n_sc_requested", "seed", "dataset"]):
        for target in ("zd_rf", "zd_agent_3b"):
            gg = g.dropna(subset=[target])
            if len(gg) < 3 or gg["cov_knn_attack"].nunique() < 2:
                continue
            pr, pp = pearsonr(gg["cov_knn_attack"], gg[target])
            sr, sp = spearmanr(gg["cov_knn_attack"], gg[target])
            corr_rows.append({"n_sc_requested": n_req, "seed": seed, "dataset": ds,
                              "target": target, "n": len(gg),
                              "pearson_r": float(pr), "pearson_p": float(pp),
                              "spearman_r": float(sr), "spearman_p": float(sp)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(f"{out}_corr.csv", index=False)

    summary = (corr.groupby(["n_sc_requested", "dataset", "target"])
                   .agg(pearson_mean=("pearson_r", "mean"), pearson_std=("pearson_r", "std"),
                        spearman_mean=("spearman_r", "mean"), spearman_std=("spearman_r", "std"),
                        max_pearson_p=("pearson_p", "max"), n_seeds=("seed", "nunique"),
                        n_points=("n", "max"))
                   .reset_index())
    summary.to_csv(f"{out}_summary.csv", index=False)

    print("\n=== cov(c) vs zero-day rate, by |S_c| (mean +/- std over 5 seeds) ===")
    for target in ("zd_rf", "zd_agent_3b"):
        print(f"\n-- target = {target} --")
        print(f"{'|S_c|':>6} {'dataset':13s} {'Pearson':>16} {'Spearman':>16} {'max p':>9}")
        for _, r in summary[summary.target == target].iterrows():
            label = "all" if r.n_sc_requested == -1 else int(r.n_sc_requested)
            ps = "  nan" if pd.isna(r.pearson_std) else f"{r.pearson_std:.3f}"
            ss = "  nan" if pd.isna(r.spearman_std) else f"{r.spearman_std:.3f}"
            print(f"{str(label):>6} {r.dataset:13s} "
                  f"{r.pearson_mean:+.3f}+/-{ps:>6} {r.spearman_mean:+.3f}+/-{ss:>6} "
                  f"{r.max_pearson_p:9.2e}")

    with open(f"{out}_summary.json", "w") as fh:
        json.dump({"sweep": [(-1 if s is None else s) for s in SWEEP], "seeds": SEEDS,
                   "eval_cap_published": EVAL_CAP, "knn_ref_per_group": KNN_REF_PER_GROUP,
                   "verify_gate": "PASS", "summary": summary.to_dict(orient="records")},
                  fh, indent=2)
    print(f"\n=== DONE -> {out}_{{perfold,corr,summary}}.csv ===")


if __name__ == "__main__":
    main()
