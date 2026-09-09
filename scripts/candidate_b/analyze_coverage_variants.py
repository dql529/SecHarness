"""
analyze_coverage_variants.py — Candidate B, Stage A diagnostic.

Stage A's margin-Mahalanobis coverage showed a strong POOLED correlation with
ZD-rate but weak WITHIN-dataset correlation (Simpson's paradox). Root cause:
per-class Mahalanobis distances are not comparable across distributions of very
different volume (diffuse benign vs tight attack clusters), which biases the
margin.

This script reuses the SAME leave-one-attack-out folds and the SAME RF ZD-rates
(read from coverage_zd_stageA.csv — NOT recomputed) and recomputes 3 coverage
VARIANTS to find a volume-invariant metric whose within-dataset correlation with
ZD-rate holds up:

  v1 margin_maha   : original (per-class LedoitWolf precision)  [reproduced]
  v2 pooled_white  : margin under a SINGLE pooled known-pool covariance
                     (whitening once -> volume-invariant across classes)
  v3 knn_attack    : fraction of held-out samples whose nearest known-pool
                     neighbour (standardized space) is an attack rather than
                     benign  (deployment-computable, distribution-free)

All three are RF-INDEPENDENT (no RF output enters any coverage computation).
Output: results/tables/candidate_b/coverage_variants.csv + _summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact loaders / caps / constants from Stage A (no divergence)
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

KNN_REF_PER_GROUP = 4_000  # balanced reference: benign vs attack pool for kNN


def _feat(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
          oh_cols: List[str]) -> np.ndarray:
    num = df[num_cols].to_numpy(dtype=np.float64)
    if cat_cols:
        oh = pd.get_dummies(df[cat_cols].astype(str), columns=cat_cols)
        oh = oh.reindex(columns=oh_cols, fill_value=0).to_numpy(dtype=np.float64)
        return np.hstack([num, oh])
    return num


def _maha(mu: np.ndarray, X: np.ndarray) -> float:
    lw = LedoitWolf().fit(X)
    prec = lw.precision_ + 1e-6 * np.eye(X.shape[1])
    d = mu - lw.location_
    return float(np.sqrt(max(float(d @ prec @ d), 0.0)))


def coverage_variants(known_benign: pd.DataFrame, known_attacks: Dict[str, pd.DataFrame],
                      held: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
                      rng: np.random.Generator) -> Dict[str, float]:
    # Fixed one-hot vocab from known pool
    if cat_cols:
        known_all = pd.concat([known_benign] + list(known_attacks.values()), ignore_index=True)
        oh_cols = list(pd.get_dummies(known_all[cat_cols].astype(str), columns=cat_cols).columns)
    else:
        oh_cols = []

    Xb = _feat(known_benign, num_cols, cat_cols, oh_cols)
    Xa = {n: _feat(a, num_cols, cat_cols, oh_cols) for n, a in known_attacks.items()}
    Xh = _feat(held, num_cols, cat_cols, oh_cols)

    # Impute NaN with known-pool medians, standardize on known pool
    pool = np.vstack([Xb] + list(Xa.values()))
    col_med = np.nanmedian(pool, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)

    def clean(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_med, idx[1])
        return X

    pool_imp = clean(pool)
    mu = pool_imp.mean(axis=0)
    sd = pool_imp.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    std = lambda X: (clean(X) - mu) / sd

    Xb_s = std(Xb)
    Xa_s = {n: std(x) for n, x in Xa.items()}
    Xh_s = std(Xh)
    mu_c = Xh_s.mean(axis=0)

    # --- v1: per-class margin Mahalanobis (reproduce Stage A) ---
    d_benign = _maha(mu_c, Xb_s)
    d_atk = {n: _maha(mu_c, x) for n, x in Xa_s.items()}
    v1 = d_benign - min(d_atk.values())

    # --- v2: pooled-covariance whitened margin (single shared metric) ---
    lw_pool = LedoitWolf().fit(np.vstack([Xb_s] + list(Xa_s.values())))
    P = lw_pool.precision_ + 1e-6 * np.eye(Xb_s.shape[1])

    def md_pooled(mean_vec: np.ndarray) -> float:
        d = mu_c - mean_vec
        return float(np.sqrt(max(float(d @ P @ d), 0.0)))

    v2_benign = md_pooled(Xb_s.mean(axis=0))
    v2_atk_min = min(md_pooled(x.mean(axis=0)) for x in Xa_s.values())
    v2 = v2_benign - v2_atk_min

    # --- v3: kNN attack fraction (distribution-free) ---
    n_per = min(KNN_REF_PER_GROUP, len(Xb_s))
    ref_b = Xb_s[rng.choice(len(Xb_s), n_per, replace=False)]
    atk_stack = np.vstack(list(Xa_s.values()))
    n_a = min(KNN_REF_PER_GROUP, len(atk_stack))
    ref_a = atk_stack[rng.choice(len(atk_stack), n_a, replace=False)]
    ref = np.vstack([ref_b, ref_a])
    ref_is_attack = np.array([0] * len(ref_b) + [1] * len(ref_a))
    n_eval = min(EVAL_CAP, len(Xh_s))
    Xh_eval = Xh_s[rng.choice(len(Xh_s), n_eval, replace=False)] if len(Xh_s) > n_eval else Xh_s
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    _, ind = nn.kneighbors(Xh_eval)
    v3 = float(ref_is_attack[ind[:, 0]].mean())

    return {"cov_margin_maha": v1, "cov_pooled_white": v2, "cov_knn_attack": v3}


def run_dataset(name: str, loader, zd_lookup: Dict) -> List[Dict]:
    df, num_cols, cat_cols, benign, eligible = loader()
    benign_all = df[df["is_benign"]]
    rng = np.random.default_rng(SEED)
    rows = []
    for c in eligible:
        held_full = df[df["cls"] == c]
        if len(_cap(held_full, EVAL_CAP, SEED)) < 10:
            continue
        known_benign = _cap(benign_all, BENIGN_CAP, SEED)
        known_attacks = {a: _cap(df[df["cls"] == a], ATTACK_CAP, SEED)
                         for a in eligible if a != c}
        cov = coverage_variants(known_benign, known_attacks, held_full,
                                num_cols, cat_cols, rng)
        zd = zd_lookup.get((name, c))
        if zd is None:
            print(f"  [warn] no ZD-rate for ({name},{c}) in Stage A CSV")
            continue
        rows.append({"dataset": name, "held_class": c, "zd_rate": zd, **cov})
        print(f"  {c:18s} maha={cov['cov_margin_maha']:+8.3f} "
              f"pooled={cov['cov_pooled_white']:+7.3f} knn={cov['cov_knn_attack']:.3f} "
              f"zd={zd:.3f}")
    return rows


def _corr(g: pd.DataFrame, col: str) -> Dict:
    if len(g) < 3:
        return {}
    pr, pp = pearsonr(g[col], g["zd_rate"])
    sr, sp = spearmanr(g[col], g["zd_rate"])
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": len(g)}


def main() -> None:
    stage_a = pd.read_csv(OUT_DIR / "coverage_zd_stageA.csv")
    zd_lookup = {(r.dataset, r.held_class): r.zd_rate for r in stage_a.itertuples()}

    print("=== recomputing coverage variants (reusing Stage A folds + ZD) ===")
    rows = run_dataset("CIC-IDS2017", load_cic, zd_lookup) + \
        run_dataset("UNSW-NB15", load_unsw, zd_lookup)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "coverage_variants.csv", index=False)

    variants = ["cov_margin_maha", "cov_pooled_white", "cov_knn_attack"]
    summary: Dict = {}
    print("\n=== within-dataset + pooled correlation per variant ===")
    for v in variants:
        summary[v] = {}
        groups = {"pooled": out, "CIC-IDS2017": out[out.dataset == "CIC-IDS2017"],
                  "UNSW-NB15": out[out.dataset == "UNSW-NB15"]}
        line = f"{v:18s}"
        for gname, g in groups.items():
            block = _corr(g, v)
            if block:
                summary[v][gname] = block
                line += f"  {gname[:4]}: P={block['pearson_r']:+.2f}(p={block['pearson_p']:.2f})" \
                        f"/S={block['spearman_r']:+.2f}"
        print(line)

    with open(OUT_DIR / "coverage_variants_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== DONE -> {OUT_DIR/'coverage_variants.csv'} ===")


if __name__ == "__main__":
    main()
