"""
run_multidetector_validation.py — Candidate B, cross-detector tautology kill.

Stage A + variants established that the PRE-REGISTERED primary coverage metric
(kNN-attack-fraction, cov_knn_attack) strongly predicts the binned-RF zero-day
detection rate within AND across datasets. The open critique: "kNN-coverage is
itself a 1-NN attack/benign classifier, so it trivially correlates with another
nearest-neighbour-like detector."

This script refutes that by checking whether the SAME pre-registered kNN-coverage
predicts the zero-day detection rate of THREE structurally distinct detector
families trained per leave-one-attack-out fold (none of which is the kNN itself):

  - LogisticRegression    (linear decision boundary)
  - HistGradientBoosting  (boosted additive trees)
  - RandomForest          (bagged axis-aligned trees)

All three are trained on raw, standardized numeric features (+ one-hot cats for
UNSW) of the known pool (benign + all attacks except the held-out class), with
fine multi-class labels. ZD-rate = fraction of held-out class samples predicted
to be an attack (predicted label not in {benign, normal}).

kNN-coverage is READ from coverage_variants.csv (frozen, not recomputed) so the
metric is fixed exactly as pre-registered before this confirmatory run.

Success (pre-registered): within-dataset Pearson p<0.05 for >=2 of 3 families.

Output: results/tables/candidate_b/multidetector_validation.csv + _summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.candidate_b.analyze_coverage_variants import _feat  # noqa: E402
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

BENIGN_NAMES = {"benign", "normal"}


def _detectors() -> Dict:
    return {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        "histgb": HistGradientBoostingClassifier(random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=12,
                                     random_state=SEED, n_jobs=-1),
    }


def _prep_features(known_benign, known_attacks, held, num_cols, cat_cols):
    """Standardized feature matrices (known-pool fit) + multi-class labels."""
    if cat_cols:
        known_all = pd.concat([known_benign] + list(known_attacks.values()),
                              ignore_index=True)
        oh_cols = list(pd.get_dummies(known_all[cat_cols].astype(str),
                                      columns=cat_cols).columns)
    else:
        oh_cols = []

    parts_X, parts_y = [], []
    Xb = _feat(known_benign, num_cols, cat_cols, oh_cols)
    parts_X.append(Xb)
    parts_y.append(known_benign["cls"].to_numpy())
    for name, a in known_attacks.items():
        parts_X.append(_feat(a, num_cols, cat_cols, oh_cols))
        parts_y.append(a["cls"].to_numpy())
    X_known = np.vstack(parts_X)
    y_known = np.concatenate(parts_y)
    X_held = _feat(held, num_cols, cat_cols, oh_cols)

    # Impute NaN with known-pool medians, standardize on known pool
    col_med = np.nanmedian(X_known, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)

    def clean(X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_med, idx[1])
        return X

    X_known = clean(X_known)
    X_held = clean(X_held)
    mu = X_known.mean(axis=0)
    sd = X_known.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_known - mu) / sd, y_known, (X_held - mu) / sd


def _zd_rate(detector, X_known, y_known, X_held) -> float:
    detector.fit(X_known, y_known)
    preds = detector.predict(X_held)
    is_attack = np.array([str(p).lower() not in BENIGN_NAMES for p in preds])
    return float(is_attack.mean())


def run_dataset(name: str, loader) -> List[Dict]:
    df, num_cols, cat_cols, benign, eligible = loader()
    benign_all = df[df["is_benign"]]
    rows = []
    for c in eligible:
        held_eval = _cap(df[df["cls"] == c], EVAL_CAP, SEED)
        if len(held_eval) < 10:
            continue
        known_benign = _cap(benign_all, BENIGN_CAP, SEED)
        known_attacks = {a: _cap(df[df["cls"] == a], ATTACK_CAP, SEED)
                         for a in eligible if a != c}
        X_known, y_known, X_held = _prep_features(
            known_benign, known_attacks, held_eval, num_cols, cat_cols)
        row = {"dataset": name, "held_class": c, "n_eval": len(held_eval)}
        for dname, det in _detectors().items():
            row[f"zd_{dname}"] = _zd_rate(det, X_known, y_known, X_held)
        rows.append(row)
        print(f"  {c:18s} logreg={row['zd_logreg']:.3f} "
              f"histgb={row['zd_histgb']:.3f} rf={row['zd_rf']:.3f} (n={row['n_eval']})")
    return rows


def _corr(g: pd.DataFrame, cov_col: str, zd_col: str) -> Dict:
    if len(g) < 3:
        return {}
    pr, pp = pearsonr(g[cov_col], g[zd_col])
    sr, sp = spearmanr(g[cov_col], g[zd_col])
    return {"n": len(g), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def main() -> None:
    # Frozen pre-registered coverage (kNN-attack-fraction) from variants run
    variants = pd.read_csv(OUT_DIR / "coverage_variants.csv")
    cov_lookup = {(r.dataset, r.held_class): r.cov_knn_attack
                  for r in variants.itertuples()}

    print("=== training LogReg / HistGB / RF per leave-one-attack-out fold ===")
    rows = run_dataset("CIC-IDS2017", load_cic) + run_dataset("UNSW-NB15", load_unsw)
    out = pd.DataFrame(rows)
    out["cov_knn_attack"] = [cov_lookup.get((r.dataset, r.held_class), np.nan)
                             for r in out.itertuples()]
    out = out.dropna(subset=["cov_knn_attack"])
    out.to_csv(OUT_DIR / "multidetector_validation.csv", index=False)

    detectors = ["zd_logreg", "zd_histgb", "zd_rf"]
    summary: Dict = {"primary_metric": "cov_knn_attack", "detectors": {}}
    print("\n=== kNN-coverage vs each detector ZD-rate (within + pooled) ===")
    n_pass = {"pooled": 0, "CIC-IDS2017": 0, "UNSW-NB15": 0}
    for d in detectors:
        summary["detectors"][d] = {}
        groups = {"pooled": out,
                  "CIC-IDS2017": out[out.dataset == "CIC-IDS2017"],
                  "UNSW-NB15": out[out.dataset == "UNSW-NB15"]}
        line = f"{d:10s}"
        for gname, g in groups.items():
            b = _corr(g, "cov_knn_attack", d)
            if b:
                summary["detectors"][d][gname] = b
                if b["pearson_p"] < 0.05 and b["pearson_r"] > 0:
                    n_pass[gname] += 1
                line += f"  {gname[:4]}:P={b['pearson_r']:+.2f}(p={b['pearson_p']:.3f})"
        print(line)

    # Pre-registered success: within-dataset p<0.05 for >=2 of 3 families
    summary["preregistered_success"] = {
        "criterion": "within-dataset Pearson p<0.05 & r>0 for >=2 of 3 detector families",
        "CIC_families_passing": n_pass["CIC-IDS2017"],
        "UNSW_families_passing": n_pass["UNSW-NB15"],
        "verdict": "PASS" if (n_pass["CIC-IDS2017"] >= 2 and n_pass["UNSW-NB15"] >= 2)
        else "PARTIAL" if (n_pass["CIC-IDS2017"] >= 2 or n_pass["UNSW-NB15"] >= 2)
        else "FAIL",
    }
    with open(OUT_DIR / "multidetector_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== VERDICT: {summary['preregistered_success']['verdict']} "
          f"(CIC {n_pass['CIC-IDS2017']}/3, UNSW {n_pass['UNSW-NB15']}/3 families pass) ===")
    print(f"-> {OUT_DIR/'multidetector_validation.csv'}")


if __name__ == "__main__":
    main()
