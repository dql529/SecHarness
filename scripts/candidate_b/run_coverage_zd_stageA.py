"""
run_coverage_zd_stageA.py — Candidate B, Stage A.

Deployment-time COVERAGE judge vs zero-day (ZD) detectability, measured by a
leave-one-attack-out (LOAO) protocol over CIC-IDS2017 (11 fine classes) and
UNSW-NB15 (9 classes).

For each held-out attack class c:
  1. known pool = benign + (all eligible attack classes EXCEPT c)
  2. coverage(c) = RF-INDEPENDENT margin-Mahalanobis distance computed on raw,
     standardized numeric features:
         coverage(c) = D_Mahal(mu_c -> benign dist)
                     - min_a D_Mahal(mu_c -> known-attack_a dist)
     Higher = c sits deeper in known-attack territory (relative to benign) =>
     predicted to be MORE detectable as a zero-day.
  3. RF ZD detection rate = fraction of held-out c samples a RandomForest
     (trained on the known pool only, never seeing c) flags as "attack".

Output: ~20 (coverage, zd_rate) points -> results/tables/candidate_b/.
This stage is pure sklearn (no LLM); it produces the cheap scatter that gates
the expensive Stage B (full-harness LLM eval).

NOTE: coverage is computed WITHOUT any reference to RF outputs/confidence, so
"coverage predicts ZD" is not circular.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.beta_agent_ml import BetaAgentML  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    UNSW_CAT_COLS,
    UNSW_NUM_COLS,
    apply_bins,
    build_kv_text,
    fit_bins,
)

DATASETS_DIR = Path.home() / "Datasets" / "network_ids"

# Seed is env-driven for multi-seed robustness runs (CB_SEED). The primary
# (pre-registered) results use seed 42 and live in candidate_b/; other seeds are
# isolated in candidate_b/seed_<N>/ so they never disturb the primary artifacts.
SEED = int(os.environ.get("CB_SEED", "42"))
OUT_DIR = PROJECT_ROOT / "results" / "tables" / "candidate_b"
if SEED != 42:
    OUT_DIR = OUT_DIR / f"seed_{SEED}"
BENIGN_CAP = 10_000      # cap benign rows for covariance / RF training
ATTACK_CAP = 5_000       # cap per known-attack rows
EVAL_CAP = 200           # held-out samples used to measure ZD rate (plan N=200)
COV_REG = 1e-6           # tiny ridge added to LedoitWolf precision for safety


# ---------------------------------------------------------------------------
# CIC label normalisation (raw "Label" -> fine class; excludes rare classes)
# ---------------------------------------------------------------------------
CIC_BENIGN = "Benign"
# Eligible 11 fine classes (plan go/no-go). Rare classes excluded entirely:
#   Infiltration (36), Web Attack-Sql Injection (21), Heartbleed (11).
CIC_ELIGIBLE = [
    "DoS Hulk", "DDoS", "DoS GoldenEye", "FTP-Patator", "DoS slowloris",
    "DoS Slowhttptest", "SSH-Patator", "PortScan", "WebAttack-BF", "Bot",
    "WebAttack-XSS",
]


def normalize_cic_label(raw: str) -> Optional[str]:
    """Map a raw CIC 'Label' value to a canonical fine class, or None to drop."""
    s = str(raw).strip()
    if s.lower() == "benign":
        return CIC_BENIGN
    low = s.lower()
    if low.startswith("web attack"):
        if "brute force" in low:
            return "WebAttack-BF"
        if "xss" in low:
            return "WebAttack-XSS"
        if "sql" in low:
            return None  # excluded (21 samples)
        return None
    # Direct passthrough for classes whose raw spelling matches eligible list
    direct = {
        "dos hulk": "DoS Hulk", "ddos": "DDoS", "dos goldeneye": "DoS GoldenEye",
        "ftp-patator": "FTP-Patator", "dos slowloris": "DoS slowloris",
        "dos slowhttptest": "DoS Slowhttptest", "ssh-patator": "SSH-Patator",
        "portscan": "PortScan", "bot": "Bot",
    }
    if low in direct:
        return direct[low]
    return None  # Infiltration, Heartbleed, anything unexpected -> drop


def slugify_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "col"


# ---------------------------------------------------------------------------
# Dataset loading -> unified frame with: numeric cols, cat cols, 'cls', 'is_benign'
# ---------------------------------------------------------------------------
def load_unsw() -> Tuple[pd.DataFrame, List[str], List[str], str, List[str]]:
    """Return (df, num_cols, cat_cols, benign_label, eligible_attacks)."""
    files = [
        DATASETS_DIR / "UNSW-NB15" / "UNSW_NB15_training-set.parquet",
        DATASETS_DIR / "UNSW-NB15" / "UNSW_NB15_testing-set.parquet",
    ]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"UNSW parquet missing: {f}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["cls"] = df["attack_cat"].astype(str).str.strip()
    num_cols = [c for c in UNSW_NUM_COLS if c in df.columns]
    cat_cols = [c for c in UNSW_CAT_COLS if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().replace({"-": "NONE", "nan": "UNK", "": "UNK"})
    benign = "Normal"
    eligible = sorted([c for c in df["cls"].unique() if c != benign])  # 9 attack classes
    df["is_benign"] = df["cls"] == benign
    keep = num_cols + cat_cols + ["cls", "is_benign"]
    return df[keep].copy(), num_cols, cat_cols, benign, eligible


def load_cic() -> Tuple[pd.DataFrame, List[str], List[str], str, List[str]]:
    files = sorted((DATASETS_DIR / "CICIDS2017").glob("*-no-metadata.parquet"))
    if not files:
        raise FileNotFoundError(f"No CIC parquet in {DATASETS_DIR/'CICIDS2017'}")
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    label_col = "Label" if "Label" in df.columns else [c for c in df.columns if c.strip() == "Label"][0]
    df["cls"] = df[label_col].map(normalize_cic_label)
    df = df[df["cls"].notna()].copy()
    # numeric feature cols = everything except the label
    feat_raw = [c for c in df.columns if c != label_col and c != "cls"]
    slug = {c: slugify_col(c) for c in feat_raw}
    df = df.rename(columns=slug)
    num_cols = [slug[c] for c in feat_raw]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    benign = CIC_BENIGN
    eligible = [c for c in CIC_ELIGIBLE if c in set(df["cls"].unique())]
    df["is_benign"] = df["cls"] == benign
    keep = num_cols + ["cls", "is_benign"]
    return df[keep].copy(), num_cols, [], benign, eligible


# ---------------------------------------------------------------------------
# Coverage: margin-Mahalanobis on standardized numeric features
# ---------------------------------------------------------------------------
def _cap(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed) if len(df) > n else df


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) with zero-variance guard."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _apply_std(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _mahalanobis(mu_target: np.ndarray, X_dist: np.ndarray) -> float:
    """Mahalanobis distance from point mu_target to distribution X_dist."""
    lw = LedoitWolf().fit(X_dist)
    prec = lw.precision_ + COV_REG * np.eye(X_dist.shape[1])
    diff = mu_target - lw.location_
    d2 = float(diff @ prec @ diff)
    return float(np.sqrt(max(d2, 0.0)))


def compute_coverage(
    known_benign: pd.DataFrame,
    known_attacks: Dict[str, pd.DataFrame],
    held: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, float]:
    """Margin-Mahalanobis coverage for the held-out class.

    Features = standardized numeric + one-hot categoricals, fit on the union of
    the known pool (deployment-realistic: only known data is available).
    """
    def feat(df: pd.DataFrame, oh_cols: List[str]) -> np.ndarray:
        num = df[num_cols].to_numpy(dtype=np.float64)
        if cat_cols:
            oh = pd.get_dummies(df[cat_cols].astype(str), columns=cat_cols)
            oh = oh.reindex(columns=oh_cols, fill_value=0).to_numpy(dtype=np.float64)
            return np.hstack([num, oh])
        return num

    # One-hot vocabulary fixed from the known pool
    if cat_cols:
        known_all = pd.concat([known_benign] + list(known_attacks.values()), ignore_index=True)
        oh_cols = list(pd.get_dummies(known_all[cat_cols].astype(str), columns=cat_cols).columns)
    else:
        oh_cols = []

    Xb = feat(known_benign, oh_cols)
    # Median-impute NaNs using known-pool medians, then standardize on known pool
    pool = np.vstack([Xb] + [feat(a, oh_cols) for a in known_attacks.values()])
    col_med = np.nanmedian(pool, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)

    def clean_std(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        idx = np.where(np.isnan(X))
        X = X.copy()
        X[idx] = np.take(col_med, idx[1])
        return _apply_std(X, mu, sd)

    pool_imp = pool.copy()
    nan_idx = np.where(np.isnan(pool_imp))
    pool_imp[nan_idx] = np.take(col_med, nan_idx[1])
    mu, sd = _standardize_fit(pool_imp)

    Xb_s = clean_std(Xb, mu, sd)
    Xh_s = clean_std(feat(held, oh_cols), mu, sd)
    mu_c = Xh_s.mean(axis=0)

    d_benign = _mahalanobis(mu_c, Xb_s)
    d_attacks = {}
    for name, a in known_attacks.items():
        Xa_s = clean_std(feat(a, oh_cols), mu, sd)
        d_attacks[name] = _mahalanobis(mu_c, Xa_s)
    d_attack_min = min(d_attacks.values())
    nearest = min(d_attacks, key=d_attacks.get)

    return {
        "coverage": d_benign - d_attack_min,
        "d_benign": d_benign,
        "d_attack_min": d_attack_min,
        "nearest_known_attack": nearest,
    }


# ---------------------------------------------------------------------------
# RF ZD detection rate via existing BetaAgentML (kv-text serialization)
# ---------------------------------------------------------------------------
def _serialize(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
               bin_specs: Dict) -> pd.Series:
    """Build kv text for rows using pre-fitted bins (reuses preprocess utils)."""
    work = df.copy()
    for col in num_cols:
        bins, labels = bin_specs[col]
        work[f"{col}_cat"] = apply_bins(work[col], bins, labels)
    return work.apply(lambda r: build_kv_text(r, cat_cols, num_cols), axis=1)


def rf_zd_rate(
    known_benign: pd.DataFrame,
    known_attacks: Dict[str, pd.DataFrame],
    held: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, float]:
    """Train RF on known pool (fine multi-class labels), measure ZD detect rate."""
    train_df = pd.concat([known_benign] + list(known_attacks.values()), ignore_index=True)
    # Fit bins on known training numeric only
    bin_specs = {col: fit_bins(train_df[col]) for col in num_cols}
    train_df = train_df.copy()
    train_df["text"] = _serialize(train_df, num_cols, cat_cols, bin_specs)
    train_df["label_name"] = train_df["cls"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as tf:
        train_df[["text", "label_name"]].to_csv(tf.name, index=False)
        agent = BetaAgentML(classifier_type="random_forest")
        stats = agent.train(tf.name, text_col="text", label_col="label_name")

    held = held.copy()
    held["text"] = _serialize(held, num_cols, cat_cols, bin_specs)
    verdicts = agent.analyze_batch(held["text"].tolist())
    detected = sum(1 for v in verdicts if v.verdict == "attack")
    return {
        "zd_total": len(verdicts),
        "zd_detected": detected,
        "zd_rate": detected / len(verdicts) if verdicts else 0.0,
        "rf_train_acc": stats.get("train_accuracy", float("nan")),
        "rf_n_classes": len(stats.get("classes", [])),
    }


# ---------------------------------------------------------------------------
# Per-dataset LOAO driver
# ---------------------------------------------------------------------------
def run_dataset(name: str, loader) -> List[Dict]:
    print(f"\n=== {name}: loading raw parquet ===")
    df, num_cols, cat_cols, benign, eligible = loader()
    print(f"  rows={len(df)} num_feat={len(num_cols)} cat_feat={len(cat_cols)} "
          f"benign='{benign}' eligible_attacks={len(eligible)}: {eligible}")

    benign_all = df[df["is_benign"]]
    rows: List[Dict] = []
    for c in eligible:
        known_benign = _cap(benign_all, BENIGN_CAP, SEED)
        known_attacks = {
            a: _cap(df[df["cls"] == a], ATTACK_CAP, SEED)
            for a in eligible if a != c
        }
        held_full = df[df["cls"] == c]
        held_eval = _cap(held_full, EVAL_CAP, SEED)
        if len(held_eval) < 10:
            print(f"  [skip] {c}: only {len(held_eval)} samples")
            continue

        cov = compute_coverage(known_benign, known_attacks, held_full, num_cols, cat_cols)
        zd = rf_zd_rate(known_benign, known_attacks, held_eval, num_cols, cat_cols)
        row = {"dataset": name, "held_class": c, "n_known_classes": len(known_attacks) + 1,
               **cov, **zd}
        rows.append(row)
        print(f"  {c:18s} cov={cov['coverage']:+.3f} "
              f"(d_benign={cov['d_benign']:.2f} d_atk_min={cov['d_attack_min']:.2f} "
              f"near={cov['nearest_known_attack']}) zd_rate={zd['zd_rate']:.3f} "
              f"(n={zd['zd_total']})")
    return rows


def _corr_block(cov: np.ndarray, zd: np.ndarray, n_boot: int = 10_000) -> Dict:
    """Pearson + Spearman with bootstrap 95% CIs. Returns {} if n < 3."""
    n = len(cov)
    if n < 3:
        return {}
    pr, pp = pearsonr(cov, zd)
    sr, sp = spearmanr(cov, zd)
    rng = np.random.default_rng(SEED)
    boot_p, boot_s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(cov[idx]) < 1e-12 or np.std(zd[idx]) < 1e-12:
            continue
        boot_p.append(pearsonr(cov[idx], zd[idx])[0])
        boot_s.append(spearmanr(cov[idx], zd[idx])[0])
    def ci(b: List[float]) -> List[float]:
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b else [float("nan")] * 2
    return {
        "n": n,
        "pearson_r": float(pr), "pearson_p": float(pp), "pearson_ci95": ci(boot_p),
        "spearman_r": float(sr), "spearman_p": float(sp), "spearman_ci95": ci(boot_s),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = run_dataset("CIC-IDS2017", load_cic) + run_dataset("UNSW-NB15", load_unsw)
    out = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "coverage_zd_stageA.csv"
    out.to_csv(csv_path, index=False)

    summary: Dict = {"n_points": len(out), "seed": SEED, "benign_cap": BENIGN_CAP,
                     "attack_cap": ATTACK_CAP, "eval_cap": EVAL_CAP, "correlations": {}}
    # Pooled + per-dataset (Simpson's-paradox defense)
    groups = {"pooled": out}
    for ds in out["dataset"].unique():
        groups[ds] = out[out["dataset"] == ds]
    for key, g in groups.items():
        block = _corr_block(g["coverage"].to_numpy(), g["zd_rate"].to_numpy())
        if block:
            summary["correlations"][key] = block

    with open(OUT_DIR / "coverage_zd_stageA_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== DONE: {len(out)} points -> {csv_path} ===")
    for key, b in summary["correlations"].items():
        print(f"  [{key:12s} n={b['n']:2d}] "
              f"Pearson r={b['pearson_r']:+.3f} CI{b['pearson_ci95']} p={b['pearson_p']:.3g} | "
              f"Spearman r={b['spearman_r']:+.3f} p={b['spearman_p']:.3g}")


if __name__ == "__main__":
    main()
