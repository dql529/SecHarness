"""
preprocess.py — Data preprocessing for SecHarness experiments.

Handles both UNSW-NB15 and CIC-IDS2017 datasets:
  1. Load raw / pre-serialised data
  2. Structured-text serialisation (quantile-binned kv format)
  3. Leave-K-Attack-Out zero-day splits
  4. Export train / val / zero-day CSVs + summary stats

Reuses serialisation logic from uav_paper_3 (quantile binning → kv text).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # paper5/project/
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# UNSW-NB15 config
# ---------------------------------------------------------------------------
UNSW_LABEL_MAP = {
    "Normal": 0,
    "Backdoor": 1,
    "DoS": 2,
    "Exploits": 3,
    "Reconnaissance": 4,
    "Shellcode": 5,
    "Worms": 6,
}
UNSW_ID2LABEL = {v: k for k, v in UNSW_LABEL_MAP.items()}

UNSW_NUM_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "sload", "dload", "tcprtt", "synack", "ackdat",
    "ct_srv_src", "ct_state_ttl",
]
UNSW_CAT_COLS = ["proto", "state", "service"]

# ---------------------------------------------------------------------------
# CIC-IDS2017 config
# ---------------------------------------------------------------------------
# Map fine-grained source_label → coarse attack group for multi-class
CIC_ATTACK_GROUP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Bot",
    # Various encodings of the em-dash in CIC-IDS2017 CSVs
    "Web Attack \u2013 Brute Force": "WebAttack",
    "Web Attack \u2013 XSS": "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",
    "Web Attack \ufffd Brute Force": "WebAttack",
    "Web Attack \ufffd XSS": "WebAttack",
    "Web Attack \ufffd Sql Injection": "WebAttack",
    "Web Attack \x96 Brute Force": "WebAttack",
    "Web Attack \x96 XSS": "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}
CIC_LABEL_MAP = {
    "Benign": 0,
    "DoS": 1,
    "DDoS": 2,
    "PortScan": 3,
    "BruteForce": 4,
    "Bot": 5,
    "WebAttack": 6,
    "Infiltration": 7,
    "Heartbleed": 8,
}
CIC_ID2LABEL = {v: k for k, v in CIC_LABEL_MAP.items()}

# ---------------------------------------------------------------------------
# Leave-K-Attack-Out zero-day splits
# ---------------------------------------------------------------------------
ZERODAY_SPLITS = {
    "unsw_nb15": {
        # Hold out Shellcode + Worms as zero-day (less common, distinct patterns)
        "known": ["Normal", "Exploits", "Reconnaissance", "DoS", "Backdoor"],
        "zeroday": ["Shellcode", "Worms"],
    },
    "cic_ids2017": {
        # Hold out Bot + WebAttack as zero-day
        "known": ["Benign", "DoS", "DDoS", "PortScan", "BruteForce"],
        "zeroday": ["Bot", "WebAttack"],
    },
    "cic_ids2017_full": {
        # Hold out Bot + WebAttack + Infiltration as zero-day
        "known": ["Benign", "DoS", "DDoS", "PortScan", "BruteForce"],
        "zeroday": ["Bot", "WebAttack", "Infiltration"],
    },
}

BIN_LABELS = ["low", "medium", "high", "very_high"]


# ---------------------------------------------------------------------------
# Utility functions (adapted from uav_paper_3)
# ---------------------------------------------------------------------------
def norm_str(x, *, default: str = "UNK") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return default
    if s == "-":
        return "NONE"
    return s


def fit_bins(train_series: pd.Series, max_bins: int = 4) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Fit quantile bin edges on positive values from training data only."""
    s = pd.to_numeric(train_series, errors="coerce")
    pos = s[(s > 0) & (~s.isna())]
    if pos.empty:
        return None, None
    q = min(max_bins, len(BIN_LABELS))
    edges = np.quantile(pos.to_numpy(), np.linspace(0, 1, q + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.array([0.0, np.inf], dtype=np.float64), ["positive"]
    edges[0] = 0.0
    edges[-1] = np.inf
    return edges.astype(np.float64), BIN_LABELS[: len(edges) - 1]


def apply_bins(series: pd.Series, bins, labels) -> pd.Series:
    """Apply pre-fitted bins. NaN → missing, 0 → zero, >0 → bin label."""
    s = pd.to_numeric(series, errors="coerce")
    is_missing = s.isna() | (s < 0)
    is_zero = (s == 0) & (~is_missing)
    if bins is None or labels is None:
        out = pd.Series(["zero"] * len(s), index=series.index, dtype="string")
        out[is_missing] = "missing"
        return out
    s_pos = s.where(s > 0, np.nan)
    cat = pd.cut(s_pos, bins=bins, labels=labels, include_lowest=True)
    out = cat.astype("string").fillna("missing")
    out[is_zero] = "zero"
    out[is_missing] = "missing"
    return out


def build_kv_text(row: pd.Series, cat_cols: List[str], num_cols: List[str]) -> str:
    """Build key=value structured text representation."""
    parts = []
    for col in cat_cols:
        parts.append(f"{col}={row.get(col, 'UNK')}")
    for col in num_cols:
        parts.append(f"{col}={row.get(col + '_cat', 'missing')}")
    return " ; ".join(parts)


# ---------------------------------------------------------------------------
# UNSW-NB15 preprocessing
# ---------------------------------------------------------------------------
def preprocess_unsw(seed: int = 42, val_ratio: float = 0.2) -> Dict:
    """Load, clean, serialise, and split UNSW-NB15."""
    csv_path = DATA_RAW / "unsw15_filtered_nolog.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"UNSW data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["source_row_id"] = np.arange(len(df), dtype=np.int64)

    # Clean attack_cat and map to labels
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    df["label"] = df["attack_cat"].map(UNSW_LABEL_MAP)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(UNSW_ID2LABEL)

    # Normalise categorical columns
    for c in UNSW_CAT_COLS:
        if c in df.columns:
            df[c] = df[c].apply(norm_str).astype("string")

    # Ensure numeric columns
    for c in UNSW_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Stratified train/val split
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Fit bins on train only, apply to both
    bin_specs = {}
    for col in UNSW_NUM_COLS:
        if col not in train_df.columns:
            continue
        bins, labels = fit_bins(train_df[col])
        train_df[f"{col}_cat"] = apply_bins(train_df[col], bins, labels)
        val_df[f"{col}_cat"] = apply_bins(val_df[col], bins, labels)
        bin_specs[col] = {
            "bins": None if bins is None else [float(x) for x in bins.tolist()],
            "labels": labels,
        }

    # Build text
    train_df["text"] = train_df.apply(
        lambda r: build_kv_text(r, UNSW_CAT_COLS, UNSW_NUM_COLS), axis=1
    )
    val_df["text"] = val_df.apply(
        lambda r: build_kv_text(r, UNSW_CAT_COLS, UNSW_NUM_COLS), axis=1
    )

    # Output columns
    meta = ["source_row_id", "label", "label_name", "attack_cat", "text"]
    out_cols = meta + UNSW_CAT_COLS + UNSW_NUM_COLS + [f"{c}_cat" for c in UNSW_NUM_COLS]
    train_out = train_df[[c for c in out_cols if c in train_df.columns]]
    val_out = val_df[[c for c in out_cols if c in val_df.columns]]

    # Save
    out_dir = DATA_PROCESSED / "unsw_nb15"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(out_dir / "train.csv", index=False)
    val_out.to_csv(out_dir / "val.csv", index=False)

    # Zero-day split
    zd = ZERODAY_SPLITS["unsw_nb15"]
    known_mask_train = train_out["attack_cat"].isin(zd["known"])
    known_mask_val = val_out["attack_cat"].isin(zd["known"])
    zd_mask_val = val_out["attack_cat"].isin(zd["zeroday"])

    train_known = train_out[known_mask_train]
    val_known = val_out[known_mask_val]
    val_zeroday = val_out[zd_mask_val]

    zd_dir = out_dir / "zeroday_split"
    zd_dir.mkdir(parents=True, exist_ok=True)
    train_known.to_csv(zd_dir / "train_known.csv", index=False)
    val_known.to_csv(zd_dir / "val_known.csv", index=False)
    val_zeroday.to_csv(zd_dir / "val_zeroday.csv", index=False)

    # Save bin specs
    with open(out_dir / "bin_specs.json", "w") as f:
        json.dump({"seed": seed, "val_ratio": val_ratio, "bins": bin_specs,
                    "label_map": UNSW_LABEL_MAP}, f, indent=2)

    stats = {
        "dataset": "UNSW-NB15",
        "total_rows": len(df),
        "train_rows": len(train_out),
        "val_rows": len(val_out),
        "num_features": len(UNSW_NUM_COLS),
        "cat_features": len(UNSW_CAT_COLS),
        "num_classes": len(UNSW_LABEL_MAP),
        "class_dist": df["attack_cat"].value_counts().to_dict(),
        "zeroday_known_classes": zd["known"],
        "zeroday_held_out": zd["zeroday"],
        "zeroday_train_known": len(train_known),
        "zeroday_val_known": len(val_known),
        "zeroday_val_zeroday": len(val_zeroday),
    }
    print(f"[UNSW] train={len(train_out)}, val={len(val_out)}, "
          f"zd_train={len(train_known)}, zd_val_known={len(val_known)}, zd_val_new={len(val_zeroday)}")
    return stats


# ---------------------------------------------------------------------------
# CIC-IDS2017 preprocessing
# ---------------------------------------------------------------------------
def preprocess_cic(seed: int = 42) -> Dict:
    """Load pre-serialised CIC-IDS2017, add multi-class labels, create zero-day splits."""
    cic_dir = DATA_RAW / "cic_fair_text_csv"
    train_path = cic_dir / "cic_structured_text_train_medium_E1_clean.csv"
    val_path = cic_dir / "cic_structured_text_val_medium_E1_clean.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"CIC train data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Map source_label → attack group → multi-class label
    for df in [train_df, val_df]:
        df["attack_group"] = df["source_label"].map(CIC_ATTACK_GROUP)
        # Handle unmapped labels by trying fuzzy match
        unmapped = df["attack_group"].isna()
        if unmapped.any():
            for idx in df[unmapped].index:
                sl = df.loc[idx, "source_label"]
                for pattern, group in CIC_ATTACK_GROUP.items():
                    if pattern in sl or sl in pattern:
                        df.loc[idx, "attack_group"] = group
                        break
            # Drop any still unmapped
            still_unmapped = df["attack_group"].isna()
            if still_unmapped.any():
                print(f"[WARN] Dropping {still_unmapped.sum()} unmapped CIC rows: "
                      f"{df.loc[still_unmapped, 'source_label'].unique()}")
                df.drop(df[still_unmapped].index, inplace=True)

        df["mc_label"] = df["attack_group"].map(CIC_LABEL_MAP)
        df["mc_label_name"] = df["attack_group"]

    # Output columns: keep text + metadata
    meta_cols = ["source_row_id", "source_label", "label", "label_name",
                 "attack_group", "mc_label", "mc_label_name", "text"]
    out_cols = [c for c in meta_cols if c in train_df.columns]

    out_dir = DATA_PROCESSED / "cic_ids2017"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df[out_cols].to_csv(out_dir / "train.csv", index=False)
    val_df[out_cols].to_csv(out_dir / "val.csv", index=False)

    # Zero-day split
    zd = ZERODAY_SPLITS["cic_ids2017"]
    known_mask_train = train_df["attack_group"].isin(zd["known"])
    known_mask_val = val_df["attack_group"].isin(zd["known"])
    zd_mask_val = val_df["attack_group"].isin(zd["zeroday"])

    train_known = train_df[known_mask_train][out_cols]
    val_known = val_df[known_mask_val][out_cols]
    val_zeroday = val_df[zd_mask_val][out_cols]

    zd_dir = out_dir / "zeroday_split"
    zd_dir.mkdir(parents=True, exist_ok=True)
    train_known.to_csv(zd_dir / "train_known.csv", index=False)
    val_known.to_csv(zd_dir / "val_known.csv", index=False)
    val_zeroday.to_csv(zd_dir / "val_zeroday.csv", index=False)

    stats = {
        "dataset": "CIC-IDS2017",
        "total_rows": len(train_df) + len(val_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "source_label_dist": pd.concat([train_df, val_df])["source_label"].value_counts().to_dict(),
        "attack_group_dist": pd.concat([train_df, val_df])["attack_group"].value_counts().to_dict(),
        "num_classes_multiclass": len(CIC_LABEL_MAP),
        "zeroday_known_classes": zd["known"],
        "zeroday_held_out": zd["zeroday"],
        "zeroday_train_known": len(train_known),
        "zeroday_val_known": len(val_known),
        "zeroday_val_zeroday": len(val_zeroday),
    }
    print(f"[CIC] train={len(train_df)}, val={len(val_df)}, "
          f"zd_train={len(train_known)}, zd_val_known={len(val_known)}, zd_val_new={len(val_zeroday)}")
    return stats


# ---------------------------------------------------------------------------
# CIC-IDS2017 FULL preprocessing (from raw parquet)
# ---------------------------------------------------------------------------
def slugify_col(name: str) -> str:
    """Convert column name to lowercase slug: 'Flow Duration' -> 'flow_duration'."""
    import re
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "col"


def preprocess_cic_full(
    seed: int = 42,
    val_ratio: float = 0.2,
    train_cap: int = 80_000,
) -> Dict:
    """Load full CIC-IDS2017 parquet, serialise, zero-day split, subsample.

    - Reads all parquet files from data/raw/cicids2017_full/
    - Maps source labels → attack groups via CIC_ATTACK_GROUP
    - Applies quantile binning + kv text serialisation
    - Leave-K-Attack-Out split: known train/val + zeroday test (full)
    - Subsamples training to *train_cap* (stratified), keeps zeroday full
    """
    raw_dir = DATA_RAW / "cicids2017_full"
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {raw_dir}")

    # 1. Load & concatenate
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        df["_source_file"] = pf.name
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    raw["source_row_id"] = np.arange(len(raw), dtype=np.int64)

    # 2. Extract label, map to attack group
    label_col = "Label"
    raw["source_label"] = raw[label_col].astype(str).str.strip()
    # Build case-insensitive lookup (parquet uses "Benign", CIC_ATTACK_GROUP uses "BENIGN")
    _group_lookup = {k: v for k, v in CIC_ATTACK_GROUP.items()}
    # Add title-case variants for common labels
    for sl in raw["source_label"].unique():
        if sl not in _group_lookup:
            # Try exact match with known keys
            for k, v in CIC_ATTACK_GROUP.items():
                if k.lower() == sl.lower() or k in sl or sl in k:
                    _group_lookup[sl] = v
                    break
    raw["attack_group"] = raw["source_label"].map(_group_lookup)

    # Handle unmapped
    unmapped = raw["attack_group"].isna()
    if unmapped.any():
        print(f"[WARN] Dropping {unmapped.sum()} unmapped rows: "
              f"{raw.loc[unmapped, 'source_label'].value_counts().to_dict()}")
        raw = raw[~unmapped].copy()

    raw["mc_label"] = raw["attack_group"].map(CIC_LABEL_MAP)
    raw["mc_label_name"] = raw["attack_group"]

    # Binary label for backward compatibility
    raw["label"] = (raw["attack_group"] != "Benign").astype(int)
    raw["label_name"] = raw["label"].map({0: "benign", 1: "attack"})

    # 3. Prepare numeric feature columns (exclude label + metadata)
    meta_drop = {label_col, "_source_file", "source_row_id", "source_label",
                 "attack_group", "mc_label", "mc_label_name", "label", "label_name"}
    feature_cols_raw = [c for c in raw.columns if c not in meta_drop]
    # Slugify column names
    slug_map = {c: slugify_col(c) for c in feature_cols_raw}
    raw.rename(columns=slug_map, inplace=True)
    num_cols = [slug_map[c] for c in feature_cols_raw]

    # Coerce to numeric, replace inf
    for col in num_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # 4. Zero-day split
    zd = ZERODAY_SPLITS["cic_ids2017_full"]
    known_mask = raw["attack_group"].isin(zd["known"])
    zeroday_mask = raw["attack_group"].isin(zd["zeroday"])
    # Drop classes not in known or zeroday (e.g., Heartbleed=11 samples)
    excluded = ~known_mask & ~zeroday_mask
    if excluded.any():
        print(f"[INFO] Excluding {excluded.sum()} rows not in known/zeroday: "
              f"{raw.loc[excluded, 'attack_group'].value_counts().to_dict()}")

    known_df = raw[known_mask].copy()
    zeroday_df = raw[zeroday_mask].copy()

    # 5. Stratified train/val split on known classes
    train_known, val_known = train_test_split(
        known_df, test_size=val_ratio, random_state=seed,
        stratify=known_df["attack_group"],
    )
    train_known = train_known.reset_index(drop=True)
    val_known = val_known.reset_index(drop=True)

    # 6. Subsample training if needed (stratified by attack_group)
    if len(train_known) > train_cap:
        rng = np.random.default_rng(seed)
        indices = []
        total = len(train_known)
        for _grp, grp_df in train_known.groupby("attack_group"):
            n = max(1, int(train_cap * len(grp_df) / total))
            chosen = grp_df.sample(n=min(n, len(grp_df)), random_state=seed)
            indices.extend(chosen.index.tolist())
        train_known = train_known.loc[indices].reset_index(drop=True)
        print(f"[INFO] Subsampled training: {len(train_known)} rows (cap={train_cap})")

    # 7. Fit quantile bins on training data, apply to all splits
    bin_specs = {}
    for col in num_cols:
        bins, labels = fit_bins(train_known[col])
        bin_specs[col] = {"bins": bins, "labels": labels}
        for split_df in [train_known, val_known, zeroday_df]:
            split_df[f"{col}_cat"] = apply_bins(split_df[col], bins, labels)

    # 8. Build kv text
    for split_df in [train_known, val_known, zeroday_df]:
        split_df["text"] = split_df.apply(
            lambda r: " ; ".join(
                f"{col}={r.get(col + '_cat', 'missing')}" for col in num_cols
            ),
            axis=1,
        )

    # 9. Output columns
    meta_out = ["source_row_id", "source_label", "label", "label_name",
                "attack_group", "mc_label", "mc_label_name", "text"]
    out_cols = meta_out + num_cols + [f"{c}_cat" for c in num_cols]
    out_cols = [c for c in out_cols if c in train_known.columns]

    # 10. Save
    out_dir = DATA_PROCESSED / "cic_ids2017_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    zd_dir = out_dir / "zeroday_split"
    zd_dir.mkdir(parents=True, exist_ok=True)

    train_known[out_cols].to_csv(zd_dir / "train_known.csv", index=False)
    val_known[out_cols].to_csv(zd_dir / "val_known.csv", index=False)
    zeroday_df[out_cols].to_csv(zd_dir / "val_zeroday.csv", index=False)

    # Also save combined train/val for non-zeroday experiments
    train_known[out_cols].to_csv(out_dir / "train.csv", index=False)
    pd.concat([val_known, zeroday_df])[out_cols].to_csv(out_dir / "val.csv", index=False)

    # Save bin specs
    bin_specs_ser = {}
    for col, spec in bin_specs.items():
        bin_specs_ser[col] = {
            "bins": None if spec["bins"] is None else [float(x) for x in spec["bins"].tolist()],
            "labels": spec["labels"],
        }
    with open(out_dir / "bin_specs.json", "w") as f:
        json.dump({"seed": seed, "val_ratio": val_ratio, "train_cap": train_cap,
                    "bins": bin_specs_ser, "label_map": CIC_LABEL_MAP,
                    "zeroday_split": zd}, f, indent=2)

    stats = {
        "dataset": "CIC-IDS2017-Full",
        "total_rows_raw": len(raw) + excluded.sum(),
        "total_rows_used": len(known_df) + len(zeroday_df),
        "train_rows": len(train_known),
        "val_rows": len(val_known),
        "zeroday_rows": len(zeroday_df),
        "num_features": len(num_cols),
        "num_classes": len(CIC_LABEL_MAP),
        "attack_group_dist": raw["attack_group"].value_counts().to_dict(),
        "train_dist": train_known["attack_group"].value_counts().to_dict(),
        "val_dist": val_known["attack_group"].value_counts().to_dict(),
        "zeroday_dist": zeroday_df["attack_group"].value_counts().to_dict(),
        "zeroday_known_classes": zd["known"],
        "zeroday_held_out": zd["zeroday"],
    }

    print(f"[CIC-FULL] train_known={len(train_known)}, val_known={len(val_known)}, "
          f"zeroday={len(zeroday_df)}")
    print(f"  Zero-day dist: {stats['zeroday_dist']}")
    print(f"  Train dist: {stats['train_dist']}")
    return stats


# ---------------------------------------------------------------------------
# Data exploration summary
# ---------------------------------------------------------------------------
def generate_summary(unsw_stats: Dict, cic_stats: Dict) -> None:
    """Write a summary CSV for data exploration."""
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for stats in [unsw_stats, cic_stats]:
        rows.append({
            "dataset": stats["dataset"],
            "total_rows": stats["total_rows"],
            "train_rows": stats["train_rows"],
            "val_rows": stats["val_rows"],
            "num_classes": stats.get("num_classes", stats.get("num_classes_multiclass")),
            "zeroday_known": ", ".join(stats["zeroday_known_classes"]),
            "zeroday_held_out": ", ".join(stats["zeroday_held_out"]),
            "zd_train_known": stats["zeroday_train_known"],
            "zd_val_known": stats["zeroday_val_known"],
            "zd_val_zeroday": stats["zeroday_val_zeroday"],
        })
    summary = pd.DataFrame(rows)
    out_path = RESULTS_TABLES / "data_exploration_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[SUMMARY] saved → {out_path}")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SecHarness data preprocessing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--dataset", choices=["all", "unsw", "cic", "cic_full"], default="all")
    ap.add_argument("--train-cap", type=int, default=80_000,
                    help="Max training rows for cic_full (stratified subsample)")
    args = ap.parse_args()

    unsw_stats, cic_stats = None, None

    if args.dataset in ("all", "unsw"):
        unsw_stats = preprocess_unsw(seed=args.seed, val_ratio=args.val_ratio)

    if args.dataset in ("all", "cic"):
        cic_stats = preprocess_cic(seed=args.seed)

    if args.dataset == "cic_full":
        cic_stats = preprocess_cic_full(
            seed=args.seed, val_ratio=args.val_ratio, train_cap=args.train_cap,
        )

    if unsw_stats and cic_stats:
        generate_summary(unsw_stats, cic_stats)

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()
