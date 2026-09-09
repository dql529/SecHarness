#!/usr/bin/env python3
"""
run_rf_baseline_ablation.py — RF + deterministic logging wrapper (no LLM).

Evaluates the RF used directly with a deterministic logging wrapper, in place of
the LLM agent.

Runs the same RF classifier used by check_anomaly on the same 1000
val samples, with a deterministic logging wrapper that records:
- predicted class, confidence, anomaly score
- a JSON "audit trail" mimicking SecHarness format

Compares output against E3/E4 to show whether the LLM adds anything.
"""

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))

# Paths
VAL_KNOWN = ROOT / "data/processed/unsw_nb15/zeroday_split/val_known.csv"
VAL_ZD = ROOT / "data/processed/unsw_nb15/zeroday_split/val_zeroday.csv"
ML_MODEL = ROOT / "logs/E2_beta_ml_model.pkl"
OUT_CSV = ROOT / "results/tables/v2/rf_ablation_results.csv"
OUT_AUDIT = ROOT / "logs/v2/rf_ablation_sub1000_audit.jsonl"

EVAL_N = 1000
SEED = 42


def load_eval_data() -> pd.DataFrame:
    """Load the same 1000-sample eval set used by E3/E4."""
    val_known = pd.read_csv(str(VAL_KNOWN))
    val_zd = pd.read_csv(str(VAL_ZD))

    zd_n = min(int(EVAL_N * 0.1), len(val_zd))
    known_n = EVAL_N - zd_n

    val_known = val_known.sample(n=known_n, random_state=SEED)
    val_zd = val_zd.sample(n=zd_n, random_state=SEED)

    val_known["_is_zeroday"] = False
    val_zd["_is_zeroday"] = True

    combined = pd.concat([val_known, val_zd], ignore_index=True)
    print(f"Loaded {len(combined)} samples ({known_n} known + {zd_n} zeroday)")
    return combined


def extract_features(text: str, feature_names: list[str]) -> dict:
    """Extract numeric features from kv-format text string."""
    features = {}
    for part in text.split(", "):
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip()
            if key in feature_names:
                try:
                    features[key] = float(val)
                except ValueError:
                    features[key] = val
    return features


def main():
    # Load RF model
    print(f"Loading RF model from {ML_MODEL}")
    with open(ML_MODEL, "rb") as f:
        model_data = pickle.load(f)

    rf = model_data["clf"]
    feature_names = model_data["feature_names"]
    cat_features = model_data["cat_features"]
    num_features = model_data["num_features"]
    ordinal_encoder = model_data["ordinal_encoder"]
    label_encoder = model_data["label_encoder"]

    print(f"RF model: {type(rf).__name__}, {len(feature_names)} features "
          f"({len(cat_features)} cat + {len(num_features)} num)")

    # Load data
    data = load_eval_data()

    # Parse text column (kv-format) → feature DataFrame
    # This is the SAME path used by check_anomaly via BetaAgentML._texts_to_dataframe
    def parse_kv_text(text: str) -> dict:
        result = {}
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                result[k.strip()] = v.strip()
        return result

    rows = [parse_kv_text(t) for t in data["text"].tolist()]
    feat_df = pd.DataFrame(rows)
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = "missing"
    feat_df = feat_df[feature_names]

    # Encode: same logic as BetaAgentML._encode_features
    parts = []

    # Categorical → OrdinalEncoder
    if cat_features:
        cat_df = feat_df[cat_features].fillna("UNK").astype(str)
        cat_encoded = ordinal_encoder.transform(cat_df)
        parts.append(cat_encoded)

    # Binned numeric → manual ordinal
    if num_features:
        bin_order = {"missing": 0, "zero": 1, "positive": 2, "low": 3,
                     "medium": 4, "high": 5, "very_high": 6}
        num_df = feat_df[num_features].fillna("missing").astype(str)
        num_encoded = num_df.apply(
            lambda col: col.map(lambda v: bin_order.get(v, 0))
        ).values.astype(np.float32)
        parts.append(num_encoded)

    X = np.hstack(parts)
    print(f"Feature matrix: {X.shape}")

    # Run RF predictions
    print(f"\nRunning RF on {len(data)} samples...")
    t0 = time.time()

    preds = rf.predict(X)
    probas = rf.predict_proba(X)

    elapsed = time.time() - t0
    print(f"RF inference: {elapsed:.3f}s ({elapsed/len(data)*1000:.2f} ms/sample)")

    # Decode predictions
    if label_encoder is not None:
        pred_labels = label_encoder.inverse_transform(preds)
    else:
        pred_labels = preds

    # Compute metrics
    labels = data["label_name"].values
    is_zeroday = data["_is_zeroday"].values

    tp = fp = fn = tn = 0
    zd_detected = 0
    zd_total = int(is_zeroday.sum())
    audit_records = []

    for i in range(len(data)):
        gt = labels[i]
        pred = pred_labels[i]
        gt_attack = gt.lower() not in ("normal", "benign")
        pred_attack = pred.lower() not in ("normal", "benign")

        if gt_attack and pred_attack:
            tp += 1
        elif not gt_attack and pred_attack:
            fp += 1
        elif gt_attack and not pred_attack:
            fn += 1
        else:
            tn += 1

        if is_zeroday[i] and pred_attack:
            zd_detected += 1

        # Deterministic audit record
        confidence = float(probas[i].max())
        audit_records.append({
            "version": "rf_ablation",
            "sample_index": i,
            "input": {"is_zeroday": bool(is_zeroday[i])},
            "result": {
                "verdict": "attack" if pred_attack else "benign",
                "predicted_class": str(pred),
                "confidence": round(confidence, 4),
            },
            "evaluation": {
                "detection_correct": pred_attack == gt_attack,
                "binary_pred": "attack" if pred_attack else "benign",
                "binary_gt": "attack" if gt_attack else "benign",
            },
            "efficiency": {
                "latency_ms": round(elapsed / len(data) * 1000, 2),
                "tool_calls": 0,
                "llm_calls": 0,
            },
        })

    n = len(data)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"RF ABLATION RESULTS (N={n}):")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")
    print(f"  FPR:       {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"  ZD Rate:   {zd_rate:.4f} ({zd_rate*100:.1f}%)")
    print(f"  Latency:   {elapsed/n*1000:.2f} ms/sample")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"{'='*60}")

    # Compare with E3/E4
    print(f"\nComparison:")
    print(f"  RF ablation: {accuracy*100:.1f}% acc, {f1*100:.1f}% F1, {zd_rate*100:.1f}% ZD")
    print(f"  E3 (paper):  91.7% acc, 93.9% F1, 96.0% ZD")
    print(f"  E4 (paper):  92.4% acc, 94.4% F1, 97.0% ZD")
    print(f"  Baseline RF: 92.4% acc, 94.4% F1, 97.0% ZD")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "n_samples", "accuracy", "precision",
                         "recall", "f1", "fpr", "tp", "fp", "fn", "tn",
                         "zeroday_total", "zeroday_detected",
                         "zeroday_detection_rate", "avg_latency_ms"])
        writer.writerow(["RF_ablation", n, round(accuracy, 4), round(precision, 4),
                         round(recall, 4), round(f1, 4), round(fpr, 4),
                         tp, fp, fn, tn, zd_total, zd_detected,
                         round(zd_rate, 4), round(elapsed / n * 1000, 2)])
    print(f"\nCSV saved: {OUT_CSV}")

    # Save audit JSONL
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AUDIT, "w") as f:
        for rec in audit_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Audit saved: {OUT_AUDIT}")


if __name__ == "__main__":
    main()
