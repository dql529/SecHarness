#!/usr/bin/env python3
"""
run_multiseed_e3.py — Run E3 with 5 different evaluation sampling seeds.

Assesses result stability by varying the 1000-sample subset drawn from
the validation set. The harness, model, and tools remain identical;
only the random seed for sample selection changes.

Supports both Mac (M5 Max, MPS) and gpu-box (3090Ti, CUDA).
Outputs per-seed metrics + summary.
"""

import csv
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

# Enable offline mode for HuggingFace (model already cached locally)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multiseed_e3")

# ──────────────────────────────────────────────────────────────────────────────
# Paths — auto-detect platform
# ──────────────────────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT.parent))
else:
    ROOT = Path(__file__).resolve().parents[1]  # repository root
    sys.path.insert(0, str(ROOT.parent))

VAL_KNOWN = ROOT / "data/processed/unsw_nb15/zeroday_split/val_known.csv"
VAL_ZD = ROOT / "data/processed/unsw_nb15/zeroday_split/val_zeroday.csv"
ML_MODEL = ROOT / "logs/E2_beta_ml_model.pkl"
KNOWLEDGE_DIR = ROOT / "data/knowledge"
SIGNATURES_DIR = ROOT / "data/knowledge/signatures"

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
EVAL_N = 1000
SEEDS = [42, 123, 456, 789, 2026]  # seed=42 is our original E3

OUT_DIR = ROOT / "results/tables/multiseed"
LOG_DIR = ROOT / "logs/multiseed"


def load_eval_set(seed: int) -> pd.DataFrame:
    """Load N=1000 stratified eval set with given seed."""
    val_known = pd.read_csv(str(VAL_KNOWN))
    val_zd = pd.read_csv(str(VAL_ZD))

    zd_n = min(int(EVAL_N * 0.1), len(val_zd))
    known_n = EVAL_N - zd_n

    val_known_sub = val_known.sample(n=known_n, random_state=seed)
    val_zd_sub = val_zd.sample(n=zd_n, random_state=seed)

    val_known_sub["_is_zeroday"] = False
    val_zd_sub["_is_zeroday"] = True

    combined = pd.concat([val_known_sub, val_zd_sub], ignore_index=True)
    log.info("Seed %d: %d known + %d zeroday = %d total", seed, known_n, zd_n, len(combined))
    return combined


def run_e3_single_seed(seed: int) -> dict:
    """Run E3 on one seed, return metrics dict."""
    import torch
    from project.src.v2.llm_engine import LLMEngine
    from project.src.v2.agent_loop import SecHarness, agent_loop
    from project.src.v2.harness.audit import AuditLoggerV2

    audit_path = str(LOG_DIR / f"E3_seed{seed}_sub{EVAL_N}_audit.jsonl")

    data = load_eval_set(seed)

    log.info("Loading model: %s", BASE_MODEL)
    engine = LLMEngine(
        base_model=BASE_MODEL,
        adapter_path=None,
        load_in_4bit=False,
    )

    harness = SecHarness(
        llm=engine,
        ml_model_path=str(ML_MODEL),
        signatures_dir=str(SIGNATURES_DIR),
        knowledge_dir=str(KNOWLEDGE_DIR),
        harness_enabled=True,
        enabled_tools=["check_anomaly", "lookup_signature", "query_history",
                       "load_knowledge", "classify", "escalate", "log_decision"],
        permissions_enabled=True,
        max_steps=5,
        max_new_tokens=256,
        temperature=0.1,
        confidence_threshold=0.3,
        audit_logger=AuditLoggerV2(audit_path),
        check_anomaly_degraded=False,
    )

    texts = data["text"].tolist()
    labels = data["label_name"].tolist()
    is_zeroday = data["_is_zeroday"].tolist()

    log.info("Running E3 seed=%d on %d samples...", seed, len(texts))
    t0 = time.time()

    tp = fp = fn = tn = 0
    zd_detected = 0
    zd_total = sum(is_zeroday)
    latencies = []

    for i, (text, label, zd) in enumerate(zip(texts, labels, is_zeroday)):
        t_s = time.time()
        result = agent_loop(text, harness, i, label, zd)
        latencies.append((time.time() - t_s) * 1000)

        gt_attack = label.lower() not in ("normal", "benign")
        pred_attack = result.verdict == "attack"

        if gt_attack and pred_attack:
            tp += 1
        elif not gt_attack and pred_attack:
            fp += 1
        elif gt_attack and not pred_attack:
            fn += 1
        else:
            tn += 1

        if zd and pred_attack:
            zd_detected += 1

        if (i + 1) % 100 == 0:
            acc = (tp + tn) / (i + 1)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            log.info("[seed=%d] [%d/%d] acc=%.1f%% rate=%.1f/s ETA=%.0fm",
                     seed, i + 1, len(texts), acc * 100, rate, eta / 60)

    harness.audit_logger.close()

    # Free GPU/MPS memory for next seed
    del engine, harness
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    n = len(texts)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0
    avg_latency = np.mean(latencies)

    metrics = {
        "seed": seed, "n": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "zd_rate": round(zd_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "zd_total": zd_total, "zd_detected": zd_detected,
        "avg_latency_ms": round(avg_latency, 1),
    }

    elapsed = time.time() - t0
    log.info("Seed %d done in %.1fm: acc=%.1f%% f1=%.1f%% zd=%.1f%%",
             seed, elapsed / 60, accuracy * 100, f1 * 100, zd_rate * 100)

    return metrics


def main():
    log.info("=" * 60)
    log.info("Multi-seed E3 evaluation (5 seeds × N=%d)", EVAL_N)
    log.info("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    t_total = time.time()

    for seed in SEEDS:
        log.info("\n" + "=" * 60)
        log.info("SEED %d (%d/%d)", seed, SEEDS.index(seed) + 1, len(SEEDS))
        log.info("=" * 60)

        metrics = run_e3_single_seed(seed)
        all_metrics.append(metrics)

        # Save per-seed results incrementally
        out_csv = OUT_DIR / "multiseed_e3_results.csv"
        with open(str(out_csv), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)
        log.info("Incremental results saved to %s", out_csv)

    # Summary statistics
    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]
    zds = [m["zd_rate"] for m in all_metrics]

    log.info("\n" + "=" * 60)
    log.info("MULTI-SEED SUMMARY")
    log.info("=" * 60)
    log.info("Accuracy: %.1f%% ± %.2f%% (range: %.1f%%–%.1f%%)",
             np.mean(accs) * 100, np.std(accs) * 100,
             np.min(accs) * 100, np.max(accs) * 100)
    log.info("F1:       %.1f%% ± %.2f%% (range: %.1f%%–%.1f%%)",
             np.mean(f1s) * 100, np.std(f1s) * 100,
             np.min(f1s) * 100, np.max(f1s) * 100)
    log.info("ZD Rate:  %.1f%% ± %.2f%% (range: %.1f%%–%.1f%%)",
             np.mean(zds) * 100, np.std(zds) * 100,
             np.min(zds) * 100, np.max(zds) * 100)

    total_elapsed = time.time() - t_total
    log.info("\nTotal runtime: %.1f hours", total_elapsed / 3600)

    # Save summary
    summary = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "zd_rate_mean": round(float(np.mean(zds)), 4),
        "zd_rate_std": round(float(np.std(zds)), 4),
        "per_seed": all_metrics,
    }
    summary_path = OUT_DIR / "multiseed_e3_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
