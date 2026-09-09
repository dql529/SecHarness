#!/usr/bin/env python3
"""
gpu_pipeline.py — Clean E4 pipeline for gpu-box (3090Ti).

Fixes data leakage: trains adapter on TRAIN split trajectories,
evaluates E4 on VAL split (no overlap).

Steps:
  1. Run E3 on train_known.csv (2000 samples) → E3_train_audit.jsonl
  2. Extract trajectories → traj_train_clean.jsonl
  3. Train qlora_unsw_v2_clean adapter (1 epoch)
  4. Evaluate E4 on val sub1000 → E4_clean_results.csv
"""

import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gpu_pipeline")

# ──────────────────────────────────────────────────────────────────────────────
# Paths (Windows-style, absolute)
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # repository root
PYTHON = r"E:/miniconda3/envs/llm_torch_backup/python.exe"

# Make the repository package importable from ROOT.parent
sys.path.insert(0, str(ROOT.parent))

TRAIN_CSV = ROOT / "data/processed/unsw_nb15/zeroday_split/train_known.csv"
VAL_KNOWN_CSV = ROOT / "data/processed/unsw_nb15/zeroday_split/val_known.csv"
VAL_ZD_CSV = ROOT / "data/processed/unsw_nb15/zeroday_split/val_zeroday.csv"
ML_MODEL = ROOT / "logs/E2_beta_ml_model.pkl"
KNOWLEDGE_DIR = ROOT / "data/knowledge"
SIGNATURES_DIR = ROOT / "data/knowledge/signatures"

E3_AUDIT = ROOT / "logs/E3_train_sub2000_audit.jsonl"
TRAJ_JSONL = ROOT / "logs/traj_train_clean.jsonl"
CLEAN_ADAPTER = ROOT / "models/qlora_unsw_v2_clean/adapter"
E4_RESULTS = ROOT / "results/tables/E4_clean_results.csv"
E4_AUDIT = ROOT / "logs/E4_clean_sub1000_audit.jsonl"

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
SEED = 999  # Different from val experiments (seed=42)
TRAIN_N = 2000
EVAL_N = 1000

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Run E3 on train split
# ──────────────────────────────────────────────────────────────────────────────
def step1_run_e3_train():
    log.info("=" * 60)
    log.info("STEP 1: Running E3 on train split (%d samples)", TRAIN_N)
    log.info("=" * 60)

    import pandas as pd
    import torch

    log.info("CUDA available: %s, device: %s", torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # Load and subsample train data
    log.info("Loading %s", TRAIN_CSV)
    train_df = pd.read_csv(str(TRAIN_CSV))
    train_df = train_df.sample(n=min(TRAIN_N, len(train_df)), random_state=SEED)
    train_df["_is_zeroday"] = False  # train split has no zeroday
    log.info("Train subsample: %d rows", len(train_df))

    from project.src.v2.llm_engine import LLMEngine
    from project.src.v2.agent_loop import SecHarness, agent_loop
    from project.src.v2.harness.audit import AuditLoggerV2

    log.info("Loading model: %s (no adapter)", BASE_MODEL)
    engine = LLMEngine(
        base_model=BASE_MODEL,
        adapter_path=None,
        load_in_4bit=False,  # fp16, 3090Ti has plenty VRAM
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
        audit_logger=AuditLoggerV2(str(E3_AUDIT)),
        check_anomaly_degraded=False,
    )

    texts = train_df["text"].tolist()
    labels = train_df["label_name"].tolist()
    is_zeroday = train_df["_is_zeroday"].tolist()

    log.info("Running E3 on %d train samples...", len(texts))
    t0 = time.time()
    correct = 0
    for i, (text, label, zd) in enumerate(zip(texts, labels, is_zeroday)):
        result = agent_loop(text, harness, i, label, zd)
        pred_attack = result.verdict == "attack"
        gt_attack = label.lower() not in ("normal", "benign")
        if pred_attack == gt_attack:
            correct += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            log.info("[%d/%d] acc=%.1f%% rate=%.1f/s ETA=%.0fm",
                     i + 1, len(texts), correct / (i + 1) * 100, rate, eta / 60)

    harness.audit_logger.close()
    elapsed = time.time() - t0
    log.info("STEP 1 done: %d samples in %.1fm, acc=%.1f%%",
             len(texts), elapsed / 60, correct / len(texts) * 100)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Extract trajectories
# ──────────────────────────────────────────────────────────────────────────────
def step2_extract_trajectories():
    log.info("=" * 60)
    log.info("STEP 2: Extracting trajectories from E3_train audit log")
    log.info("=" * 60)

    extract_script = ROOT / "scripts/extract_trajectories.py"
    cmd = [PYTHON, str(extract_script),
           "--audit-log", str(E3_AUDIT),
           "--output", str(TRAJ_JSONL)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("extract_trajectories failed:\n%s", result.stderr)
        sys.exit(1)
    log.info(result.stdout[-2000:])  # Last 2000 chars of output

    # Count trajectories
    count = sum(1 for _ in open(str(TRAJ_JSONL)))
    log.info("STEP 2 done: %d trajectories extracted to %s", count, TRAJ_JSONL)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Train clean adapter
# ──────────────────────────────────────────────────────────────────────────────
def step3_train_adapter():
    log.info("=" * 60)
    log.info("STEP 3: Training clean adapter on train-split trajectories")
    log.info("=" * 60)

    CLEAN_ADAPTER.parent.mkdir(parents=True, exist_ok=True)
    train_script = ROOT / "scripts/train_qlora_v2.py"
    cmd = [PYTHON, str(train_script),
           "--trajectories", str(TRAJ_JSONL),
           "--output-dir", str(CLEAN_ADAPTER.parent),
           "--base-model", BASE_MODEL,
           "--num-epochs", "2",
           "--batch-size", "4",
           "--max-seq-len", "1024",
           "--lora-rank", "16",
           "--lora-alpha", "32"]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("train_qlora_v2 failed:\n%s", result.stderr[-3000:])
        sys.exit(1)
    log.info(result.stdout[-2000:])
    log.info("STEP 3 done: clean adapter saved to %s", CLEAN_ADAPTER)


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate E4 with clean adapter
# ──────────────────────────────────────────────────────────────────────────────
def step4_eval_e4_clean():
    log.info("=" * 60)
    log.info("STEP 4: Evaluating E4 (clean adapter) on val sub%d", EVAL_N)
    log.info("=" * 60)

    import pandas as pd
    from project.src.v2.llm_engine import LLMEngine
    from project.src.v2.agent_loop import SecHarness, agent_loop
    from project.src.v2.harness.audit import AuditLoggerV2

    # Load val data (same protocol as original E4)
    val_known = pd.read_csv(str(VAL_KNOWN_CSV))
    val_zd = pd.read_csv(str(VAL_ZD_CSV))
    zd_n = min(int(EVAL_N * 0.1), len(val_zd))
    known_n = EVAL_N - zd_n
    val_known = val_known.sample(n=known_n, random_state=42)  # same seed as original
    val_zd = val_zd.sample(n=zd_n, random_state=42)
    val_known["_is_zeroday"] = False
    val_zd["_is_zeroday"] = True
    combined = pd.concat([val_known, val_zd], ignore_index=True)
    log.info("Eval set: %d known + %d zeroday = %d total", known_n, zd_n, len(combined))

    log.info("Loading clean adapter from %s", CLEAN_ADAPTER)
    engine = LLMEngine(
        base_model=BASE_MODEL,
        adapter_path=str(CLEAN_ADAPTER),
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
        audit_logger=AuditLoggerV2(str(E4_AUDIT)),
        check_anomaly_degraded=False,
    )

    texts = combined["text"].tolist()
    labels = combined["label_name"].tolist()
    is_zeroday = combined["_is_zeroday"].tolist()

    log.info("Running E4 (clean) on %d val samples...", len(texts))
    t0 = time.time()
    tp = fp = fn = tn = 0
    zd_detected = 0
    zd_total = sum(is_zeroday)
    latencies = []

    for i, (text, label, zd) in enumerate(zip(texts, labels, is_zeroday)):
        t_s = time.time()
        result = agent_loop(text, harness, i, label, zd)
        latencies.append((time.time() - t_s) * 1000)

        pred = result.verdict
        gt_attack = label.lower() not in ("normal", "benign")
        pred_attack = pred == "attack"

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
            eta = (len(texts) - i - 1) / ((i + 1) / elapsed)
            log.info("[%d/%d] acc=%.1f%% ETA=%.0fm",
                     i + 1, len(texts), acc * 100, eta / 60)

    harness.audit_logger.close()

    # Compute metrics
    n = len(texts)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0
    avg_latency = sum(latencies) / len(latencies)

    log.info("=" * 60)
    log.info("E4 CLEAN RESULTS (N=%d):", n)
    log.info("  Accuracy:  %.4f (%.1f%%)", accuracy, accuracy * 100)
    log.info("  Precision: %.4f (%.1f%%)", precision, precision * 100)
    log.info("  Recall:    %.4f (%.1f%%)", recall, recall * 100)
    log.info("  F1:        %.4f (%.1f%%)", f1, f1 * 100)
    log.info("  FPR:       %.4f (%.1f%%)", fpr, fpr * 100)
    log.info("  ZD Rate:   %.4f (%.1f%%)", zd_rate, zd_rate * 100)
    log.info("  Latency:   %.0f ms/sample", avg_latency)
    log.info("=" * 60)

    # Save CSV
    E4_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(str(E4_RESULTS), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "n_samples", "accuracy", "precision",
                         "recall", "f1", "fpr", "zd_total", "zd_detected",
                         "zd_detection_rate", "avg_latency_ms"])
        writer.writerow(["E4_clean", n, round(accuracy, 4), round(precision, 4),
                         round(recall, 4), round(f1, 4), round(fpr, 4),
                         zd_total, zd_detected, round(zd_rate, 4),
                         round(avg_latency, 1)])
    log.info("Results saved to %s", E4_RESULTS)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting clean E4 pipeline on gpu-box")
    log.info("Root: %s", ROOT)

    t_total = time.time()
    # step1_run_e3_train()  # Already done: E3_train_sub2000_audit.jsonl exists
    # step2_extract_trajectories()  # Already done: traj_train_clean.jsonl exists
    # step3_train_adapter()  # Already done: qlora_unsw_v2_clean/adapter exists
    step4_eval_e4_clean()

    log.info("PIPELINE COMPLETE in %.1f hours", (time.time() - t_total) / 3600)
    log.info("Results: %s", E4_RESULTS)
    log.info("Audit:   %s", E4_AUDIT)
