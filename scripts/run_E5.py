#!/usr/bin/env python3
"""
run_E5.py — E5 Ours-SecHarness: Full Dual-Agent + Verification Loop + Audit Trail.

Agent-Alpha (fine-tuned LLM via HF) + Agent-Beta (RF from E2)
+ ConsensusModule (alpha_priority) + AuditLogger (full observability).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from project.src.agents.alpha_agent_hf import AlphaAgentHF
from project.src.agents.beta_agent_ml import BetaAgentML
from project.src.agents.consensus import ConsensusModule
from project.src.harness.observability import AuditLogger, build_audit_record
from project.src.harness.metrics import (
    export_all_metrics, compute_binary_metrics, compute_zeroday_rate,
    compute_harness_metrics, compute_efficiency_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("E5")


def run_experiment(
    df: pd.DataFrame,
    alpha_agent: AlphaAgentHF,
    beta_agent: BetaAgentML,
    consensus: ConsensusModule,
    alogger: AuditLogger,
    text_col: str = "text",
    label_col: str = "label_name",
    tag: str = "",
) -> None:
    n = len(df)
    log.info("[%s] Starting SecHarness pipeline on %d samples", tag, n)
    t0 = time.time()

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    # Beta: batch inference (fast)
    log.info("[%s] Running Beta ML batch inference...", tag)
    beta_verdicts = beta_agent.analyze_batch(texts)
    log.info("[%s] Beta done in %.1fs", tag, time.time() - t0)

    disagree_count = 0

    # Alpha: sequential HF inference + consensus
    for i, (text, gt) in enumerate(zip(texts, labels)):
        av = alpha_agent.analyze(text)
        bv = beta_verdicts[i]

        # Consensus Module — the core verification loop
        cr = consensus.resolve(av, bv)

        if cr.is_disagreement:
            disagree_count += 1

        record = build_audit_record(
            sample_index=i,
            traffic_text=text,
            alpha_verdict=av,
            beta_verdict=bv,
            consensus_result=cr,
            ground_truth_label=gt,
        )
        alogger.log(record)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            log.info("[%s] %d/%d (%.1f/min, ETA %.0fs) [disagree=%d]",
                     tag, i + 1, n, rate * 60, eta, disagree_count)


def main():
    ap = argparse.ArgumentParser(description="E5 SecHarness full experiment")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "E5_unsw.yaml"))
    ap.add_argument("--pilot", action="store_true", help="Run pilot (100 samples)")
    ap.add_argument("--subsample", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Paths
    base_dir = PROJECT_ROOT / cfg["data"]["base_dir"].replace("project/", "")
    val_known_path = base_dir / cfg["data"]["val_known"]
    val_zeroday_path = base_dir / cfg["data"]["val_zeroday"]
    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"].replace("project/", "")
    logs_dir = PROJECT_ROOT / cfg["output"]["logs_dir"].replace("project/", "")
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg["output"]["prefix"]
    zeroday_classes = cfg["data"]["zeroday_classes"]
    seed = cfg["sampling"]["seed"]

    # Load data
    val_known = pd.read_csv(val_known_path)
    val_zeroday = pd.read_csv(val_zeroday_path)
    log.info("Loaded val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))

    # Sampling
    if args.pilot:
        n = cfg["sampling"]["pilot_n"]
        val_known = val_known.sample(n=min(n, len(val_known)), random_state=seed)
        val_zeroday = val_zeroday.sample(n=min(50, len(val_zeroday)), random_state=seed)
        log.info("PILOT mode: known=%d, zeroday=%d", len(val_known), len(val_zeroday))
    elif args.subsample > 0:
        zd_n = min(int(args.subsample * 0.05), len(val_zeroday))
        known_n = min(args.subsample - zd_n, len(val_known))
        val_known = val_known.sample(n=known_n, random_state=seed)
        val_zeroday = val_zeroday.sample(n=zd_n, random_state=seed)
        log.info("SUBSAMPLE: known=%d, zeroday=%d", len(val_known), len(val_zeroday))

    combined = pd.concat([val_known, val_zeroday], ignore_index=True)

    # Load agents
    acfg = cfg["agent_alpha_hf"]
    adapter_path = PROJECT_ROOT / acfg["adapter_path"].replace("project/", "")
    log.info("Loading Alpha agent (HF fine-tuned)...")
    alpha_agent = AlphaAgentHF(
        base_model=acfg["base_model"],
        adapter_path=str(adapter_path),
        load_in_4bit=acfg.get("load_in_4bit", False),
        temperature=acfg["temperature"],
        max_new_tokens=acfg["max_new_tokens"],
        max_input_length=acfg.get("max_input_length", 512),
    )
    log.info("Alpha ready: %s", alpha_agent)

    bcfg = cfg["agent_beta_ml"]
    model_path = PROJECT_ROOT / bcfg["model_path"].replace("project/", "")
    log.info("Loading Beta ML model from %s", model_path)
    beta_agent = BetaAgentML.load(model_path)
    log.info("Beta ready: %s", beta_agent)

    # Consensus Module
    ccfg = cfg["consensus"]
    consensus = ConsensusModule(
        alpha_weight=ccfg["alpha_weight"],
        beta_weight=ccfg["beta_weight"],
        disagreement_strategy=ccfg["disagreement_strategy"],
        confidence_threshold=ccfg["confidence_threshold"],
    )
    log.info("Consensus ready: %s", consensus)

    # Run
    audit_path = logs_dir / f"{prefix}_audit.jsonl"
    with AuditLogger(audit_path) as alogger:
        run_experiment(
            combined, alpha_agent, beta_agent, consensus, alogger,
            text_col=cfg["data"]["text_col"],
            label_col=cfg["data"]["label_col"],
            tag="E5",
        )
        records = alogger.records

    log.info("Total audit records: %d", len(records))

    # Compute and export metrics — derive dataset tag from prefix
    dataset_tag = "_".join(prefix.split("_")[1:])
    exported = export_all_metrics(
        records=records,
        experiment_id="E5",
        dataset=dataset_tag,
        output_dir=results_dir,
        zeroday_classes=zeroday_classes,
    )
    for group, path in exported.items():
        log.info("Exported %s → %s", group, path)

    # Summary
    bm = compute_binary_metrics(records)
    zd = compute_zeroday_rate(records, zeroday_classes)
    hm = compute_harness_metrics(records)
    em = compute_efficiency_metrics(records)

    log.info("=== E5 SecHarness Results ===")
    log.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  FPR=%.4f",
             bm.get("accuracy", 0), bm.get("precision", 0), bm.get("recall", 0),
             bm.get("f1", 0), bm.get("fpr", 0))
    log.info("Zero-day: %d/%d = %.4f",
             zd.get("zeroday_detected", 0), zd.get("zeroday_total", 0),
             zd.get("zeroday_detection_rate", 0))
    log.info("Disagreement rate: %.4f  Escalation rate: %.4f",
             hm.get("disagreement_rate", 0), hm.get("escalation_rate", 0))
    log.info("Consensus distribution: %s",
             {k: v for k, v in hm.items() if k.startswith("consensus_")})
    log.info("Avg latency: %.0f ms/sample (Alpha: %.0f, Beta: %.0f, Consensus: %.0f)",
             em.get("avg_total_latency_ms", 0), em.get("avg_alpha_latency_ms", 0),
             em.get("avg_beta_latency_ms", 0), em.get("avg_consensus_latency_ms", 0))

    # Extract key disagreement samples for analysis
    disagreements = alogger.disagreement_samples()
    if disagreements:
        log.info("=== Sample Disagreements (%d total) ===", len(disagreements))
        for d in disagreements[:5]:
            log.info("  [%s] GT=%s | Alpha=%s(%.2f) | Beta=%s(%.2f) | Final=%s | Type=%s",
                     d.consensus_result.consensus_type,
                     d.ground_truth_label,
                     d.alpha_verdict.verdict, d.alpha_verdict.confidence,
                     d.beta_verdict.verdict, d.beta_verdict.confidence,
                     d.consensus_result.final_verdict,
                     d.consensus_result.final_attack_type)

        # Save disagreement details
        disagree_path = logs_dir / f"{prefix}_disagreements.jsonl"
        with open(disagree_path, "w") as f:
            for d in disagreements:
                f.write(d.to_json() + "\n")
        log.info("Disagreement details saved: %s (%d records)", disagree_path, len(disagreements))

    log.info("Done.")


if __name__ == "__main__":
    main()
