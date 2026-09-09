#!/usr/bin/env python3
"""
run_E4.py — E4 Ours-NoHarness: Dual-Agent without Verification Loop.

Agent-Alpha (fine-tuned LLM via HF) + Agent-Beta (RF from E2).
No Consensus Module: Alpha's verdict is the final decision.
Beta runs for audit comparison only.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from project.src.agents.base_agent import AgentVerdict
from project.src.agents.alpha_agent_hf import AlphaAgentHF
from project.src.agents.beta_agent_ml import BetaAgentML
from project.src.agents.consensus import ConsensusResult
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
log = logging.getLogger("E4")


def alpha_only_result(alpha: AgentVerdict, beta: AgentVerdict) -> ConsensusResult:
    """E4: No consensus — Alpha's verdict is final. Beta is recorded but not used."""
    return ConsensusResult(
        final_verdict=alpha.verdict,
        final_attack_type=alpha.attack_type if alpha.verdict == "attack" else None,
        consensus_type="alpha_only" if alpha.verdict != beta.verdict else "agreement",
        combined_confidence=alpha.confidence,
        alpha_verdict=alpha,
        beta_verdict=beta,
        latency_ms=0.0,
    )


def run_experiment(
    df: pd.DataFrame,
    alpha_agent: AlphaAgentHF,
    beta_agent: BetaAgentML,
    alogger: AuditLogger,
    text_col: str = "text",
    label_col: str = "label_name",
    tag: str = "",
) -> None:
    n = len(df)
    log.info("[%s] Starting inference on %d samples", tag, n)
    t0 = time.time()

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    # Beta: batch inference (fast)
    log.info("[%s] Running Beta ML batch inference...", tag)
    beta_verdicts = beta_agent.analyze_batch(texts)
    log.info("[%s] Beta done in %.1fs", tag, time.time() - t0)

    # Alpha: sequential HF inference
    for i, (text, gt) in enumerate(zip(texts, labels)):
        av = alpha_agent.analyze(text)
        bv = beta_verdicts[i]
        cr = alpha_only_result(av, bv)

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
            log.info("[%s] %d/%d (%.1f/min, ETA %.0fs)", tag, i + 1, n, rate * 60, eta)


def main():
    ap = argparse.ArgumentParser(description="E4 Dual-Agent NoHarness experiment")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "E4_unsw.yaml"))
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

    # Run
    audit_path = logs_dir / f"{prefix}_audit.jsonl"
    with AuditLogger(audit_path) as alogger:
        run_experiment(
            combined, alpha_agent, beta_agent, alogger,
            text_col=cfg["data"]["text_col"],
            label_col=cfg["data"]["label_col"],
            tag="E4",
        )
        records = alogger.records

    log.info("Total audit records: %d", len(records))

    # Compute metrics — derive dataset tag from prefix
    dataset_tag = "_".join(prefix.split("_")[1:])
    exported = export_all_metrics(
        records=records,
        experiment_id="E4",
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

    log.info("=== E4 Results ===")
    log.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  FPR=%.4f",
             bm.get("accuracy", 0), bm.get("precision", 0), bm.get("recall", 0),
             bm.get("f1", 0), bm.get("fpr", 0))
    log.info("Zero-day: %d/%d = %.4f",
             zd.get("zeroday_detected", 0), zd.get("zeroday_total", 0),
             zd.get("zeroday_detection_rate", 0))
    log.info("Disagreement rate: %.4f", hm.get("disagreement_rate", 0))
    log.info("Avg latency: %.0f ms/sample", em.get("avg_total_latency_ms", 0))
    log.info("Done.")


if __name__ == "__main__":
    main()
