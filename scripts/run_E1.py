#!/usr/bin/env python3
"""
run_E1.py — E1 Baseline-Single: Single LLM Agent on UNSW-NB15.

Alpha Agent only, no verification loop. Alpha's verdict is the final verdict.
Produces AuditRecords compatible with metrics.py for uniform evaluation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from project.src.agents.base_agent import AgentVerdict
from project.src.agents.alpha_agent import AlphaAgent
from project.src.agents.consensus import ConsensusResult
from project.src.harness.observability import AuditRecord, AuditLogger, build_audit_record
from project.src.harness.metrics import export_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("E1")


# ---------------------------------------------------------------------------
# Passthrough: Alpha verdict → ConsensusResult (no Beta, no Consensus Module)
# ---------------------------------------------------------------------------
def alpha_to_consensus(alpha: AgentVerdict) -> ConsensusResult:
    """Convert a single Alpha verdict into a ConsensusResult (passthrough)."""
    return ConsensusResult(
        final_verdict=alpha.verdict,
        final_attack_type=alpha.attack_type,
        consensus_type="agreement",  # trivially "agrees with itself"
        combined_confidence=alpha.confidence,
        alpha_verdict=alpha,
        beta_verdict=None,
        latency_ms=0.0,
    )


def dummy_beta_verdict() -> AgentVerdict:
    """Placeholder Beta verdict for AuditRecord compatibility."""
    return AgentVerdict(
        verdict="benign", confidence=0.0,
        reasoning="[E1] No Beta agent in baseline-single",
        agent_name="Beta-Disabled",
    )


# ---------------------------------------------------------------------------
# Run experiment on a DataFrame
# ---------------------------------------------------------------------------
def run_on_df(
    df: pd.DataFrame,
    agent: AlphaAgent,
    logger: AuditLogger,
    text_col: str = "text",
    label_col: str = "label_name",
    batch_size: int = 10,
    tag: str = "",
) -> None:
    """Run E1 on all rows of df, logging AuditRecords."""
    n = len(df)
    log.info("[%s] Starting inference on %d samples (batch=%d)", tag, n, batch_size)
    t0 = time.time()

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        batch_labels = labels[start:end]

        # Alpha inference
        alpha_verdicts = agent.analyze_batch(batch_texts)

        # Build audit records
        for i, (av, gt) in enumerate(zip(alpha_verdicts, batch_labels)):
            idx = start + i
            cr = alpha_to_consensus(av)
            bv = dummy_beta_verdict()
            record = build_audit_record(
                sample_index=idx,
                traffic_text=batch_texts[i],
                alpha_verdict=av,
                beta_verdict=bv,
                consensus_result=cr,
                ground_truth_label=gt,
            )
            logger.log(record)

        elapsed = time.time() - t0
        done = end
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        log.info(
            "[%s] %d/%d done (%.1f samples/min, ETA %.0fs)",
            tag, done, n, rate * 60, eta,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="E1 Baseline-Single experiment")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "E1_unsw.yaml"))
    ap.add_argument("--pilot", action="store_true", help="Run pilot (100 samples) only")
    ap.add_argument("--subsample", type=int, default=0, help="Use N-sample subsample (0=full)")
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

    # Load data
    val_known = pd.read_csv(val_known_path)
    val_zeroday = pd.read_csv(val_zeroday_path)
    log.info("Loaded val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))

    # Sampling
    seed = cfg["sampling"]["seed"]
    if args.pilot:
        n = cfg["sampling"]["pilot_n"]
        val_known = val_known.sample(n=min(n, len(val_known)), random_state=seed)
        val_zeroday = val_zeroday.sample(n=min(50, len(val_zeroday)), random_state=seed)
        log.info("PILOT mode: val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))
    elif args.subsample > 0:
        n = args.subsample
        # Stratified subsample
        zd_n = min(int(n * 0.1), len(val_zeroday))  # ~10% zero-day
        known_n = min(n - zd_n, len(val_known))
        val_known = val_known.sample(n=known_n, random_state=seed)
        val_zeroday = val_zeroday.sample(n=zd_n, random_state=seed)
        log.info("SUBSAMPLE mode: val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))

    # Initialize Agent-Alpha
    acfg = cfg["agent_alpha"]
    agent = AlphaAgent(
        model=acfg["model"],
        ollama_url=acfg["ollama_url"],
        temperature=acfg["temperature"],
        max_tokens=acfg["max_tokens"],
        timeout=acfg["timeout"],
        max_workers=acfg["max_workers"],
    )

    # Health check
    if not agent.check_health():
        log.error("Ollama health check failed. Is the model available?")
        sys.exit(1)
    log.info("Agent-Alpha ready: %s", agent)

    # Combine known + zeroday for single pass
    combined = pd.concat([val_known, val_zeroday], ignore_index=True)
    combined["_is_zeroday"] = [False] * len(val_known) + [True] * len(val_zeroday)

    # Run
    audit_path = logs_dir / f"{prefix}_audit.jsonl"
    with AuditLogger(audit_path) as alogger:
        run_on_df(
            combined, agent, alogger,
            text_col=cfg["data"]["text_col"],
            label_col=cfg["data"]["label_col"],
            batch_size=10,
            tag="E1",
        )
        records = alogger.records

    log.info("Total audit records: %d", len(records))

    # Compute and export metrics
    exported = export_all_metrics(
        records=records,
        experiment_id=prefix.split("_")[0],
        dataset="unsw",
        output_dir=results_dir,
        zeroday_classes=zeroday_classes,
    )

    for group, path in exported.items():
        log.info("Exported %s → %s", group, path)

    # Quick summary
    from project.src.harness.metrics import compute_binary_metrics, compute_zeroday_rate
    bm = compute_binary_metrics(records)
    zd = compute_zeroday_rate(records, zeroday_classes)
    log.info("=== E1 Results ===")
    log.info("Accuracy=%.4f  F1=%.4f  FPR=%.4f", bm.get("accuracy", 0), bm.get("f1", 0), bm.get("fpr", 0))
    log.info("Zero-day detection: %d/%d = %.4f",
             zd.get("zeroday_detected", 0), zd.get("zeroday_total", 0), zd.get("zeroday_detection_rate", 0))
    log.info("Done.")


if __name__ == "__main__":
    main()
