#!/usr/bin/env python3
"""
run_E2.py — E2 Baseline-Hybrid: LLM + ML Classifier on UNSW-NB15.

Agent-Alpha (LLM) + BetaAgentML (RandomForest/XGBoost).
Simple weighted fusion of confidence scores — NOT the full Consensus Module.
Fusion rule: if either agent says "attack" AND fused_confidence > threshold → attack.
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
from project.src.agents.alpha_agent import AlphaAgent
from project.src.agents.beta_agent_ml import BetaAgentML
from project.src.agents.consensus import ConsensusResult
from project.src.harness.observability import AuditLogger, build_audit_record
from project.src.harness.metrics import export_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("E2")


# ---------------------------------------------------------------------------
# Simple weighted fusion (E2-specific, NOT the Consensus Module)
# ---------------------------------------------------------------------------
def simple_fusion(
    alpha: AgentVerdict,
    beta: AgentVerdict,
    alpha_weight: float = 0.5,
    beta_weight: float = 0.5,
    attack_threshold: float = 0.5,
) -> ConsensusResult:
    """
    Weighted average fusion for E2.

    Logic:
    - If both say benign → benign (fused confidence)
    - If both say attack → attack (fused confidence, Alpha's attack_type)
    - If one says attack: fused_conf = weighted avg of confidence scores.
      If the attacker's weighted confidence > threshold → attack, else benign.
    """
    fused_conf = alpha_weight * alpha.confidence + beta_weight * beta.confidence

    if alpha.verdict == "benign" and beta.verdict == "benign":
        return ConsensusResult(
            final_verdict="benign",
            consensus_type="agreement",
            combined_confidence=fused_conf,
            alpha_verdict=alpha,
            beta_verdict=beta,
            latency_ms=0.0,
        )

    if alpha.verdict == "attack" and beta.verdict == "attack":
        return ConsensusResult(
            final_verdict="attack",
            final_attack_type=alpha.attack_type or beta.attack_type,
            consensus_type="agreement",
            combined_confidence=fused_conf,
            alpha_verdict=alpha,
            beta_verdict=beta,
            latency_ms=0.0,
        )

    # Disagreement: use fused confidence to decide
    # The agent that says "attack" contributes its weighted confidence
    if alpha.verdict == "attack":
        attack_score = alpha_weight * alpha.confidence
        attack_type = alpha.attack_type
        ctype = "alpha_only"
    else:
        attack_score = beta_weight * beta.confidence
        attack_type = beta.attack_type
        ctype = "beta_only"

    if attack_score > attack_threshold * max(alpha_weight, beta_weight):
        final = "attack"
    else:
        final = "benign"
        attack_type = None

    return ConsensusResult(
        final_verdict=final,
        final_attack_type=attack_type,
        consensus_type=ctype,
        combined_confidence=fused_conf,
        alpha_verdict=alpha,
        beta_verdict=beta,
        latency_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
def run_on_df(
    df: pd.DataFrame,
    alpha_agent: AlphaAgent,
    beta_agent: BetaAgentML,
    alogger: AuditLogger,
    fusion_cfg: dict,
    text_col: str = "text",
    label_col: str = "label_name",
    batch_size: int = 10,
    tag: str = "",
) -> None:
    n = len(df)
    log.info("[%s] Starting inference on %d samples", tag, n)
    t0 = time.time()

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    aw = fusion_cfg.get("alpha_weight", 0.5)
    bw = fusion_cfg.get("beta_weight", 0.5)
    thresh = fusion_cfg.get("attack_threshold", 0.5)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        batch_labels = labels[start:end]

        # Alpha (LLM) + Beta (ML) inference
        alpha_verdicts = alpha_agent.analyze_batch(batch_texts)
        beta_verdicts = beta_agent.analyze_batch(batch_texts)

        for i, (av, bv, gt) in enumerate(zip(alpha_verdicts, beta_verdicts, batch_labels)):
            cr = simple_fusion(av, bv, alpha_weight=aw, beta_weight=bw, attack_threshold=thresh)
            record = build_audit_record(
                sample_index=start + i,
                traffic_text=batch_texts[i],
                alpha_verdict=av,
                beta_verdict=bv,
                consensus_result=cr,
                ground_truth_label=gt,
            )
            alogger.log(record)

        elapsed = time.time() - t0
        done = end
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        log.info("[%s] %d/%d (%.1f/min, ETA %.0fs)", tag, done, n, rate * 60, eta)


# ---------------------------------------------------------------------------
# ML-only evaluation (no LLM, for fast initial results)
# ---------------------------------------------------------------------------
def run_ml_only(
    df: pd.DataFrame,
    beta_agent: BetaAgentML,
    alogger: AuditLogger,
    text_col: str = "text",
    label_col: str = "label_name",
    tag: str = "",
) -> None:
    """Run Beta ML agent only, with dummy Alpha. Fast evaluation."""
    n = len(df)
    log.info("[%s] ML-only mode on %d samples", tag, n)

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    t0 = time.time()
    beta_verdicts = beta_agent.analyze_batch(texts)
    log.info("[%s] ML batch inference: %.1fs", tag, time.time() - t0)

    for i, (bv, gt) in enumerate(zip(beta_verdicts, labels)):
        # Create dummy Alpha that mirrors Beta for ML-only baseline
        dummy_alpha = AgentVerdict(
            verdict=bv.verdict, confidence=bv.confidence,
            attack_type=bv.attack_type,
            reasoning="[E2-MLonly] mirrored from Beta",
            agent_name="Alpha-Disabled",
        )
        cr = ConsensusResult(
            final_verdict=bv.verdict,
            final_attack_type=bv.attack_type,
            consensus_type="agreement",
            combined_confidence=bv.confidence,
            alpha_verdict=dummy_alpha,
            beta_verdict=bv,
            latency_ms=0.0,
        )
        record = build_audit_record(
            sample_index=i,
            traffic_text=texts[i],
            alpha_verdict=dummy_alpha,
            beta_verdict=bv,
            consensus_result=cr,
            ground_truth_label=gt,
        )
        alogger.log(record)

    log.info("[%s] Done: %d records", tag, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="E2 Baseline-Hybrid experiment")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "E2_unsw.yaml"))
    ap.add_argument("--pilot", action="store_true", help="Run pilot (100 samples) only")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--ml-only", action="store_true",
                     help="Skip LLM, use ML classifier only (fast baseline)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Paths
    base_dir = PROJECT_ROOT / cfg["data"]["base_dir"].replace("project/", "")
    train_path = base_dir / cfg["data"]["train"]
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
        log.info("PILOT mode: known=%d, zeroday=%d", len(val_known), len(val_zeroday))
    elif args.subsample > 0:
        zd_n = min(int(args.subsample * 0.1), len(val_zeroday))
        known_n = min(args.subsample - zd_n, len(val_known))
        val_known = val_known.sample(n=known_n, random_state=seed)
        val_zeroday = val_zeroday.sample(n=zd_n, random_state=seed)
        log.info("SUBSAMPLE: known=%d, zeroday=%d", len(val_known), len(val_zeroday))

    # --- Train ML classifier ---
    bcfg = cfg["agent_beta_ml"]
    beta_agent = BetaAgentML(
        classifier_type=bcfg["classifier_type"],
        n_estimators=bcfg["n_estimators"],
        max_depth=bcfg["max_depth"],
        random_state=bcfg["random_state"],
    )

    log.info("Training ML classifier on %s ...", train_path)
    train_stats = beta_agent.train(
        train_csv=train_path,
        text_col=cfg["data"]["text_col"],
        label_col=cfg["data"]["label_col"],
    )
    log.info("ML training done: %s", {k: v for k, v in train_stats.items()
                                       if k not in ("cat_features", "num_features_list", "classes")})

    # Save model
    model_path = PROJECT_ROOT / bcfg["model_save_path"].replace("project/", "")
    beta_agent.save(model_path)
    log.info("ML model saved: %s", model_path)

    # Combine data
    combined = pd.concat([val_known, val_zeroday], ignore_index=True)

    if args.ml_only:
        # Fast ML-only evaluation
        audit_path = logs_dir / f"{prefix}_ml_only_audit.jsonl"
        with AuditLogger(audit_path) as alogger:
            run_ml_only(
                combined, beta_agent, alogger,
                text_col=cfg["data"]["text_col"],
                label_col=cfg["data"]["label_col"],
                tag="E2-ML",
            )
            records = alogger.records

        result_prefix = f"{prefix}_ml_only"
    else:
        # Full hybrid: Alpha (LLM) + Beta (ML)
        acfg = cfg["agent_alpha"]
        alpha_agent = AlphaAgent(
            model=acfg["model"],
            ollama_url=acfg["ollama_url"],
            temperature=acfg["temperature"],
            max_tokens=acfg["max_tokens"],
            timeout=acfg["timeout"],
            max_workers=acfg["max_workers"],
        )

        if not alpha_agent.check_health():
            log.error("Ollama health check failed!")
            sys.exit(1)
        log.info("Agent-Alpha ready: %s", alpha_agent)

        audit_path = logs_dir / f"{prefix}_audit.jsonl"
        with AuditLogger(audit_path) as alogger:
            run_on_df(
                combined, alpha_agent, beta_agent, alogger,
                fusion_cfg=cfg["fusion"],
                text_col=cfg["data"]["text_col"],
                label_col=cfg["data"]["label_col"],
                batch_size=10,
                tag="E2",
            )
            records = alogger.records

        result_prefix = prefix

    log.info("Total audit records: %d", len(records))

    # Compute metrics — derive dataset tag from prefix
    exp_id = "E2"
    # Extract dataset tag: E2_unsw → unsw, E2_cic_full → cic_full
    dataset_tag = "_".join(prefix.split("_")[1:])
    if "ml_only" in result_prefix:
        dataset_tag = dataset_tag + "_ml_only"

    exported = export_all_metrics(
        records=records,
        experiment_id=exp_id,
        dataset=dataset_tag,
        output_dir=results_dir,
        zeroday_classes=zeroday_classes,
    )

    for group, path in exported.items():
        log.info("Exported %s → %s", group, path)

    # Summary
    from project.src.harness.metrics import compute_binary_metrics, compute_zeroday_rate
    bm = compute_binary_metrics(records)
    zd = compute_zeroday_rate(records, zeroday_classes)
    log.info("=== E2 Results ===")
    log.info("Accuracy=%.4f  F1=%.4f  FPR=%.4f",
             bm.get("accuracy", 0), bm.get("f1", 0), bm.get("fpr", 0))
    log.info("Zero-day: %d/%d = %.4f",
             zd.get("zeroday_detected", 0), zd.get("zeroday_total", 0),
             zd.get("zeroday_detection_rate", 0))
    log.info("ML train stats: acc=%.4f, %d samples, %d features",
             train_stats["train_accuracy"], train_stats["train_samples"], train_stats["num_features"])
    log.info("Done.")


if __name__ == "__main__":
    main()
