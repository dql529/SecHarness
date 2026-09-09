#!/usr/bin/env python3
"""
run_experiment.py — Unified v2 experiment runner.

Runs any experiment condition via YAML config:
  python project/scripts/v2/run_experiment.py --config project/configs/v2/E3_unsw.yaml [--pilot 10] [--subsample 200]
"""

import argparse
import csv
import json
import logging
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2-runner")

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "unsw": {
        "base": "data/processed/unsw_nb15/zeroday_split",
        "val_known": "val_known.csv",
        "val_zeroday": "val_zeroday.csv",
        "text_col": "text",
        "label_col": "label_name",
        "zeroday_classes": ["Shellcode", "Worms"],
    },
    "cic_full": {
        "base": "data/processed/cic_ids2017_full/zeroday_split",
        "val_known": "val_known.csv",
        "val_zeroday": "val_zeroday.csv",
        "text_col": "text",
        "label_col": "label_name",
        "zeroday_classes": ["Bot", "WebAttack", "Infiltration"],
    },
}

SEED = 42


def resolve_path(rel_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / rel_path.replace("project/", "")


def load_data(dataset: str, subsample: int | None = None, pilot: int | None = None):
    """Load val_known + val_zeroday, apply sampling."""
    ds = DATASET_PATHS[dataset]
    base = PROJECT_ROOT / ds["base"]
    val_known = pd.read_csv(base / ds["val_known"])
    val_zeroday = pd.read_csv(base / ds["val_zeroday"])
    log.info("Loaded: val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))

    if pilot:
        known_n = min(pilot, len(val_known))
        zd_n = min(max(pilot // 5, 2), len(val_zeroday))
        val_known = val_known.sample(n=known_n, random_state=SEED)
        val_zeroday = val_zeroday.sample(n=zd_n, random_state=SEED)
        log.info("PILOT: val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))
    elif subsample:
        zd_n = min(int(subsample * 0.1), len(val_zeroday))
        known_n = min(subsample - zd_n, len(val_known))
        val_known = val_known.sample(n=known_n, random_state=SEED)
        val_zeroday = val_zeroday.sample(n=zd_n, random_state=SEED)
        log.info("SUBSAMPLE: val_known=%d, val_zeroday=%d", len(val_known), len(val_zeroday))

    combined = pd.concat([val_known, val_zeroday], ignore_index=True)
    combined["_is_zeroday"] = [False] * len(val_known) + [True] * len(val_zeroday)

    return combined, ds["text_col"], ds["label_col"], ds["zeroday_classes"]


# ---------------------------------------------------------------------------
# Baseline ML runner
# ---------------------------------------------------------------------------
def run_baseline(cfg, combined, text_col, label_col, log_path):
    """Run ML-only baseline using BetaAgentML."""
    from project.src.agents.beta_agent_ml import BetaAgentML
    from project.src.v2.harness.audit import AuditRecordV2, AuditLoggerV2

    ml_path = resolve_path(cfg["data"]["ml_model_path"])
    log.info("Loading ML model from %s", ml_path)
    ml_agent = BetaAgentML.load(ml_path)

    texts = combined[text_col].tolist()
    labels = combined[label_col].tolist()
    is_zeroday = combined["_is_zeroday"].tolist()

    log.info("Running ML baseline on %d samples...", len(texts))
    t0 = time.time()
    verdicts = ml_agent.analyze_batch(texts)
    elapsed = time.time() - t0
    log.info("ML baseline done in %.1fs (%.1f samples/s)", elapsed, len(texts) / elapsed)

    with AuditLoggerV2(log_path) as alogger:
        for i, (v, gt, zd) in enumerate(zip(verdicts, labels, is_zeroday)):
            audit = AuditRecordV2(
                sample_index=i,
                traffic_text=texts[i][:500],
                is_zeroday=zd,
                model="RandomForest",
                adapter="",
                harness_enabled=False,
                temperature=0.0,
                max_steps=0,
            )
            audit.verdict = v.verdict
            audit.attack_type = v.attack_type
            audit.confidence = v.confidence
            audit.reasoning = v.reasoning
            audit.termination = "direct_verdict"
            audit.total_latency_ms = v.latency_ms
            audit.compute_efficiency()
            audit.evaluate(gt)
            alogger.log(audit)

        return alogger.records


# ---------------------------------------------------------------------------
# LLM experiment runner (E1-E4 + ablations)
# ---------------------------------------------------------------------------
def run_llm_experiment(cfg, combined, text_col, label_col, log_path, resume_from=0, seed=None):
    """Run LLM-based experiment using SecHarness."""
    from project.src.v2.llm_engine import LLMEngine
    from project.src.v2.agent_loop import SecHarness, agent_loop
    from project.src.v2.harness.audit import AuditLoggerV2

    mcfg = cfg["model"]
    hcfg = cfg["harness"]

    # Build enabled_tools list from config
    enabled_tools = None
    if hcfg["enabled"]:
        tools_cfg = hcfg.get("tools", {})
        enabled_tools = [t for t, on in tools_cfg.items() if on]
        # Action space. classify is constitutive (without it the agent cannot
        # produce a verdict), so it is always present; the optional actions are
        # ablatable via harness.actions to isolate the action-space component.
        actions_cfg = hcfg.get("actions", {})
        enabled_tools.append("classify")
        enabled_tools.extend(a for a in ("escalate", "log_decision")
                             if actions_cfg.get(a, True))

    # Resolve paths
    adapter_path = None
    if mcfg.get("adapter"):
        adapter_path = str(resolve_path(mcfg["adapter"]))

    ml_model_path = None
    if cfg["data"].get("ml_model_path"):
        ml_model_path = str(resolve_path(cfg["data"]["ml_model_path"]))

    knowledge_dir = str(PROJECT_ROOT / "data" / "knowledge")
    signatures_dir = str(PROJECT_ROOT / "data" / "knowledge" / "signatures")

    # Initialize LLM
    log.info("Loading LLM: base=%s, adapter=%s", mcfg["base"], adapter_path)
    engine = LLMEngine(
        base_model=mcfg["base"],
        adapter_path=adapter_path,
        load_in_4bit=mcfg.get("load_in_4bit", False),
        seed=seed,
    )

    # Initialize audit logger
    alogger = AuditLoggerV2(log_path)

    # Initialize SecHarness
    check_anomaly_degraded = hcfg.get("tools", {}).get("check_anomaly_degraded", False)
    rf_context = hcfg.get("rf_context", False)
    harness = SecHarness(
        llm=engine,
        ml_model_path=ml_model_path if (hcfg["enabled"] or rf_context) else None,
        signatures_dir=signatures_dir,
        knowledge_dir=knowledge_dir if hcfg["enabled"] else None,
        harness_enabled=hcfg["enabled"],
        enabled_tools=enabled_tools,
        permissions_enabled=hcfg.get("permissions", {}).get("enabled", True),
        max_steps=hcfg.get("max_steps", 5),
        max_new_tokens=mcfg.get("max_new_tokens", 256),
        temperature=mcfg.get("temperature", 0.1),
        confidence_threshold=hcfg.get("permissions", {}).get("confidence_threshold", 0.3),
        audit_logger=alogger,
        check_anomaly_degraded=check_anomaly_degraded,
        rf_context=rf_context,
    )

    texts = combined[text_col].tolist()
    labels = combined[label_col].tolist()
    is_zeroday = combined["_is_zeroday"].tolist()
    total = len(texts)

    # Estimate time
    if hcfg["enabled"]:
        est_per_sample = 4.0  # seconds, multi-step
    else:
        est_per_sample = 1.0  # seconds, single-shot
    remaining = total - resume_from
    est_total = remaining * est_per_sample
    log.info(
        "=== Experiment: %s | Dataset: %s | Samples: %d | Harness: %s | Est: %.0fs ===",
        cfg["experiment"]["id"], cfg["data"]["dataset"],
        remaining, hcfg["enabled"], est_total,
    )

    t0 = time.time()
    try:
        for i in range(resume_from, total):
            result = agent_loop(
                traffic_text=texts[i],
                harness=harness,
                sample_index=i,
                ground_truth=labels[i],
                is_zeroday=is_zeroday[i],
            )

            done = i - resume_from + 1
            if done % 10 == 0 or done == remaining:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (remaining - done) / rate if rate > 0 else 0
                log.info(
                    "Progress: %d/%d (%.1f%%) | %.2f s/sample | ETA %.0fs",
                    done, remaining, done / remaining * 100,
                    1 / rate if rate > 0 else 0, eta,
                )
    except KeyboardInterrupt:
        log.warning("Interrupted at sample %d. Use --resume to continue.", i)
    finally:
        alogger.close()

    return alogger.records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(records, zeroday_classes, exp_id, results_dir):
    """Compute and export all metrics from audit records."""
    if not records:
        log.warning("No records to compute metrics from.")
        return

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Binary metrics
    tp = fp = fn = tn = 0
    zd_total = zd_detected = 0
    total_latency = 0.0
    total_tokens = 0
    tool_counts = {}
    chain_lengths = []
    self_corrected = 0
    escalated = 0
    termination_counts = {}

    for r in records:
        if isinstance(r, dict):
            gt = r.get("evaluation", {}).get("binary_gt", "")
            pred = r.get("evaluation", {}).get("binary_pred", "")
            is_zd = r.get("input", {}).get("is_zeroday", False)
            latency = r.get("efficiency", {}).get("total_latency_ms", 0)
            tokens = r.get("efficiency", {}).get("total_tokens", 0)
            tools_used = r.get("result", {}).get("tools_used", [])
            n_tools = len(r.get("tool_chain", []))
            sc = r.get("result", {}).get("self_corrected", False)
            esc = r.get("result", {}).get("escalated", False)
            term = r.get("result", {}).get("termination", "")
        else:
            gt = r.binary_gt
            pred = r.binary_pred
            is_zd = r.is_zeroday
            latency = r.total_latency_ms
            tokens = r.total_tokens
            tools_used = r.tools_used
            n_tools = r.total_steps
            sc = r.self_corrected
            esc = r.escalated
            term = r.termination

        if gt == "attack" and pred == "attack":
            tp += 1
        elif gt == "benign" and pred == "attack":
            fp += 1
        elif gt == "attack" and pred == "benign":
            fn += 1
        else:
            tn += 1

        if is_zd:
            zd_total += 1
            if pred == "attack":
                zd_detected += 1

        total_latency += latency
        total_tokens += tokens
        chain_lengths.append(n_tools)
        if sc:
            self_corrected += 1
        if esc:
            escalated += 1
        termination_counts[term] = termination_counts.get(term, 0) + 1

        for tool in tools_used:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    n = len(records)
    acc = (tp + tn) / n if n > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    zd_rate = zd_detected / zd_total if zd_total > 0 else 0
    avg_latency = total_latency / n if n > 0 else 0
    avg_tokens = total_tokens / n if n > 0 else 0
    avg_chain = sum(chain_lengths) / n if n > 0 else 0
    sc_rate = self_corrected / n if n > 0 else 0
    esc_rate = escalated / n if n > 0 else 0

    # Tool usage rate (fraction of samples that used each tool)
    tool_usage_rates = {t: c / n for t, c in tool_counts.items()} if n > 0 else {}

    metrics = {
        "experiment_id": exp_id,
        "n_samples": n,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "zeroday_total": zd_total,
        "zeroday_detected": zd_detected,
        "zeroday_detection_rate": round(zd_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_tokens": round(avg_tokens, 1),
        "avg_chain_length": round(avg_chain, 2),
        "self_correction_rate": round(sc_rate, 4),
        "escalation_rate": round(esc_rate, 4),
        "tool_usage": tool_usage_rates,
        "termination_counts": termination_counts,
    }

    # Print summary
    log.info("=== %s Results ===", exp_id)
    log.info("Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  FPR=%.4f", acc, prec, rec, f1, fpr)
    log.info("ZD: %d/%d = %.4f", zd_detected, zd_total, zd_rate)
    log.info("Latency=%.1fms/sample  Tokens=%.0f/sample  Chain=%.1f steps", avg_latency, avg_tokens, avg_chain)
    log.info("Self-correction=%.4f  Escalation=%.4f", sc_rate, esc_rate)
    log.info("Termination: %s", termination_counts)

    # Save CSV
    csv_path = results_dir / f"{exp_id}_results.csv"
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for t, rate in tool_usage_rates.items():
        flat[f"tool_{t}_rate"] = round(rate, 4)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)
    log.info("Results → %s", csv_path)

    # Save JSON (full)
    json_path = results_dir / f"{exp_id}_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Full results → %s", json_path)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def check_no_overwrite(paths, resume: bool = False, allow: bool = False) -> None:
    """Refuse to start when any output of this run already exists.

    AuditLoggerV2 opens the log in append mode and compute_metrics truncates the
    results files, so an accidental re-run either appended a second run onto the
    first (E4_unsw_sub200: 400 records) or replaced the results.json a pilot run
    shares with the full run (tau_13B_noTools_unsw, 2026-08-18). Either way the
    earlier run is no longer recoverable. Exit code 3 keeps orchestrators honest.

    --resume: only bypasses the guard when the audit log itself exists (it is
    continued); stale results/meta of the interrupted session are moved aside and no
    results are written for the resumed session (see main). --allow-overwrite: existing
    outputs are moved aside as .bak_<stamp> first; nothing is ever deleted.
    """
    paths = [Path(p) for p in paths]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return
    if allow:
        # Never delete: rename aside with a timestamp so nothing is lost if this run dies.
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for p in existing:
            bak = _unused_name(p.with_name(f"{p.name}.bak_{stamp}"))
            log.warning("--allow-overwrite: moving %s (%d bytes) -> %s", p, p.stat().st_size, bak.name)
            p.rename(bak)
        return
    if resume and paths[0].exists():
        # Continuing an existing audit log (paths[0]). Stale results/meta from the
        # interrupted session are moved aside so nobody reads partial numbers.
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for p in existing[1:]:
            bak = _unused_name(p.with_name(f"{p.name}.partial_before_resume_{stamp}"))
            log.warning("--resume: moving stale %s -> %s", p.name, bak.name)
            p.rename(bak)
        return
    for p in existing:
        log.error("Refusing to write %s: it already exists (%d bytes).", p, p.stat().st_size)
    log.error("Pass --run-tag/--seed for a new run, --resume to continue an EXISTING audit log, "
              "or --allow-overwrite to move these outputs aside (.bak_<stamp>) first.")
    sys.exit(3)


def _unused_name(path: Path) -> Path:
    """Return `path`, or `path` with a numeric suffix if it already exists (never clobber)."""
    cand, k = path, 1
    while cand.exists():
        cand = path.with_name(f"{path.name}_{k}")
        k += 1
    return cand


def _sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _engine_version(base_model):
    """vLLM exposes GET /version on its OpenAI-compatible server; Ollama exposes /api/version.
    Recorded so two runs on different stacks can be told apart later. Best effort."""
    if not (isinstance(base_model, str) and base_model.startswith("api://")):
        return None
    from urllib.request import urlopen
    host = base_model[len("api://"):].split("/", 1)[0]
    for path in ("/version", "/api/version"):
        try:
            with urlopen(f"http://{host}{path}", timeout=3) as r:
                return {"endpoint": path, "body": r.read(500).decode("utf-8", "replace")}
        except Exception:
            continue
    return None


def _gpu_name():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out or None
    except Exception:
        return None


def _git_commit():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="v2 unified experiment runner")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--pilot", type=int, default=0, help="Run N samples only (pilot test)")
    ap.add_argument("--subsample", type=int, default=0, help="Random subsample of N")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    ap.add_argument("--seed", type=int, default=None,
                    help="LLM sampling seed for this run (vLLM/Ollama per-request seed, torch seed for HF); "
                         "appended to every output name as _seed<N>")
    ap.add_argument("--run-tag", default="",
                    help="Free tag appended to every output name (e.g. 20260910T0130); use it so a re-run "
                         "of the same config never lands on the same file")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="Append to / overwrite existing outputs of this name (default: refuse)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment"]["id"]
    dataset = cfg["data"]["dataset"]
    is_baseline = cfg["model"].get("base") is None

    pilot = args.pilot if args.pilot > 0 else None
    subsample = args.subsample if args.subsample > 0 else cfg["data"].get("subsample")

    # Output paths (resolved and guarded BEFORE the dataset is loaded)
    log_dir = resolve_path(cfg["output"]["log_dir"])
    results_dir = resolve_path(cfg["output"]["results_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_pilot{pilot}" if pilot else (f"_sub{subsample}" if subsample else "")
    # Run identity. 2026-08-18 lesson: outputs were keyed by config prefix only, so a
    # re-run of tau_13B_noTools_unsw replaced the earlier run's log and results.json.
    # Seed and tag now become part of every output name, and an existing log refuses
    # to be touched unless --resume or --allow-overwrite is explicit.
    run_suffix = (f"_seed{args.seed}" if args.seed is not None else "") + (f"_{args.run_tag}" if args.run_tag else "")
    suffix += run_suffix
    log_path = log_dir / f"{exp_id}{suffix}_audit.jsonl"
    # Results files carry the seed/tag but not the _pilot/_sub half (the untagged
    # name must stay `{exp_id}_results.json` for run_tau_orchestrator.py), which is
    # exactly why a pilot run could truncate a full run's results: guard them too.
    results_id = exp_id + run_suffix
    meta_path = results_dir / f"{exp_id}{suffix}_meta.json"
    check_no_overwrite(
        [log_path, results_dir / f"{results_id}_results.json", results_dir / f"{results_id}_results.csv", meta_path],
        resume=args.resume, allow=args.allow_overwrite,
    )
    if args.resume and not log_path.exists():
        log.warning("--resume given but %s does not exist (different seed/tag?); starting a fresh run", log_path)

    # Load data
    combined, text_col, label_col, zeroday_classes = load_data(dataset, subsample, pilot)

    # Resume support
    resume_from = 0
    if args.resume and log_path.exists():
        from project.src.v2.harness.audit import AuditLoggerV2
        existing = AuditLoggerV2.load_from_file(log_path)
        resume_from = len(existing)
        log.info("Resuming from sample %d (found %d existing records)", resume_from, resume_from)

    # Run
    log.info("Experiment: %s | baseline=%s | pilot=%s | subsample=%s", exp_id, is_baseline, pilot, subsample)
    t_start = time.time()

    if is_baseline:
        records = run_baseline(cfg, combined, text_col, label_col, log_path)
    else:
        records = run_llm_experiment(cfg, combined, text_col, label_col, log_path, resume_from, seed=args.seed)

    elapsed = time.time() - t_start
    log.info("Total runtime: %.1fs for %d samples", elapsed, len(records))

    metrics_over = f"run:{len(records)}"
    if resume_from > 0:
        # The in-memory list holds only this session's records, and AuditRecordV2 has
        # no inverse of its nested to_dict(), so metrics over the whole log cannot be
        # rebuilt here. Pre-existing defect: a resumed run silently wrote tail-only
        # results.json. Now: no results files are written for a resumed run; the meta
        # sidecar says so and the full-log recompute path is named.
        from project.src.v2.harness.audit import AuditLoggerV2
        total = len(AuditLoggerV2.load_from_file(log_path))
        metrics_over = f"NOT_WRITTEN_resume:{len(records)}_of_{total}"
        log.error("Resume: this session produced %d of %d records in %s; results files are NOT written. "
                  "Recompute metrics over the full log with scripts/audit_20260728/recompute_all_metrics.py.",
                  len(records), total, log_path)

    # Compute metrics. Results files carry the seed/tag too (the base name stays
    # unchanged for untagged runs so the tau orchestrator keeps finding them).
    if resume_from == 0:
        compute_metrics(records, zeroday_classes, results_id, results_dir)

    # Save run metadata (provenance sidecar: enough to tell two runs apart later)
    meta = {
        "experiment_id": exp_id,
        "results_id": results_id,
        "audit_log": str(log_path),
        "config_path": str(args.config),
        "config_sha256": _sha256_file(args.config),
        "seed": args.seed,
        "run_tag": args.run_tag or None,
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "gpu": _gpu_name(),
        "engine_version": _engine_version(cfg["model"].get("base")),
        "metrics_over": metrics_over,
        "resume_from": resume_from,
        "model_base": cfg["model"].get("base"),
        "n_samples": len(records),
        "pilot": pilot,
        "subsample": subsample,
        "runtime_seconds": round(elapsed, 1),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if resume_from > 0:
        # The interrupted session's sidecar was moved aside by the guard; this one is
        # named per resume and never clobbers an earlier resume's sidecar.
        meta_path = _unused_name(meta_path.with_name(meta_path.name.replace("_meta.json", f"_meta_resume{resume_from}.json")))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata → %s", meta_path)


if __name__ == "__main__":
    main()
