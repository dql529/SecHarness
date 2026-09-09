#!/usr/bin/env python3
"""
run_E5_a800.py — Exp-D-cic driver for AutoDL A800 (vLLM backend).

Variant of run_E5.py that uses AlphaAgentHF instead of AlphaAgentHF.
Runs 3 seeds × N=1000 samples with pilot-first gate and resume support.

CONTRACT: load_config
  inputs: config_path:Path
  output: dict — validated config with all required keys
  preconditions: file exists and is valid YAML
  error modes: SystemExit(1) on missing required field or parse error

CONTRACT: preflight_checks
  inputs: cfg:dict, project_root:Path
  output: None
  preconditions: none
  error modes: SystemExit(1) if endpoint unreachable / model missing /
               Beta pkl unloadable / data files missing

CONTRACT: run_pilot
  inputs: df:pd.DataFrame, alpha:AlphaAgentHF, beta:BetaAgentML,
          consensus:ConsensusModule, cfg:dict, project_root:Path, force_skip:bool
  output: tuple[list[dict], int, int] — (records, parse_fail_count, disagree_count)
  preconditions: df has text_col and label_col columns
  error modes: SystemExit(2) if parse-fail rate > 20% (> 4/20)

CONTRACT: run_seed
  inputs: seed:int, df:pd.DataFrame, alpha:AlphaAgentHF, beta:BetaAgentML,
          consensus:ConsensusModule, cfg:dict,
          project_root:Path, force_resume:bool
  output: list[dict] — flat audit records (dicts)
  preconditions: preflight_checks passed, pilot passed
  error modes: logs errors, does not raise

CONTRACT: compute_seed_summary
  inputs: records:list[dict], seed:int
  output: dict — {seed, N, disagreement_rate, zeroday_disagreement_rate,
                  known_disagreement_rate, normal_disagreement_rate, chi2_stat, chi2_p}
  preconditions: records non-empty
  error modes: returns NaN for stats if insufficient data

CONTRACT: main
  inputs: CLI argv
  output: None; exit 0 on success
  error modes: exit 1 on config/preflight error; exit 2 on pilot parse-fail gate
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from scipy import stats as scipy_stats

# Resolve project root (run_E5_a800.py lives in project/scripts/v2_tdsc/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from project.src.agents.alpha_agent_hf import AlphaAgentHF
from project.src.agents.beta_agent_ml import BetaAgentML
from project.src.agents.consensus import ConsensusModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_E5_a800")

# Required top-level config keys
_REQUIRED_KEYS: tuple[str, ...] = (
    "experiment",
    "data",
    "agent_alpha_hf",
    "agent_beta_ml",
    "consensus",
    "sampling",
    "output",
)

# Ground-truth category classification
NORMAL_LABELS: frozenset[str] = frozenset({"normal", "benign", "BENIGN"})


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and validate experiment config from YAML.

    CONTRACT: see module docstring.
    """
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)
    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        log.error("Failed to parse YAML config: %s", exc)
        sys.exit(1)

    for key in _REQUIRED_KEYS:
        if key not in cfg:
            log.error("Missing required config key: %r", key)
            sys.exit(1)

    sampling = cfg["sampling"]
    if "seeds" not in sampling:
        log.error("Missing config key: sampling.seeds (list of ints)")
        sys.exit(1)
    if "subsample_n" not in sampling:
        log.error("Missing config key: sampling.subsample_n")
        sys.exit(1)
    if "pilot_n" not in sampling:
        log.error("Missing config key: sampling.pilot_n")
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight_checks(cfg: dict, project_root: Path) -> None:
    """Verify all dependencies are reachable before starting any inference.

    CONTRACT: see module docstring.
    """
    alpha_cfg = cfg["agent_alpha_hf"]

    # (a) LoRA adapter present (if specified)
    adapter_path_str = alpha_cfg.get("adapter_path")
    if adapter_path_str:
        adapter_dir = project_root / adapter_path_str
        cfg_file = adapter_dir / "adapter_config.json"
        if not cfg_file.exists():
            log.error("Pre-flight FAIL: LoRA adapter_config.json missing at %s", cfg_file)
            sys.exit(1)
        log.info("Pre-flight OK: LoRA adapter found at %s", adapter_dir)
    else:
        log.info("Pre-flight OK: zero-shot mode (no adapter)")

    # (c) Beta-ML model loadable
    beta_model_path = project_root / cfg["agent_beta_ml"]["model_path"]
    if not beta_model_path.exists():
        log.error("Pre-flight FAIL: Beta-ML model not found at %s", beta_model_path)
        sys.exit(1)
    try:
        import pickle
        with open(beta_model_path, "rb") as fh:
            pickle.load(fh)
        log.info("Pre-flight OK: Beta-ML model loadable at %s", beta_model_path)
    except Exception as exc:
        log.error("Pre-flight FAIL: cannot load Beta-ML model — %s", exc)
        sys.exit(1)

    # (d) Data files exist
    base_dir = project_root / cfg["data"]["base_dir"]
    for split in ("val_known", "val_zeroday"):
        fpath = base_dir / cfg["data"][split]
        if not fpath.exists():
            log.error("Pre-flight FAIL: data file missing: %s", fpath)
            sys.exit(1)
    log.info("Pre-flight OK: data files found in %s", base_dir)


# ---------------------------------------------------------------------------
# Ground-truth category helper
# ---------------------------------------------------------------------------

def classify_gt_category(
    label: str,
    zeroday_classes: list[str],
    normal_labels: frozenset[str] = NORMAL_LABELS,
) -> str:
    """Map a raw ground-truth label to one of {normal, known, zeroday}."""
    if label.lower() in {lbl.lower() for lbl in normal_labels}:
        return "normal"
    if label in zeroday_classes or label.lower() in {z.lower() for z in zeroday_classes}:
        return "zeroday"
    return "known"


# ---------------------------------------------------------------------------
# Per-sample audit record builder
# ---------------------------------------------------------------------------

def build_flat_record(
    idx: int,
    traffic_text: str,
    alpha_v: object,
    beta_v: object,
    cr: object,
    gt_label: str,
    zeroday_classes: list[str],
) -> dict:
    """Build a flat dict audit record matching run_E5.py schema."""
    from project.src.agents.base_agent import AgentVerdict
    from project.src.agents.consensus import ConsensusResult

    assert isinstance(alpha_v, AgentVerdict)
    assert isinstance(beta_v, AgentVerdict)
    assert isinstance(cr, ConsensusResult)

    gt_category = classify_gt_category(gt_label, zeroday_classes)
    disagree = cr.is_disagreement

    return {
        "sample_index": idx,
        "traffic_text": traffic_text[:200],
        "alpha_verdict": alpha_v.verdict,
        "alpha_confidence": alpha_v.confidence,
        "alpha_attack_type": alpha_v.attack_type,
        "alpha_reasoning": (alpha_v.reasoning or "")[:500],
        "beta_verdict": beta_v.verdict,
        "beta_confidence": beta_v.confidence,
        "consensus_type": cr.consensus_type,
        "consensus_verdict": cr.final_verdict,
        "disagreement": disagree,
        "ground_truth_label": gt_label,
        "ground_truth_category": gt_category,
        "latency_ms": alpha_v.latency_ms + beta_v.latency_ms + cr.latency_ms,
    }


# ---------------------------------------------------------------------------
# Pilot run
# ---------------------------------------------------------------------------

def run_pilot(
    df: pd.DataFrame,
    alpha: AlphaAgentHF,
    beta: BetaAgentML,
    consensus: ConsensusModule,
    cfg: dict,
    project_root: Path,
    force_skip: bool = False,
    force: bool = False,
) -> tuple[list[dict], int, int]:
    """Run N=pilot_n samples with seed=42 as a smoke-test gate.

    CONTRACT: see module docstring.
    Exit code 2 if parse-fail rate > 20% (> 4/20) — --force does NOT override.
    Exit code 2 if disagreement floor not met and force=False.
    --force overrides the disagreement-floor gate only.
    """
    if force_skip:
        log.info("Pilot skipped (--force-skip-pilot)")
        return [], 0, 0

    pilot_n: int = cfg["sampling"]["pilot_n"]
    text_col: str = cfg["data"]["text_col"]
    label_col: str = cfg["data"]["label_col"]
    zeroday_classes: list[str] = cfg["data"]["zeroday_classes"]

    pilot_df = df.sample(n=min(pilot_n, len(df)), random_state=42)
    texts = pilot_df[text_col].tolist()
    labels = pilot_df[label_col].tolist()

    log.info("Pilot: running %d samples (seed=42)", len(texts))
    beta_verdicts = beta.analyze_batch(texts)

    records: list[dict] = []
    parse_fail_count = 0
    disagree_count = 0

    for i, (text, gt) in enumerate(zip(texts, labels)):
        av = alpha.analyze(text)
        bv = beta_verdicts[i]
        cr = consensus.resolve(av, bv)

        is_parse_fail = av.confidence == 0.0 and av.reasoning.startswith("PARSE_FAIL:")
        if is_parse_fail:
            parse_fail_count += 1

        if cr.is_disagreement:
            disagree_count += 1

        records.append(build_flat_record(i, text, av, bv, cr, gt, zeroday_classes))

        if (i + 1) % 5 == 0 or (i + 1) == len(texts):
            log.info("Pilot %d/%d | parse_fail=%d disagree=%d", i + 1, len(texts), parse_fail_count, disagree_count)

    # Gate 1: parse-fail rate > 20%
    fail_rate = parse_fail_count / len(texts)
    if fail_rate > 0.20:
        log.error(
            "Pilot parse-fail rate too high (%d/%d = %.1f%%); "
            "vLLM endpoint or adapter misconfigured. Exiting.",
            parse_fail_count, len(texts), fail_rate * 100,
        )
        sys.exit(2)

    # Gate 2: disagreement floor — exit 2 unless force=True overrides.
    # Note: --force overrides this gate but NOT the parse-fail gate above.
    gt_categories = [classify_gt_category(lbl, zeroday_classes) for lbl in labels]
    has_mixed = len(set(gt_categories)) > 1
    if has_mixed and disagree_count < 2:
        if not force:
            log.error(
                "Pilot disagreement floor not met: only %d/%d disagreements "
                "with mixed ground truth. Beta-ML may be always matching Alpha. "
                "Use --force to override and proceed to multi-seed.",
                disagree_count, len(texts),
            )
            sys.exit(2)
        else:
            log.warning(
                "Pilot disagreement floor not met (%d/%d disagrees) — "
                "proceeding because --force was set.",
                disagree_count, len(texts),
            )

    log.info(
        "Pilot PASSED: parse_fail=%d/%d (%.1f%%), disagree=%d/%d",
        parse_fail_count, len(texts), fail_rate * 100,
        disagree_count, len(texts),
    )
    return records, parse_fail_count, disagree_count


# ---------------------------------------------------------------------------
# Per-seed full run
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    df: pd.DataFrame,
    alpha: AlphaAgentHF,
    beta: BetaAgentML,
    consensus: ConsensusModule,
    logs_dir: Path,
    cfg: dict,
    force_resume: bool = False,
) -> list[dict]:
    """Run subsample_n samples for one seed with resume support.

    CONTRACT: see module docstring.
    """
    prefix: str = cfg["output"]["prefix"]
    subsample_n: int = cfg["sampling"]["subsample_n"]
    text_col: str = cfg["data"]["text_col"]
    label_col: str = cfg["data"]["label_col"]
    zeroday_classes: list[str] = cfg["data"]["zeroday_classes"]

    audit_path = logs_dir / f"audit_{prefix}_seed{seed}.jsonl"

    # Resume logic — determine start index and file mode
    existing_lines = 0
    file_mode = "w"
    start_idx = 0

    if audit_path.exists():
        with open(audit_path, encoding="utf-8") as fh:
            existing_lines = sum(1 for line in fh if line.strip())

        if existing_lines == subsample_n:
            log.info("[seed=%d] existing=%d, action=skip (already complete)", seed, existing_lines)
            # Load and return existing records
            records: list[dict] = []
            with open(audit_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif 0 < existing_lines < subsample_n:
            log.info("[seed=%d] existing=%d, action=resume from index %d", seed, existing_lines, existing_lines)
            start_idx = existing_lines
            file_mode = "a"
        else:
            log.info("[seed=%d] existing=%d, action=start (empty/missing)", seed, existing_lines)
            file_mode = "w"
            start_idx = 0
    else:
        log.info("[seed=%d] existing=0, action=start (no file)", seed)

    # NOTE: resume correctness assumes df row order/count unchanged between runs.
    # If CSVs change between a partial run and resume, audit log will silently desync.
    # Sample the dataframe
    sampled_df = df.sample(n=min(subsample_n, len(df)), random_state=seed).reset_index(drop=True)
    texts = sampled_df[text_col].tolist()
    labels = sampled_df[label_col].tolist()

    # Slice from resume point
    texts_to_run = texts[start_idx:]
    labels_to_run = labels[start_idx:]

    log.info("[seed=%d] running %d samples (from index %d)", seed, len(texts_to_run), start_idx)

    # Beta batch inference
    beta_verdicts = beta.analyze_batch(texts_to_run)

    records_new: list[dict] = []
    disagree_count = 0
    t0 = time.time()

    with open(audit_path, file_mode, encoding="utf-8") as fout:
        for i, (text, gt) in enumerate(zip(texts_to_run, labels_to_run)):
            global_idx = start_idx + i
            av = alpha.analyze(text)
            bv = beta_verdicts[i]
            cr = consensus.resolve(av, bv)

            rec = build_flat_record(global_idx, text, av, bv, cr, gt, zeroday_classes)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            records_new.append(rec)

            if cr.is_disagreement:
                disagree_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(texts_to_run):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(texts_to_run) - i - 1) / rate if rate > 0 else 0.0
                log.info(
                    "[seed=%d] %d/%d (%.1f/min, ETA %.0fs) [disagree=%d]",
                    seed, global_idx + 1, subsample_n, rate * 60, eta, disagree_count,
                )

    # Load all records (including previously completed ones) for summary
    all_records: list[dict] = []
    with open(audit_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    log.info("[seed=%d] complete: %d records written to %s", seed, len(all_records), audit_path)
    return all_records


# ---------------------------------------------------------------------------
# Per-seed summary statistics
# ---------------------------------------------------------------------------

def compute_seed_summary(records: list[dict], seed: int) -> dict:
    """Compute disagreement rates and chi2 for one seed.

    CONTRACT: see module docstring.
    """
    n = len(records)
    if n == 0:
        return {
            "seed": seed, "N": 0,
            "disagreement_rate": float("nan"),
            "zeroday_disagreement_rate": float("nan"),
            "known_disagreement_rate": float("nan"),
            "normal_disagreement_rate": float("nan"),
            "chi2_stat": float("nan"),
            "chi2_p": float("nan"),
        }

    total_disagree = sum(1 for r in records if r["disagreement"])
    disagree_rate = total_disagree / n

    by_cat: dict[str, list[int]] = {"zeroday": [], "known": [], "normal": []}
    for r in records:
        cat = r.get("ground_truth_category", "known")
        if cat in by_cat:
            by_cat[cat].append(1 if r["disagreement"] else 0)

    def cat_rate(cat: str) -> float:
        vals = by_cat[cat]
        return sum(vals) / len(vals) if vals else float("nan")

    zd_rate = cat_rate("zeroday")
    kn_rate = cat_rate("known")
    nm_rate = cat_rate("normal")

    # Chi2 contingency: disagreement × category (zeroday vs not-zeroday)
    chi2_stat: float = float("nan")
    chi2_p: float = float("nan")
    zd_disagree = sum(1 for r in records if r["disagreement"] and r.get("ground_truth_category") == "zeroday")
    zd_agree = sum(1 for r in records if not r["disagreement"] and r.get("ground_truth_category") == "zeroday")
    nzd_disagree = sum(1 for r in records if r["disagreement"] and r.get("ground_truth_category") != "zeroday")
    nzd_agree = sum(1 for r in records if not r["disagreement"] and r.get("ground_truth_category") != "zeroday")

    contingency = [[zd_disagree, zd_agree], [nzd_disagree, nzd_agree]]
    if all(cell >= 0 for row in contingency for cell in row) and (zd_disagree + zd_agree) > 0 and (nzd_disagree + nzd_agree) > 0:
        try:
            chi2_res = scipy_stats.chi2_contingency(contingency, correction=False)
            chi2_stat = float(chi2_res.statistic)
            chi2_p = float(chi2_res.pvalue)
        except Exception as exc:
            log.warning("chi2_contingency failed for seed %d: %s", seed, exc)

    return {
        "seed": seed,
        "N": n,
        "disagreement_rate": disagree_rate,
        "zeroday_disagreement_rate": zd_rate,
        "known_disagreement_rate": kn_rate,
        "normal_disagreement_rate": nm_rate,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="E5-A800: Dual-Agent pipeline (AlphaVLLM + BetaML) on A800"
    )
    ap.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to experiment YAML config (e.g. project/configs/v2_tdsc/E5_cic_a800.yaml)",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds from config (comma-separated, e.g. 42,123,456)",
    )
    ap.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run pilot (N=pilot_n) only, then exit",
    )
    ap.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume from existing audit log (do not re-run completed seeds)",
    )
    ap.add_argument(
        "--force-skip-pilot",
        action="store_true",
        help="Skip pilot gate (for re-runs where pilot already passed)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Continue past disagreement floor WARNING in pilot (does not override parse-fail gate)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    preflight_checks(cfg, PROJECT_ROOT)

    # Paths
    logs_dir = PROJECT_ROOT / cfg["output"]["logs_dir"]
    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix: str = cfg["output"]["prefix"]

    base_dir = PROJECT_ROOT / cfg["data"]["base_dir"]
    text_col: str = cfg["data"]["text_col"]
    label_col: str = cfg["data"]["label_col"]

    # Load data
    val_known = pd.read_csv(base_dir / cfg["data"]["val_known"])
    val_zeroday = pd.read_csv(base_dir / cfg["data"]["val_zeroday"])
    df_all = pd.concat([val_known, val_zeroday], ignore_index=True)
    log.info("Loaded val_known=%d, val_zeroday=%d, total=%d", len(val_known), len(val_zeroday), len(df_all))

    # Instantiate agents
    alpha_cfg = cfg["agent_alpha_hf"]
    adapter_path_str = alpha_cfg.get("adapter_path")
    adapter_abs = str(PROJECT_ROOT / adapter_path_str) if adapter_path_str else None
    alpha = AlphaAgentHF(
        base_model=alpha_cfg.get("base_model", "unsloth/Llama-3.2-3B-Instruct"),
        adapter_path=adapter_abs,
        load_in_4bit=bool(alpha_cfg.get("load_in_4bit", False)),
        temperature=float(alpha_cfg.get("temperature", 0.1)),
        max_new_tokens=int(alpha_cfg.get("max_new_tokens", 150)),
        device=str(alpha_cfg.get("device", "auto")),
    )

    beta_model_path = PROJECT_ROOT / cfg["agent_beta_ml"]["model_path"]
    beta = BetaAgentML.load(str(beta_model_path))

    cons_cfg = cfg["consensus"]
    consensus = ConsensusModule(
        alpha_weight=float(cons_cfg["alpha_weight"]),
        beta_weight=float(cons_cfg["beta_weight"]),
        disagreement_strategy=cons_cfg["disagreement_strategy"],
        confidence_threshold=float(cons_cfg["confidence_threshold"]),
    )

    # Determine seeds
    if args.seeds:
        seeds: list[int] = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = list(cfg["sampling"]["seeds"])

    # --- Pilot ---
    # Gate 1 (parse-fail): always enforced — sys.exit(2) on failure.
    # Gate 2 (disagreement floor): enforced unless --force is set — sys.exit(2).
    pilot_records, pilot_parse_fails, pilot_disagrees = run_pilot(
        df_all, alpha, beta, consensus, cfg, PROJECT_ROOT,
        force_skip=args.force_skip_pilot,
        force=args.force,
    )

    # Save pilot audit log
    if pilot_records and not args.force_skip_pilot:
        pilot_path = logs_dir / f"audit_{prefix}_pilot.jsonl"
        with open(pilot_path, "w", encoding="utf-8") as fout:
            for rec in pilot_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Pilot audit log written: %s", pilot_path)

    if args.pilot_only:
        log.info("--pilot-only flag set. Pilot complete. Exiting.")
        sys.exit(0)

    # --- Multi-seed run ---
    # Seed 42 was already exercised by pilot; still run full subsample_n for seed 42
    summary_rows: list[dict] = []
    for seed in seeds:
        log.info("Starting full run for seed=%d", seed)
        # Use AuditLogger for bookkeeping (logging to JSONL is handled inside run_seed)
        seed_records = run_seed(
            seed=seed,
            df=df_all,
            alpha=alpha,
            beta=beta,
            consensus=consensus,
            logs_dir=logs_dir,
            cfg=cfg,
            force_resume=args.force_resume,
        )
        summary = compute_seed_summary(seed_records, seed)
        summary_rows.append(summary)
        log.info(
            "Seed %d summary: N=%d disagree_rate=%.3f zeroday_disagree=%.3f chi2_p=%.4f",
            seed,
            summary["N"],
            summary["disagreement_rate"] if not math.isnan(summary["disagreement_rate"]) else -1,
            summary["zeroday_disagreement_rate"] if not math.isnan(summary["zeroday_disagreement_rate"]) else -1,
            summary["chi2_p"] if not math.isnan(summary["chi2_p"]) else -1,
        )

    # Write summary CSV
    if summary_rows:
        summary_path = results_dir / f"{prefix}_summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        log.info("Summary CSV written: %s", summary_path)

    log.info("All seeds complete. Run analyze_disagreement_cic.py to generate T4 tables.")


if __name__ == "__main__":
    main()
