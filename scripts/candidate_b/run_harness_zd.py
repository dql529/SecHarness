"""
run_harness_zd.py — Candidate B, Stage B: LLM-harness zero-day rate per fold.

Direct (non-transitive) evidence that the pre-registered kNN-coverage predicts
the DEPLOYED SecHarness LLM agent's zero-day blind spots, not just the RF tool's.

For each leave-one-attack-out fold (CIC 11 + UNSW 9):
  1. train a BetaAgentML RF on the known pool (benign + other attacks), save pkl
  2. build the full SecHarness (LLM via Ollama + check_anomaly[this RF] +
     lookup_signature + query_history + load_knowledge), harness_enabled
  3. run the agent loop on N held-out-class samples -> zero-day detection rate
     = fraction the agent finally classifies as "attack"

Correlate harness ZD-rate with the frozen kNN-coverage (coverage_variants.csv).

Reuses Stage A loaders + the preprocess serialization utilities + BetaAgentML +
the existing SecHarness agent loop (nothing re-implemented).

Usage:
  python scripts/candidate_b/run_harness_zd.py --n 60 \
    --model "api://localhost:11434/llama3.2:3b-instruct-q4_0" [--datasets cic unsw]
  python scripts/candidate_b/run_harness_zd.py --smoke   # 1 class x 2 samples
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.candidate_b.run_coverage_zd_stageA import (  # noqa: E402
    ATTACK_CAP,
    BENIGN_CAP,
    OUT_DIR,
    SEED,
    _cap,
    load_cic,
    load_unsw,
)
from src.agents.beta_agent_ml import BetaAgentML  # noqa: E402
from src.data.preprocess import apply_bins, build_kv_text, fit_bins  # noqa: E402
from src.v2.agent_loop import AgentResult, SecHarness, agent_loop  # noqa: E402
from src.v2.llm_engine import LLMEngine  # noqa: E402

KNOWLEDGE_DIR = str(PROJECT_ROOT / "data" / "knowledge")
SIGNATURES_DIR = str(PROJECT_ROOT / "data" / "knowledge" / "signatures")
HARNESS_TOOLS = ["check_anomaly", "lookup_signature", "query_history",
                 "load_knowledge", "classify", "escalate", "log_decision"]


def serialize(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
              bin_specs: Dict) -> pd.Series:
    work = df.copy()
    for col in num_cols:
        bins, labels = bin_specs[col]
        work[f"{col}_cat"] = apply_bins(work[col], bins, labels)
    return work.apply(lambda r: build_kv_text(r, cat_cols, num_cols), axis=1)


def train_fold_rf(known: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
                  pkl_path: Path) -> Dict:
    """Fit bins on known numeric, serialize, train BetaAgentML, save pkl."""
    bin_specs = {col: fit_bins(known[col]) for col in num_cols}
    kdf = known.copy()
    kdf["text"] = serialize(kdf, num_cols, cat_cols, bin_specs)
    kdf["label_name"] = kdf["cls"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as tf:
        kdf[["text", "label_name"]].to_csv(tf.name, index=False)
        agent = BetaAgentML(classifier_type="random_forest")
        agent.train(tf.name, text_col="text", label_col="label_name")
    agent.save(pkl_path)
    return bin_specs


def harness_zd_rate(model: str, pkl_path: Path, held_texts: List[str],
                    held_class: str, workers: int = 16) -> Dict:
    """Run the full SecHarness agent loop on held-out samples; return ZD rate.

    Samples are independent, so they run concurrently over a shared (read-only
    after init) harness to exploit the server's request batching. Each sample is
    isolated: a harness exception is caught, counted, and treated as a miss.
    """
    engine = LLMEngine(base_model=model, max_input_length=2048)
    harness = SecHarness(
        llm=engine,
        ml_model_path=str(pkl_path),
        signatures_dir=SIGNATURES_DIR,
        knowledge_dir=KNOWLEDGE_DIR,
        harness_enabled=True,
        enabled_tools=HARNESS_TOOLS,
        permissions_enabled=True,
        max_steps=5,
    )

    def one(args):
        i, text = args
        try:
            return agent_loop(text, harness, sample_index=i,
                              ground_truth=held_class, is_zeroday=True)
        except Exception as e:  # noqa: BLE001 — isolate, count, continue
            return AgentResult(verdict="benign", termination="error",
                               reasoning=f"[ERROR] {type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, enumerate(held_texts)))
    errors = sum(1 for r in results if r.termination == "error")
    detected = sum(1 for r in results if r.verdict == "attack")
    parse_fail = sum(1 for r in results if r.termination == "parse_fail")
    max_steps = sum(1 for r in results if r.termination == "max_steps")
    valid = len(results) - parse_fail - errors
    return {"zd_total": len(results), "zd_detected": detected,
            "harness_zd_rate": detected / len(results) if results else 0.0,
            "harness_zd_rate_excl_parse": detected / valid if valid else 0.0,
            "parse_fail": parse_fail, "max_steps": max_steps, "errors": errors}


def run(model: str, n_eval: int, datasets: List[str], smoke: bool,
        workers: int, ckpt: Path) -> List[Dict]:
    loaders = {"cic": ("CIC-IDS2017", load_cic), "unsw": ("UNSW-NB15", load_unsw)}
    pkldir = OUT_DIR / "rf_pkls"
    pkldir.mkdir(parents=True, exist_ok=True)

    # Resume: load already-finished (dataset, class) rows from checkpoint
    rows: List[Dict] = []
    done = set()
    if ckpt.exists() and not smoke:
        prev = pd.read_csv(ckpt)
        rows = prev.to_dict("records")
        done = {(r["dataset"], r["held_class"]) for r in rows}
        print(f"[resume] {len(done)} classes already done in {ckpt.name}", flush=True)

    for key in datasets:
        name, loader = loaders[key]
        df, num_cols, cat_cols, benign, eligible = loader()
        benign_all = df[df["is_benign"]]
        classes = eligible[:1] if smoke else eligible
        n = 2 if smoke else n_eval
        for c in classes:
            if (name, c) in done:
                continue
            held = _cap(df[df["cls"] == c], n, SEED)
            if len(held) < 2:
                continue
            known = pd.concat(
                [_cap(benign_all, BENIGN_CAP, SEED)] +
                [_cap(df[df["cls"] == a], ATTACK_CAP, SEED) for a in eligible if a != c],
                ignore_index=True)
            pkl = pkldir / f"{key}_{c.replace(' ', '_').replace('/', '_')}.pkl"
            bin_specs = train_fold_rf(known, num_cols, cat_cols, pkl)
            held = held.copy()
            held["text"] = serialize(held, num_cols, cat_cols, bin_specs)
            zd = harness_zd_rate(model, pkl, held["text"].tolist(), c, workers=workers)
            row = {"dataset": name, "held_class": c, **zd}
            rows.append(row)
            print(f"  {name} {c:18s} harness_zd={zd['harness_zd_rate']:.3f} "
                  f"(excl_pf={zd['harness_zd_rate_excl_parse']:.3f}, n={zd['zd_total']}, "
                  f"parse_fail={zd['parse_fail']}, max_steps={zd['max_steps']}, "
                  f"errors={zd['errors']})", flush=True)
            if not smoke:  # checkpoint after every class
                pd.DataFrame(rows).to_csv(ckpt, index=False)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="api://localhost:11434/llama3.2:3b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="3b", help="suffix for output files")
    ap.add_argument("--datasets", nargs="+", default=["cic", "unsw"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ckpt = OUT_DIR / f"harness_zd_{args.tag}_partial.csv"
    print(f"=== Stage B harness ZD (model={args.model}, n={args.n}, "
          f"workers={args.workers}, tag={args.tag}, datasets={args.datasets}, "
          f"smoke={args.smoke}) ===", flush=True)
    rows = run(args.model, args.n, args.datasets, args.smoke, args.workers, ckpt)
    out = pd.DataFrame(rows)
    if args.smoke:
        print("smoke rows:\n", out.to_string())
        return

    # merge frozen kNN coverage, correlate
    cov = pd.read_csv(OUT_DIR / "coverage_variants.csv")
    covmap = {(r.dataset, r.held_class): r.cov_knn_attack for r in cov.itertuples()}
    out["cov_knn_attack"] = [covmap.get((r.dataset, r.held_class), np.nan)
                             for r in out.itertuples()]
    n_before = len(out)
    out = out.dropna(subset=["cov_knn_attack"])
    if len(out) < n_before:
        print(f"[warn] dropped {n_before - len(out)} rows lacking coverage", flush=True)
    out.to_csv(OUT_DIR / f"harness_zd_{args.tag}.csv", index=False)

    summary: Dict = {"model": args.model, "tag": args.tag, "n_eval": args.n}
    for gname, g in {"pooled": out, "CIC-IDS2017": out[out.dataset == "CIC-IDS2017"],
                     "UNSW-NB15": out[out.dataset == "UNSW-NB15"]}.items():
        if len(g) >= 3:
            pr, pp = pearsonr(g["cov_knn_attack"], g["harness_zd_rate"])
            sr, sp = spearmanr(g["cov_knn_attack"], g["harness_zd_rate"])
            summary[gname] = {"n": len(g), "pearson_r": float(pr), "pearson_p": float(pp),
                              "spearman_r": float(sr), "spearman_p": float(sp)}
            print(f"  [{gname:12s} n={len(g):2d}] harness-ZD vs kNN-cov "
                  f"Pearson r={pr:+.3f} (p={pp:.3g}) Spearman r={sr:+.3f}", flush=True)
    with open(OUT_DIR / f"harness_zd_{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== DONE -> {OUT_DIR/('harness_zd_'+args.tag+'.csv')} ===", flush=True)


if __name__ == "__main__":
    main()
