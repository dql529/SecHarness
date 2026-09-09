"""
run_selective.py — Candidate B, the selective-agent METHOD (RQ9).

Coverage forecasts WHERE the RF relay is blind. The selective agent uses a
per-sample, deployment-time local-coverage signal to decide, per incoming
record, whether to trust the cheap RF relay or pay for LLM reasoning:

  selective(tau) verdict(x):
    if relay(x) == attack            -> attack          # trust RF's attack calls
    elif local_cov(x) >= tau         -> benign          # RF-benign in covered region: trust
    else                             -> noRF_reason(x)  # RF-benign in low-cov: 2nd opinion

local_cov(x) = fraction of x's k nearest neighbours, among a class-balanced
known benign/attack reference in standardized feature space, that are attacks.
It needs no label for x and no RF output -> deployable and non-circular.

Per leave-one-attack-out fold we evaluate BOTH harnesses (relay = full SecHarness
with RF; noRF = SecHarness without check_anomaly) on N held-out attack samples
(zero-day) AND N held-out benign samples, recording per-sample
{local_cov, relay_verdict, norf_verdict}. Sweeping tau then traces a
ZD-recall vs benign-FPR curve; the claim is that the selective curve dominates
both the pure-relay point (misses low-cov zero-days) and the pure-noRF point
(floods FPR).

Usage (on A800):
  python scripts/candidate_b/run_selective.py --n 100 --workers 32 \
    --model api://localhost:8000/llama3.2-3b --tag 3b
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.candidate_b.analyze_coverage_variants import _feat  # noqa: E402
from scripts.candidate_b.run_coverage_zd_stageA import (  # noqa: E402
    ATTACK_CAP, BENIGN_CAP, OUT_DIR, SEED, _cap, load_cic, load_unsw,
)
from scripts.candidate_b.run_harness_zd import (  # noqa: E402
    HARNESS_TOOLS, KNOWLEDGE_DIR, SIGNATURES_DIR, serialize, train_fold_rf,
)
from src.v2.agent_loop import AgentResult, SecHarness, agent_loop  # noqa: E402
from src.v2.llm_engine import LLMEngine  # noqa: E402

KNN_K = 10
KNN_REF_PER_GROUP = 4000
NORF_TOOLS = ["lookup_signature", "query_history", "load_knowledge",
              "classify", "escalate", "log_decision"]  # no check_anomaly


def build_harness(model: str, ml_model_path, tools: List[str]) -> SecHarness:
    return SecHarness(
        llm=LLMEngine(base_model=model, max_input_length=2048),
        ml_model_path=ml_model_path,
        signatures_dir=SIGNATURES_DIR,
        knowledge_dir=KNOWLEDGE_DIR,
        harness_enabled=True,
        enabled_tools=tools,
        permissions_enabled=True,
        max_steps=5,
    )


def verdicts(harness: SecHarness, texts: List[str], workers: int) -> List[str]:
    def one(args):
        i, t = args
        try:
            r = agent_loop(t, harness, sample_index=i, is_zeroday=True)
        except Exception as e:  # noqa: BLE001
            r = AgentResult(verdict="benign", termination="error")
        return r.verdict
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, enumerate(texts)))


def local_cov(known_benign, known_attacks, eval_df, num_cols, cat_cols,
              rng) -> np.ndarray:
    """Per-sample fraction of kNN (balanced known ref) that are attacks."""
    if cat_cols:
        ka = pd.concat([known_benign] + list(known_attacks.values()), ignore_index=True)
        oh = list(pd.get_dummies(ka[cat_cols].astype(str), columns=cat_cols).columns)
    else:
        oh = []
    Xb = _feat(known_benign, num_cols, cat_cols, oh)
    Xa = np.vstack([_feat(a, num_cols, cat_cols, oh) for a in known_attacks.values()])
    Xe = _feat(eval_df, num_cols, cat_cols, oh)
    pool = np.vstack([Xb, Xa])
    med = np.nan_to_num(np.nanmedian(pool, axis=0))

    def clean(X):
        X = X.copy(); idx = np.where(np.isnan(X)); X[idx] = np.take(med, idx[1]); return X
    pool = clean(pool); Xb, Xa, Xe = clean(Xb), clean(Xa), clean(Xe)
    mu, sd = pool.mean(0), pool.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
    nb = min(KNN_REF_PER_GROUP, len(Xb)); na = min(KNN_REF_PER_GROUP, len(Xa))
    ref = np.vstack([(Xb[rng.choice(len(Xb), nb, replace=False)] - mu) / sd,
                     (Xa[rng.choice(len(Xa), na, replace=False)] - mu) / sd])
    is_atk = np.array([0] * nb + [1] * na)
    nn = NearestNeighbors(n_neighbors=KNN_K).fit(ref)
    _, ind = nn.kneighbors((Xe - mu) / sd)
    return is_atk[ind].mean(axis=1)


def run(model: str, n_eval: int, workers: int, datasets: List[str],
        tag: str, ckpt: Path) -> pd.DataFrame:
    loaders = {"cic": ("CIC-IDS2017", load_cic), "unsw": ("UNSW-NB15", load_unsw)}
    pkldir = OUT_DIR / "rf_pkls"
    pkldir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    done = set()
    if ckpt.exists():
        prev = pd.read_csv(ckpt)
        rows = prev.to_dict("records")
        done = {(r["dataset"], r["held_class"], r["sample_type"], r["idx"]) for r in rows}
        print(f"[resume] {len(done)} sample-rows in {ckpt.name}", flush=True)

    for key in datasets:
        name, loader = loaders[key]
        df, num_cols, cat_cols, benign, eligible = loader()
        benign_all = df[df["is_benign"]]
        rng = np.random.default_rng(SEED)
        for c in eligible:
            if (name, c, "attack", 0) in done:
                continue
            held = _cap(df[df["cls"] == c], n_eval, SEED)
            if len(held) < 10:
                continue
            known_benign = _cap(benign_all, BENIGN_CAP, SEED)
            known_attacks = {a: _cap(df[df["cls"] == a], ATTACK_CAP, SEED)
                             for a in eligible if a != c}
            # held-out benign eval: GUARANTEED disjoint from training benign
            # (_cap samples randomly, so drop the exact sampled indices).
            beneval = _cap(benign_all.drop(known_benign.index), n_eval, SEED + 1)

            known_pool = pd.concat([known_benign] + list(known_attacks.values()),
                                   ignore_index=True)
            pkl = pkldir / f"sel_{key}_{c.replace(' ', '_').replace('/', '_')}.pkl"
            bin_specs = train_fold_rf(known_pool, num_cols, cat_cols, pkl)
            relay = build_harness(model, str(pkl), HARNESS_TOOLS)
            norf = build_harness(model, None, NORF_TOOLS)

            for stype, edf in (("attack", held), ("benign", beneval)):
                edf = edf.copy()
                edf["text"] = serialize(edf, num_cols, cat_cols, bin_specs)
                lc = local_cov(known_benign, known_attacks, edf, num_cols, cat_cols, rng)
                rv = verdicts(relay, edf["text"].tolist(), workers)
                nv = verdicts(norf, edf["text"].tolist(), workers)
                for j in range(len(edf)):
                    rows.append({"dataset": name, "held_class": c, "sample_type": stype,
                                 "idx": j, "local_cov": float(lc[j]),
                                 "relay_verdict": rv[j], "norf_verdict": nv[j]})
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            ar = np.mean([r["relay_verdict"] == "attack" for r in rows
                          if r["dataset"] == name and r["held_class"] == c and r["sample_type"] == "attack"])
            print(f"  {name} {c:18s} relay_ZD={ar:.3f} (eval done)", flush=True)
    return pd.DataFrame(rows)


def sweep(df: pd.DataFrame) -> Dict:
    """Compute ZD-recall vs FPR for pure-relay, pure-noRF, and selective(tau)."""
    atk = df[df.sample_type == "attack"]; ben = df[df.sample_type == "benign"]

    def rates(sel_atk, sel_ben):
        zd = float(np.mean(sel_atk == "attack")) if len(sel_atk) else 0.0
        fpr = float(np.mean(sel_ben == "attack")) if len(sel_ben) else 0.0
        return zd, fpr

    out = {"relay": rates(atk.relay_verdict.values, ben.relay_verdict.values),
           "norf": rates(atk.norf_verdict.values, ben.norf_verdict.values),
           "selective_curve": []}

    def selective(sub, tau):
        v = sub.relay_verdict.values.copy().astype(object)
        relay_benign = sub.relay_verdict.values == "benign"
        lowcov = sub.local_cov.values < tau
        route = relay_benign & lowcov
        v[route] = sub.norf_verdict.values[route]
        # relay_benign & not lowcov stays benign (already relay's benign)
        return v

    for tau in np.round(np.arange(0.0, 1.01, 0.1), 2):
        zd, fpr = rates(selective(atk, tau), selective(ben, tau))
        out["selective_curve"].append({"tau": float(tau), "zd_recall": zd, "fpr": fpr})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="api://localhost:8000/llama3.2-3b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", default="3b")
    ap.add_argument("--datasets", nargs="+", default=["cic", "unsw"])
    args = ap.parse_args()
    ckpt = OUT_DIR / f"selective_{args.tag}_samples.csv"
    print(f"=== Selective agent (model={args.model}, n={args.n}, workers={args.workers}, "
          f"tag={args.tag}) ===", flush=True)
    df = run(args.model, args.n, args.workers, args.datasets, args.tag, ckpt)
    res = sweep(df)
    res["model"] = args.model
    res["n_eval"] = args.n
    with open(OUT_DIR / f"selective_{args.tag}_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"relay  : ZD-recall={res['relay'][0]:.3f} FPR={res['relay'][1]:.3f}", flush=True)
    print(f"noRF   : ZD-recall={res['norf'][0]:.3f} FPR={res['norf'][1]:.3f}", flush=True)
    print("selective(tau):", flush=True)
    for p in res["selective_curve"]:
        print(f"  tau={p['tau']:.1f}  ZD-recall={p['zd_recall']:.3f}  FPR={p['fpr']:.3f}", flush=True)
    print(f"=== DONE -> {OUT_DIR/('selective_'+args.tag+'_summary.json')} ===", flush=True)


if __name__ == "__main__":
    main()
