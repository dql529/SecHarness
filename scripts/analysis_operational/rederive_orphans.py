#!/usr/bin/env python3
"""Re-derive every ORPHAN / registry-only statistic found by the 2026-07-28
provenance audit, from raw audit logs + the on-disk RF model only.

Covers:
  A. Logistic harness x fine-tuning interaction (paper 5_experiments.tex:128,
     "z = -13.81, p = 2.1e-43") - no script/artifact exists anywhere in repo.
     Computed for both E3 variants (method-A relay and superseded audit log).
  B. cic_statistical_tests.txt content (McNemar chi2=1.500 p=0.2207,
     bootstrap [-0.9, 0.0]) - stored txt has no generating script.
  C. E4-clean data-leakage sensitivity check (6_discussion.tex:23).
  D. RQ7 tau table delta arithmetic (paper tab:per_component_scaling).
  E. Supplementary Table S2 latency column (multiseed logs).
  F. Bootstrap F1 diff E3 vs E2 (S3 row "+3.7pp CI [2.2, 5.1]") - absent from
     rq1_methodA json.
  G. Fisher exact ZD E3 vs E2 (S3 row "OR = 0.116, p = 0.000115") - same.
  H. E3 vs E1 discordant count (paper: 721) from stored rq1 json.

Read-only w.r.t. repo; writes orphan_rederivation.json next to this script.
Run: cd project && .venv/bin/python scripts/audit_20260728/rederive_orphans.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import stats

if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
from src.agents.beta_agent_ml import BetaAgentML  # noqa: E402

OUT = Path(__file__).parent / "orphan_rederivation.json"
R: dict = {}


def load(path: Path):
    texts, gt, pred, zd, lat = [], [], [], [], []
    adapter = model = None
    for line in (PROJ / path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        texts.append(r["input"]["traffic_text"])
        gt.append(r["evaluation"]["binary_gt"].lower())
        pred.append(str(r["evaluation"]["binary_pred"]).lower())
        zd.append(bool(r["input"].get("is_zeroday", False)))
        lat.append(float(r["efficiency"]["total_latency_ms"]))
        adapter = r["agent_config"].get("adapter", adapter)
        model = r["agent_config"].get("model", model)
    return texts, np.array(gt), np.array(pred), np.array(zd), np.array(lat), adapter, model


def logit_wald(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson logistic fit; returns (beta, se)."""
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    W = p * (1 - p)
    cov = np.linalg.inv((X * W[:, None]).T @ X)
    return beta, np.sqrt(np.diag(cov))


def main() -> None:
    agent = BetaAgentML.load(PKL := PROJ / "logs/E2_beta_ml_model.pkl")

    # shared UNSW seed-42 subset
    texts, gt, e1, zd, _, _, _ = load(Path("logs/v2/E1_unsw_sub1000_audit.jsonl"))
    _, _, e2, _, _, _, _ = load(Path("logs/v2/E2_unsw_sub1000_audit.jsonl"))
    _, _, e3_art, _, _, _, _ = load(Path("logs/v2/E3_unsw_sub1000_audit.jsonl"))
    _, _, e4, _, _, _, _ = load(Path("logs/v2/E4_unsw_sub1000_audit.jsonl"))
    feat = agent._texts_to_dataframe(texts)
    X = agent._encode_features(feat, fit=False)
    labels = agent.label_encoder.inverse_transform(np.argmax(agent.clf.predict_proba(X), axis=1))
    rf = np.array(["benign" if str(l).lower() in ("normal", "benign") else "attack" for l in labels])

    # --- A. logistic interaction ---
    print("=" * 70, "\nA. Logistic harness x FT interaction (paper: z=-13.81, p=2.1e-43)")
    for tag, e3v in [("methodA_E3:=RF", rf), ("superseded_audit_E3", e3_art)]:
        y = np.concatenate([(e1 == gt), (e2 == gt), (e3v == gt), (e4 == gt)]).astype(float)
        h = np.concatenate([np.zeros(1000), np.zeros(1000), np.ones(1000), np.ones(1000)])
        ft = np.concatenate([np.zeros(1000), np.ones(1000), np.zeros(1000), np.ones(1000)])
        Xd = np.column_stack([np.ones(4000), h, ft, h * ft])
        beta, se = logit_wald(y, Xd)
        z = beta / se
        pvals = 2 * stats.norm.sf(np.abs(z))
        print(f"  [{tag}] interaction beta={beta[3]:+.4f} z={z[3]:+.3f} p={pvals[3]:.3e}")
        R[f"A_interaction_{tag}"] = {"beta": beta.tolist(), "z": z.tolist(), "p": pvals.tolist()}

    # --- B. CIC statistical tests ---
    print("=" * 70, "\nB. CIC E3 vs E4 (stored txt: chi2=1.500 p=0.2207, boot [-0.9,+0.0], mean -0.40pp)")
    _, cgt, ce3, _, _, _, _ = load(Path("logs/v2/E3_cic_full_sub1000_audit.jsonl"))
    _, cgt4, ce4, _, _, _, _ = load(Path("logs/v2/E4_cic_full_sub1000_audit.jsonl"))
    assert np.array_equal(cgt, cgt4)
    a_only = int(np.sum((ce3 == cgt) & (ce4 != cgt)))   # E3-only correct
    b_only = int(np.sum((ce3 != cgt) & (ce4 == cgt)))
    n_disc = a_only + b_only
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
    p = float(stats.chi2.sf(chi2, 1))
    rng = np.random.RandomState(42)
    diffs = []
    n = len(cgt)
    for _ in range(10000):
        idx = rng.randint(0, n, n)
        diffs.append((ce4[idx] == cgt[idx]).mean() - (ce3[idx] == cgt[idx]).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  discordant=({a_only},{b_only}) chi2={chi2:.3f} p={p:.6f}")
    print(f"  bootstrap E4-E3 acc: mean={np.mean(diffs)*100:+.2f}pp 95% CI [{lo*100:+.1f},{hi*100:+.1f}]")
    R["B_cic"] = {"discordant": [a_only, b_only], "chi2": chi2, "p": p,
                  "boot_mean_pp": float(np.mean(diffs) * 100),
                  "boot_ci_pp": [float(lo * 100), float(hi * 100)]}

    # --- C. E4-clean leakage check ---
    print("=" * 70, "\nC. E4-clean vs E4 (paper: identical metrics 92.4/94.4/97.0/11.9)")
    tc, g_c, p_c, zd_c, _, ad_c, _ = load(Path("logs/E4_clean_sub1000_audit.jsonl"))
    t4, g_4, p_4, zd_4, _, ad_4, _ = load(Path("logs/v2/E4_unsw_sub1000_audit.jsonl"))
    same_subset = tc == t4
    def mets(pred, g, z):
        tp = np.sum((g == "attack") & (pred == "attack")); fp = np.sum((g == "benign") & (pred == "attack"))
        fn = np.sum((g == "attack") & (pred == "benign")); tn = np.sum((g == "benign") & (pred == "benign"))
        pr = tp / (tp + fp); rc = tp / (tp + fn)
        return dict(acc=(tp + tn) / len(g), f1=2 * pr * rc / (pr + rc), fpr=fp / (fp + tn),
                    zd=float(np.sum(z & (pred == "attack")) / z.sum()))
    mc, m4 = mets(p_c, g_c, zd_c), mets(p_4, g_4, zd_4)
    match = float((p_c == p_4).mean()) if same_subset else None
    print(f"  adapters: clean={ad_c}  original={ad_4}")
    print(f"  same subset: {same_subset}; per-sample verdict match: {match}")
    print(f"  clean   : acc={mc['acc']:.4f} f1={mc['f1']:.4f} fpr={mc['fpr']:.4f} zd={mc['zd']:.2f}")
    print(f"  original: acc={m4['acc']:.4f} f1={m4['f1']:.4f} fpr={m4['fpr']:.4f} zd={m4['zd']:.2f}")
    R["C_e4_clean"] = {"adapter_clean": ad_c, "adapter_orig": ad_4, "same_subset": bool(same_subset),
                      "verdict_match": match, "clean": mc, "orig": m4}

    # --- D. tau delta arithmetic ---
    print("=" * 70, "\nD. RQ7 tau table arithmetic (paper tab:per_component_scaling)")
    # tau_results.csv only holds the NEW conditions (7B-noTools + 13Bx5 + 32Bx5);
    # 3B rows and 7B full/-K/-O/-P reuse Registry 11a/11b and were already
    # reproduced from raw audit logs by recompute_all_metrics.py.
    import csv as _csv
    tau = {}
    with (PROJ / "results/tables/v2_tdsc/tau_results.csv").open() as f:
        for row in _csv.DictReader(f):
            if row["status"] != "success":
                continue
            tau[(row["model_size"], row["ablation"])] = (float(row["accuracy"]), float(row["fpr"]))
    paper = {  # model -> cond -> acc% (paper tab:per_component_scaling)
        "13B": {"full": 90.5, "noKnowledge": 89.5, "noObservation": 91.5, "noPermissions": 90.5, "noTools": 46.0},
        "32B": {"full": 89.5, "noKnowledge": 89.5, "noObservation": 91.0, "noPermissions": 89.5, "noTools": 71.5},
        "7B": {"noTools": 41.5},
    }
    paper_dfpr_o = {"13B": -6.9, "32B": -17.2}
    bad = 0
    for m, conds in paper.items():
        for c, acc_paper in conds.items():
            key = (m, c)
            if key not in tau:
                print(f"  MISSING in tau_results.csv: {key}")
                bad += 1
                continue
            acc_csv = tau[key][0] * 100
            ok = abs(acc_csv - acc_paper) < 0.051
            print(f"  {m}/{c}: csv={acc_csv:.1f} paper={acc_paper} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad += 1
        if (m, "full") in tau and (m, "noObservation") in tau:
            dfpr = (tau[(m, "noObservation")][1] - tau[(m, "full")][1]) * 100
            ok = abs(dfpr - paper_dfpr_o[m]) < 0.06
            print(f"  {m}: dFPR(-O) csv={dfpr:+.2f}pp paper={paper_dfpr_o[m]:+.1f}pp {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad += 1
    print(f"  tau arithmetic mismatches: {bad}")
    R["D_tau_mismatches"] = bad

    # --- E. multiseed S2 latency ---
    print("=" * 70, "\nE. Supplementary S2 latency column (paper: 4470/5083/5102/4915/4894, mean 4893+/-245)")
    paper_lat = {42: 4470, 123: 5083, 456: 5102, 789: 4915, 2026: 4894}
    lats = []
    for seed, plat in paper_lat.items():
        _, _, _, _, lat, _, _ = load(Path(f"logs/multiseed/E3_seed{seed}_sub1000_audit.jsonl"))
        lats.append(lat.mean())
        print(f"  seed {seed}: log mean={lat.mean():.1f} ms  paper={plat}  "
              f"{'OK' if abs(lat.mean() - plat) < 1 else 'MISMATCH'}")
    mean, std = np.mean(lats), np.std(lats)  # population, like paper's ±
    print(f"  mean={mean:.0f} std_pop={std:.0f} (paper 4893 +/- 245)")
    R["E_latency"] = {"per_seed": [float(x) for x in lats], "mean": float(mean), "std_pop": float(std)}

    # --- F/G. E3 vs E2 F1 bootstrap + Fisher ZD (method A) ---
    print("=" * 70, "\nF. Bootstrap F1 diff E3(:=RF) vs E2 (paper S3: +3.7pp CI [2.2, 5.1])")
    rng = np.random.RandomState(42)
    def f1_of(pred, g):
        tp = np.sum((g == "attack") & (pred == "attack")); fp = np.sum((g == "benign") & (pred == "attack"))
        fn = np.sum((g == "attack") & (pred == "benign"))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    diffs = []
    for _ in range(10000):
        idx = rng.randint(0, 1000, 1000)
        diffs.append(f1_of(rf[idx], gt[idx]) - f1_of(e2[idx], gt[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  mean={np.mean(diffs)*100:+.2f}pp CI [{lo*100:+.1f}, {hi*100:+.1f}]")
    R["F_f1_boot"] = {"mean_pp": float(np.mean(diffs) * 100), "ci_pp": [float(lo * 100), float(hi * 100)]}

    print("\nG. Fisher exact ZD E3 vs E2 (paper S3: OR=0.116, p=0.000115)")
    e2_zd = int(np.sum(zd & (e2 == "attack"))); e3_zd = int(np.sum(zd & (rf == "attack"))); nzd = int(zd.sum())
    table = [[e2_zd, nzd - e2_zd], [e3_zd, nzd - e3_zd]]
    orr, pf = stats.fisher_exact(table)
    print(f"  table={table} OR={orr:.4f} p={pf:.6f}")
    R["G_fisher"] = {"table": table, "OR": float(orr), "p": float(pf)}

    # --- H. E3 vs E1 discordant from stored rq1 json ---
    print("=" * 70, "\nH. E3 vs E1 discordant (paper: 721, p=4.1e-127)")
    j = json.loads((PROJ / "results/tables/repro/rq1_methodA_rederivation.json").read_text())
    mn = j["primary"]["mcnemar"]["E3_vs_E1"]
    print(f"  stored rq1 json: n_discordant={mn['n_discordant']} chi2={mn['chi2']:.2f} p={mn['p']:.3e}")
    relay = [s["recorded_relay_match"] for s in j["multiseed"]["per_seed"]]
    print(f"  multiseed recorded relay match per seed: {relay}")
    R["H_e3_vs_e1"] = mn
    R["H_relay_match"] = relay

    OUT.write_text(json.dumps(R, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
