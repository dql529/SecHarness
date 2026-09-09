"""Step 2: does the llama3.1:8b vs Qwen2.5-7B prompt-channel deference gap survive the RF drift?

step1_prompt_channel.csv reports, on the UNSW 200-sample subset, llama3.1:8b 198/200 = 0.990 and
Qwen/Qwen2.5-7B-Instruct 172/200 = 0.860 -- a 13pp same-scale cross-family gap, against ~1.5pp for
a 10x scale span. Both numbers depend on re-running the RF offline, because the prompt-channel arm
(E1_rf_context_*) injects the RF opinion into the prompt (agent_loop.py:203-215) and never
persists it. step1's own gate said that replay reproduces the runtime only 96.1% of the time, so
the gap was suspect.

Two things are established before anything is recomputed here.

1. ANALYTIC. Let t_i be the true binarised RF output, v_i^M the agent verdict (read from the log,
   so exact), and E the set of samples where the replay recovers t_i wrongly, |E| = e*n. Because
   both quantities are binary, on i in E the indicator flips exactly:
       1[v_i = t^_i] = 1 - 1[v_i = t_i]
   so for one model
       D^_M - D_M = (1/n) * sum_{i in E} (1 - 2*1[v_i = t_i])
                  = (|E and disagree_M| - |E and agree_M|) / n   in [-e, +e]
   and for the gap G = D_A - D_B,
       G^ - G = (D^_A - D_A) - (D^_B - D_B)   in [-2e, +2e].
   So |measured_gap - true_gap| <= 2e, and the bound is TIGHT: it is attained when every sample in
   E is one the replay pushes into agreement for A and out of agreement for B. The drift being
   common-mode (the RF sees only traffic_text, never which LLM is under test) does NOT shrink this
   -- the same wrong t^_i can help one model and hurt the other, which is precisely the worst case.
   Common-mode only guarantees E is the same SET for both models, not that its effect cancels.

2. EMPIRICAL. step1a_rf_drift_forensics.py shows e = 0: the gate failed only because step1
   replayed the wrong log field. Replaying tool_chain[].input.traffic_text (the argument actually
   dispatched) reproduces all four logged fields 1000/1000 on both UNSW and CIC. So the analytic
   bound collapses to |measured - true| <= 0. This script does not take that on trust; it
   re-derives e directly on the prompt-channel samples.

The method for that: the E3 tool-channel logs are a lookup table traffic_text -> the RF output the
runtime really produced. The prompt-channel subsample is drawn from the same UNSW val split, so
many of its 200 texts also appear in an E3 log. On the overlap, deference can be computed from the
LOGGED RF output and never from a replay at all -- an audit that is independent of the pickle.
Only the E3 logs written after the agent_loop.py:458-460 argument-override existed are usable as
that table (E3_unsw_sub200 and E3_unsw_sub1000 dispatched LLM-authored arguments ~50% of the time,
so their logged output does not correspond to their own input.traffic_text).

Usage: .venv/bin/python scripts/deference/step2_conditioned_gap.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import types
import warnings
from pathlib import Path

from scipy.stats import fisher_exact, hypergeom, norm

sys.modules.setdefault("requests", types.ModuleType("requests"))  # src/agents/__init__ pulls it
PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
warnings.filterwarnings("ignore")  # pickle written by sklearn 1.7.2, venv has 1.8.0

from src.v2.tools.check_anomaly import CheckAnomalyTool  # noqa: E402

LOGS = PROJ / "logs"
RF_UNSW = LOGS / "E2_beta_ml_model.pkl"

ARMS = {
    "llama3.1:8b": "v2/E1_rf_context_8B_ollama_unsw_sub200_audit.jsonl",
    "Qwen2.5-7B": "v2/E1_rf_context_7B_unsw_sub200_audit.jsonl",
    "qwen2.5:14b": "v2/E1_rf_context_14B_unsw_sub200_audit.jsonl",
    "qwen2.5:32b": "v2/E1_rf_context_32B_local_unsw_sub200_audit.jsonl",
    "llama-3.3-70b": "v2/E1_rf_context_70B_unsw_sub200_audit.jsonl",
    "gemma4:e4b": "v2/E1_rf_context_12B_unsw_sub200_audit.jsonl",
    "Llama-3.2-3B": "v2/E1_rf_context_unsw_sub1000_audit.jsonl",
}
# E3/E4/tau logs whose dispatched tool argument == input.traffic_text (verified by step1a),
# usable as a runtime RF lookup table
CLEAN_E3 = ["v2/E3_8B_unsw_sub200_audit.jsonl", "v2/E3_14B_unsw_sub200_audit.jsonl",
            "v2/E3_70B_unsw_sub200_audit.jsonl", "v2/E3_32B_local_unsw_sub200_audit.jsonl",
            "v2/E4_unsw_sub1000_audit.jsonl", "multiseed/E3_seed42_sub1000_audit.jsonl",
            "multiseed/E3_seed123_sub1000_audit.jsonl", "multiseed/E3_seed456_sub1000_audit.jsonl",
            "multiseed/E3_seed789_sub1000_audit.jsonl", "multiseed/E3_seed2026_sub1000_audit.jsonl",
            "v2_tdsc/tau_13B_full_unsw_sub200_audit.jsonl",
            "v2_tdsc/tau_32B_full_unsw_sub200_audit.jsonl"]

Z975 = 1.959963984540054   # norm.ppf(0.975)


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def wilson(k: int, n: int, z: float = Z975) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def bin_tool(pred: str | None) -> str | None:
    if not pred:
        return None
    return "benign" if str(pred).strip().lower() in ("normal", "benign") else "malicious"


def bin_verdict(v: str | None) -> str | None:
    if not v:
        return None
    v = str(v).strip().lower()
    if v in ("benign", "normal"):
        return "benign"
    if v in ("malicious", "attack", "anomaly"):
        return "malicious"
    return None


def two_prop(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float, float]:
    """Pooled two-proportion z test + Newcombe square-and-add CI for the difference."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    pbar = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("inf")
    pval = 2 * (1 - norm.cdf(abs(z)))
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return z, float(pval), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    # ---- runtime RF lookup table from the clean tool-channel logs --------------------------
    table: dict[str, str] = {}
    conflicts = 0
    for rel in CLEAN_E3:
        p = LOGS / rel
        if not p.exists():
            continue
        for r in read_jsonl(p):
            for s in r.get("tool_chain") or []:
                if s.get("tool") != "check_anomaly":
                    continue
                arg = (s.get("input") or {}).get("traffic_text")
                true_t = (r.get("input") or {}).get("traffic_text")
                out = s.get("output")
                if not arg or arg != true_t or not isinstance(out, str):
                    break  # LLM-authored or truncated argument: not a usable table entry
                try:
                    pred = json.loads(out).get("prediction")
                except json.JSONDecodeError:
                    break
                if arg in table and table[arg] != pred:
                    conflicts += 1
                table[arg] = pred
                break
    print(f"runtime RF lookup table built from {len(CLEAN_E3)} clean tool-channel logs: "
          f"{len(table)} distinct traffic_text -> logged RF prediction, {conflicts} conflicts")
    print("  (a conflict would mean the runtime RF was not a function of its argument; "
          "expected 0)\n")

    rf = CheckAnomalyTool(str(RF_UNSW), degraded=False)

    rows: list[dict] = []
    dis_full: dict[str, set[int]] = {}
    dis_ovl: dict[str, set[int]] = {}
    facts: dict[str, dict[int, dict]] = {}
    for label, rel in ARMS.items():
        p = LOGS / rel
        if not p.exists():
            print(f"MISSING {rel}")
            continue
        recs = [r for r in read_jsonl(p) if (r.get("input") or {}).get("traffic_text")]
        model = sorted({(r.get("agent_config") or {}).get("model", "?") for r in recs})
        n_raw = len(recs)
        agree_replay = agree_logged = n_full = n_ovl = e_bad = 0
        dfull: set[int] = set()
        dovl: set[int] = set()
        per: dict[int, dict] = {}
        for r in recs:
            t = r["input"]["traffic_text"]
            av = bin_verdict((r.get("result") or {}).get("verdict"))
            if av is None:
                continue
            rep = json.loads(rf(t))
            rep_b = bin_tool(rep.get("prediction"))
            i = int(r.get("sample_index", -1))
            n_full += 1
            if rep_b == av:
                agree_replay += 1
            else:
                dfull.add(i)
            per[i] = {"conf": rep.get("confidence"), "score": rep.get("anomaly_score"),
                      "tool": rep_b, "agent": av,
                      "gt": (r.get("input") or {}).get("ground_truth_label")}
            if t in table:
                n_ovl += 1
                log_b = bin_tool(table[t])
                if log_b != rep_b:
                    e_bad += 1
                if log_b == av:
                    agree_logged += 1
                else:
                    dovl.add(i)
        dis_full[label] = dfull
        dis_ovl[label] = dovl
        facts[label] = per
        rows.append({"label": label, "model": model[0] if len(model) == 1 else f"MIXED{model}",
                     "n_raw": n_raw, "n_full": n_full, "k_full": agree_replay,
                     "n_ovl": n_ovl, "k_ovl": agree_logged, "e_bad": e_bad})

    print("=" * 112)
    print("PROMPT CHANNEL, TWO WAYS OF GETTING THE TOOL SIDE")
    print("  replayed : deference vs the offline RF replay on all samples (what step1 does)")
    print("  logged   : deference vs the RUNTIME-LOGGED RF output, on the subset of samples whose")
    print("             traffic_text also appears in a clean tool-channel log (no pickle involved)")
    print("=" * 112)
    hdr = (f"  {'model':<32}{'n':>5}{'replayed':>10}{'95% Wilson':>18}"
           f"{'n_ovl':>7}{'logged':>9}{'95% Wilson':>18}{'recov.err':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        pf, lf, hf = wilson(r["k_full"], r["n_full"])
        po, lo, ho = wilson(r["k_ovl"], r["n_ovl"]) if r["n_ovl"] else (float("nan"),) * 3
        ov = f"{po:.3f}" if r["n_ovl"] else "n/a"
        ci = f"[{lo:.3f}, {ho:.3f}]" if r["n_ovl"] else "n/a"
        print(f"  {r['label']:<32}{r['n_full']:>5}{pf:>10.3f}   [{lf:.3f}, {hf:.3f}]"
              f"{r['n_ovl']:>7}{ov:>9}   {ci:<18}{r['e_bad']:>4}/{r['n_ovl']}")

    e_tot = sum(r["e_bad"] for r in rows)
    n_tot = sum(r["n_ovl"] for r in rows)
    e_rate = e_tot / n_tot if n_tot else float("nan")
    print(f"\n  pooled per-sample RF recovery error e = {e_tot}/{n_tot} = {e_rate:.4f}")
    print(f"  => analytic worst case |measured_gap - true_gap| <= 2e = {2*e_rate:.4f} "
          f"({200*e_rate:.1f} samples of 200 per model)")

    # ---- the contrast under test -------------------------------------------------------------
    A, B = "llama3.1:8b", "Qwen2.5-7B"
    ra = next(r for r in rows if r["label"] == A)
    rb = next(r for r in rows if r["label"] == B)
    print("\n" + "=" * 112)
    print(f"THE CONTRAST UNDER TEST: {A} vs {B} (same 7-8B scale, different family)")
    print("=" * 112)
    for tag, ka, na, kb, nb in (("all samples, replayed RF",
                                 ra["k_full"], ra["n_full"], rb["k_full"], rb["n_full"]),
                                ("overlap only, RUNTIME-LOGGED RF",
                                 ra["k_ovl"], ra["n_ovl"], rb["k_ovl"], rb["n_ovl"])):
        if not na or not nb:
            continue
        z, pv, lo, hi = two_prop(ka, na, kb, nb)
        pa, _, _ = wilson(ka, na)
        pb, _, _ = wilson(kb, nb)
        odds, fp = fisher_exact([[ka, na - ka], [kb, nb - kb]])
        print(f"\n  {tag}")
        print(f"    {A:<14} {ka}/{na} = {pa:.4f}")
        print(f"    {B:<14} {kb}/{nb} = {pb:.4f}")
        print(f"    gap = {pa - pb:+.4f} ({100*(pa-pb):+.1f}pp)   "
              f"Newcombe 95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    two-proportion z = {z:.3f}, p = {pv:.3e}   Fisher exact p = {fp:.3e}")
        print(f"    verdict: {'SURVIVES (CI excludes 0)' if lo > 0 else 'DOES NOT SURVIVE'}")

    # scale contrast for reference: within-family 7B -> 70B
    print("\n  reference scale contrast within the same measurement:")
    for a, b in (("Qwen2.5-7B", "qwen2.5:32b"), ("llama3.1:8b", "llama-3.3-70b")):
        xa = next((r for r in rows if r["label"] == a), None)
        xb = next((r for r in rows if r["label"] == b), None)
        if not xa or not xb:
            continue
        z, pv, lo, hi = two_prop(xa["k_full"], xa["n_full"], xb["k_full"], xb["n_full"])
        pa = xa["k_full"] / xa["n_full"]
        pb = xb["k_full"] / xb["n_full"]
        print(f"    {a:<14} vs {b:<14} gap {100*(pa-pb):+5.1f}pp  "
              f"CI [{100*lo:+.1f}, {100*hi:+.1f}]pp  p={pv:.3f}")

    # ---- is the gap the SAME samples plus more, or a different set? --------------------------
    print("\n" + "=" * 112)
    print("STRUCTURE OF THE GAP: do the two families override the same samples?")
    print("=" * 112)
    common = set(facts[A]) & set(facts[B])
    n = len(common)
    DA = {i for i in dis_full[A] if i in common}
    DB = {i for i in dis_full[B] if i in common}
    obs = len(DA & DB)
    exp = len(DA) * len(DB) / n if n else float("nan")
    jac = obs / len(DA | DB) if (DA | DB) else float("nan")
    hp = float(hypergeom.sf(obs - 1, n, len(DA), len(DB))) if DA and DB else float("nan")
    print(f"  common sample indices: {n}")
    print(f"  |override({A})| = {len(DA)}   |override({B})| = {len(DB)}")
    print(f"  intersection observed {obs}, expected under independence {exp:.2f}, "
          f"Jaccard {jac:.3f}, hypergeom P[X>={obs}] = {hp:.3e}")
    print(f"  nested? override({A}) subset of override({B}): "
          f"{DA.issubset(DB)}   ({len(DA - DB)} samples only {A} overrides)")

    print("\n  direction and tool confidence on overridden samples (replayed RF fields):")
    print(f"    {'model':<16}{'tool=atk->benign':>18}{'tool=ben->atk':>15}"
          f"{'med conf ovr':>14}{'med conf rest':>15}")
    for label in (A, B, "qwen2.5:14b", "llama-3.3-70b"):
        if label not in facts:
            continue
        per = facts[label]
        d = dis_full[label]
        a2b = sum(1 for i in d if per[i]["tool"] == "malicious")
        b2a = sum(1 for i in d if per[i]["tool"] == "benign")
        co = sorted(per[i]["conf"] for i in d)
        cr = sorted(per[i]["conf"] for i in per if i not in d)
        mo = f"{co[len(co)//2]:.3f}" if co else "n/a"
        mr = f"{cr[len(cr)//2]:.3f}" if cr else "n/a"
        print(f"    {label:<16}{a2b:>18}{b2a:>15}{mo:>14}{mr:>15}")

    # pairwise Jaccard across the whole prompt channel, for context
    print("\n  pairwise Jaccard of prompt-channel override sets (all arms, common indices only):")
    labels = [r["label"] for r in rows]
    for a, b in itertools.combinations(labels, 2):
        cm = set(facts[a]) & set(facts[b])
        Da = {i for i in dis_full[a] if i in cm}
        Db = {i for i in dis_full[b] if i in cm}
        if not (Da or Db):
            continue
        u = len(Da | Db)
        print(f"    {a:<16} vs {b:<16} |A|={len(Da):>3} |B|={len(Db):>3} "
              f"inter={len(Da & Db):>3} Jaccard={len(Da & Db)/u:.3f}")


if __name__ == "__main__":
    main()
