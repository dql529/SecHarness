"""Step 3: are the override samples the SAME samples across models, or different ones?

step0_deference.csv shows a suspicious pile-up on the UNSW 200-sample subset: E3_8B (which
actually ran Qwen2.5-7B), E3_70B, E3_32B_local, tau_32B_full and tau_13B_noKnowledge all land on
exactly 182/200 agreement, and E3_14B on 185/200. Equal counts do not imply equal sets, so this
script asks whether the disagreeing samples coincide.

Why it matters. The agent and the RandomForest read the SAME representation: preprocess.py bins
every numeric feature to a quantile label and BetaAgentML parses that same kv text, so the LLM
holds no private evidence its tool lacks. Under identical inputs and identical tool outputs there
are only two honest readings, and the overlap statistic separates them:

  overlap HIGH  -> override is a property of the INPUT, not of the model. A cross-family gap in
                   the override RATE is then a threshold effect on a shared difficulty ordering,
                   not a behavioural disposition of a post-training recipe.
  overlap LOW   -> models disagree essentially at random on borderline records. Also not a
                   mechanism, and it caps how much any rate difference can mean.

Null model for "low": the disagreement sets are drawn independently and uniformly at random from
the n common samples at each model's own observed marginal rate. Then for a pair with counts
d_i, d_j the intersection is hypergeometric, E|D_i n D_j| = d_i*d_j/n, and for K models
E|intersection of all| = n * prod(d_i/n). Both are reported next to the observed values, with a
one-sided hypergeometric tail probability per pair.

Disagreement direction is reported separately because the two directions are not equally
interesting: tool-says-attack -> agent-says-benign is a missed detection, the security-relevant
failure; tool-says-benign -> agent-says-attack only costs an analyst's time.

Deference here is read from the LOGGED tool output (tool_chain[].output), never from an offline
replay, so nothing in this script depends on the RF reproduction gate.

Usage: .venv/bin/python scripts/deference/step3_disagreement_overlap.py [--group NAME]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from scipy.stats import hypergeom, mannwhitneyu

PROJ = Path(__file__).resolve().parents[2]
LOGS = PROJ / "logs"

# every UNSW tool-channel log drawn from the same 200-sample subsample
GROUP: dict[str, list[str]] = {
    "core": [  # one log per distinct backbone, full harness configuration
        "v2/E3_unsw_sub200_audit.jsonl",            # unsloth/Llama-3.2-3B-Instruct
        "v2/E3_8B_unsw_sub200_audit.jsonl",         # Qwen/Qwen2.5-7B-Instruct (filename lies)
        "v2/E3_14B_unsw_sub200_audit.jsonl",        # qwen2.5:14b
        "v2/E3_32B_local_unsw_sub200_audit.jsonl",  # qwen2.5:32b
        "v2/E3_70B_unsw_sub200_audit.jsonl",        # llama-3.3-70b-instruct-awq
        "v2_tdsc/tau_13B_full_unsw_sub200_audit.jsonl",   # Qwen2.5-14B-Instruct-AWQ
        "v2_tdsc/tau_32B_full_unsw_sub200_audit.jsonl",   # Qwen2.5-32B-Instruct-AWQ
        "v2/E4_ablation_full_unsw_sub200_audit.jsonl",    # Llama-3.2-3B, fine-tuned
    ],
    "all182": [  # the cells that tie at 182/200 in step0_deference.csv, plus 14B at 185
        "v2/E3_8B_unsw_sub200_audit.jsonl",
        "v2/E3_70B_unsw_sub200_audit.jsonl",
        "v2/E3_32B_local_unsw_sub200_audit.jsonl",
        "v2_tdsc/tau_32B_full_unsw_sub200_audit.jsonl",
        "v2_tdsc/tau_13B_noKnowledge_unsw_sub200_audit.jsonl",
        "v2/E3_14B_unsw_sub200_audit.jsonl",
    ],
}


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


def tool_out(rec: dict) -> dict | None:
    """First check_anomaly output as a dict. This is the runtime evidence, straight from the log."""
    for step in rec.get("tool_chain") or []:
        if step.get("tool") != "check_anomaly":
            continue
        o = step.get("output")
        if isinstance(o, dict):
            return o
        if isinstance(o, str):
            try:
                return json.loads(o)
            except json.JSONDecodeError:
                return None
        return None
    return None


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


def load(rel: str) -> tuple[str, dict[int, dict]]:
    """-> (recorded model, {sample_index: per-sample facts}). Last record wins on duplicate index."""
    path = LOGS / rel
    per: dict[int, dict] = {}
    models: set[str] = set()
    for r in read_jsonl(path):
        to = tool_out(r)
        tb, ab = bin_tool((to or {}).get("prediction")), bin_verdict((r.get("result") or {}).get("verdict"))
        if to is None or tb is None or ab is None:
            continue
        models.add((r.get("agent_config") or {}).get("model", "?"))
        per[int(r.get("sample_index", -1))] = {
            "tool": tb, "agent": ab, "disagree": tb != ab,
            "conf": to.get("confidence"), "score": to.get("anomaly_score"),
            "pred": to.get("prediction"),
            "gt": (r.get("input") or {}).get("ground_truth_label"),
            "text": (r.get("input") or {}).get("traffic_text"),
        }
    model = sorted(models)[0] if len(models) == 1 else f"MIXED({len(models)})"
    return model, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="core", choices=sorted(GROUP))
    args = ap.parse_args()

    loaded: list[tuple[str, str, dict[int, dict]]] = []
    for rel in GROUP[args.group]:
        if not (LOGS / rel).exists():
            print(f"MISSING {rel}")
            continue
        model, per = load(rel)
        loaded.append((Path(rel).name, model, per))

    common = set.intersection(*[set(p) for _, _, p in loaded])
    idx = sorted(common)
    n = len(idx)
    print("=" * 104)
    print(f"GROUP '{args.group}': {len(loaded)} logs, {n} sample indices present in all of them")
    print("=" * 104)

    # the subsample must actually be the same records, not just the same indices
    ref = loaded[0][2]
    bad_text = [i for i in idx if any(p[i]["text"] != ref[i]["text"] for _, _, p in loaded)]
    same_tool = [i for i in idx if len({p[i]["pred"] for _, _, p in loaded}) == 1]
    print(f"  identical traffic_text at every shared index: {n - len(bad_text)}/{n}"
          f"   ({'SAME SUBSAMPLE' if not bad_text else 'INDEX COLLISION -- results below are void'})")
    print(f"  identical tool prediction at every shared index: {len(same_tool)}/{n}"
          f"   (the tool is model-independent, so this should be all of them)")
    print()

    hdr = f"  {'log':<48}{'model':<30}{'n':>4}{'disagree':>10}{'rate':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    D: dict[str, set[int]] = {}
    for name, model, per in loaded:
        d = {i for i in idx if per[i]["disagree"]}
        D[name] = d
        m = model.replace("Qwen/", "").replace("meta-llama/", "").replace("unsloth/", "")
        print(f"  {name[:47]:<48}{m[:29]:<30}{n:>4}{len(d):>10}{len(d)/n:>8.3f}")

    # ---- pairwise overlap vs the independence null -------------------------------------------
    print("\n" + "=" * 104)
    print("PAIRWISE DISAGREEMENT-SET OVERLAP   (obs = observed intersection, exp = d_i*d_j/n under")
    print("independent uniform selection; p = P[X >= obs] under hypergeom(n, d_i, d_j))")
    print("=" * 104)
    h = (f"  {'model A':<26}{'model B':<26}{'|A|':>4}{'|B|':>4}{'obs':>5}{'exp':>7}"
         f"{'Jacc':>7}{'expJ':>7}{'p':>10}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    jaccs: list[float] = []
    for a, b in itertools.combinations(D, 2):
        A, B = D[a], D[b]
        if not A or not B:
            continue
        obs = len(A & B)
        exp = len(A) * len(B) / n
        jac = obs / len(A | B) if (A | B) else float("nan")
        expj = exp / (len(A) + len(B) - exp) if (len(A) + len(B) - exp) else float("nan")
        p = float(hypergeom.sf(obs - 1, n, len(A), len(B)))
        jaccs.append(jac)
        print(f"  {a.replace('_unsw_sub200_audit.jsonl','')[:25]:<26}"
              f"{b.replace('_unsw_sub200_audit.jsonl','')[:25]:<26}"
              f"{len(A):>4}{len(B):>4}{obs:>5}{exp:>7.1f}{jac:>7.3f}{expj:>7.3f}{p:>10.2e}")
    if jaccs:
        print(f"\n  mean pairwise Jaccard = {sum(jaccs)/len(jaccs):.3f}  "
              f"(min {min(jaccs):.3f}, max {max(jaccs):.3f}) over {len(jaccs)} pairs")

    # ---- how many models disagree per sample -------------------------------------------------
    print("\n" + "=" * 104)
    print("PER-SAMPLE CONSENSUS: how many of the models overrode each sample")
    print("=" * 104)
    K = len(D)
    cnt = {i: sum(i in d for d in D.values()) for i in idx}
    hist = {k: sum(1 for i in idx if cnt[i] == k) for k in range(K + 1)}
    # E[# samples overridden by exactly k models] under independence, Poisson-binomial
    rates = [len(d) / n for d in D.values()]

    def pb(k: int) -> float:
        tot = 0.0
        for combo in itertools.combinations(range(K), k):
            s = set(combo)
            pr = 1.0
            for j, r in enumerate(rates):
                pr *= r if j in s else (1 - r)
            tot += pr
        return tot * n

    print(f"  {'k models override':<22}{'observed':>10}{'expected (indep.)':>20}")
    print("  " + "-" * 50)
    for k in range(K + 1):
        print(f"  {k:<22}{hist[k]:>10}{pb(k):>20.2f}")
    allk = [i for i in idx if cnt[i] == K]
    one = [i for i in idx if cnt[i] == 1]
    print(f"\n  overridden by ALL {K} models: {len(allk)}  (expected under independence "
          f"{n * math.prod(rates):.2f})")
    print(f"  overridden by exactly ONE model: {len(one)}  (expected {pb(1):.2f})")
    print(f"  overridden by at least one model: {sum(1 for i in idx if cnt[i] > 0)}/{n}")

    # ---- direction, ground truth, and tool confidence ----------------------------------------
    print("\n" + "=" * 104)
    print("DIRECTION AND TOOL CONFIDENCE ON OVERRIDDEN SAMPLES")
    print("=" * 104)
    print(f"  {'log':<40}{'tool=atk->agent=benign':>24}{'tool=ben->agent=atk':>22}")
    print("  " + "-" * 84)
    for name, _, per in loaded:
        a2b = sum(1 for i in D[name] if per[i]["tool"] == "malicious")
        b2a = sum(1 for i in D[name] if per[i]["tool"] == "benign")
        print(f"  {name.replace('_unsw_sub200_audit.jsonl','')[:39]:<40}{a2b:>24}{b2a:>22}")

    print(f"\n  {'log':<34}{'median conf':>13}{'median conf':>13}{'MWU p':>10}")
    print(f"  {'':<34}{'(override)':>13}{'(no override)':>13}")
    print("  " + "-" * 70)
    for name, _, per in loaded:
        ov = [per[i]["conf"] for i in D[name] if per[i]["conf"] is not None]
        no = [per[i]["conf"] for i in idx if i not in D[name] and per[i]["conf"] is not None]
        if len(ov) < 3 or len(no) < 3:
            print(f"  {name[:33]:<34}{'n/a (too few)':>37}")
            continue
        p = float(mannwhitneyu(ov, no, alternative="two-sided").pvalue)
        print(f"  {name.replace('_unsw_sub200_audit.jsonl','')[:33]:<34}"
              f"{sorted(ov)[len(ov)//2]:>13.3f}{sorted(no)[len(no)//2]:>13.3f}{p:>10.2e}")

    print("\n  ground-truth label of the samples overridden by ALL models:")
    if allk:
        gt: dict[str, int] = {}
        for i in allk:
            gt[ref[i]["gt"]] = gt.get(ref[i]["gt"], 0) + 1
        for k, v in sorted(gt.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {v}")
        print("  their tool predictions and confidences:")
        for i in allk[:12]:
            print(f"    idx={i:<5} gt={str(ref[i]['gt']):<16} tool={ref[i]['pred']:<16}"
                  f" conf={ref[i]['conf']}  anomaly_score={ref[i]['score']}")
    else:
        print("    none")


if __name__ == "__main__":
    main()
