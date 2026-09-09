#!/usr/bin/env python3
"""Quantify the numerical impact of the ml-prediction fix by replaying audit logs.

No GPU, no re-inference. The audit logs record the agent's own classify arguments, so the
post-fix verdict can be derived exactly:

  * LOW_CONF path  — pre-fix: ConfidenceGate saw the error dict, read .get("prediction",
    "Normal") and overwrote the verdict with benign. Post-fix: ml_prediction is None, the
    gate short-circuits, so the agent's own classify verdict stands. VERDICT MAY CHANGE.
  * MAX_STEPS path — pre-fix: _force_verdict read the same default "Normal" -> benign.
    Post-fix: it takes the explicit default-benign branch. SAME VERDICT, honest log only.

Usage: python3 replay_fix_impact.py <project_root>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS = [
    ("3B  -T      (Table 7)", "logs/v2/E4_ablation_noTools_unsw_sub200_audit.jsonl"),
    ("32B -T      (Table 7)", "logs/v2_tdsc/tau_32B_noTools_unsw_sub200_audit.jsonl"),
    ("8B  E3-noRF (Sec 4.7)", "logs/v2/E3_noRF_8B_unsw_sub200_audit.jsonl"),
    ("14B E3-noRF (Sec 4.7)", "logs/v2/E3_noRF_14B_unsw_sub200_audit.jsonl"),
    ("32B E3-noRF (Sec 4.7)", "logs/v2/E3_noRF_32B_unsw_sub200_audit.jsonl"),
]


def agent_own_verdict(rec: dict) -> str | None:
    """The verdict the agent itself emitted, before permission rewriting.

    NOT tool_chain: that records the post-permission args (verdict already overwritten
    with benign, reasoning already replaced by the LOW_CONF_FALLBACK text). The untouched
    agent output survives only in llm_calls[].raw_output, which may be truncated, so the
    verdict is recovered by regex rather than json.loads.
    """
    for call in reversed(rec.get("llm_calls", [])):
        raw = call.get("raw_output") or ""
        if '"classify"' not in raw:
            continue
        m = re.search(r'"verdict"\s*:\s*"(attack|benign)"', raw, re.I)
        if m:
            return m.group(1).lower()
    return None


def metrics(pairs):
    tp = sum(1 for p, g in pairs if p == "attack" and g == "attack")
    fp = sum(1 for p, g in pairs if p == "attack" and g == "benign")
    fn = sum(1 for p, g in pairs if p == "benign" and g == "attack")
    tn = sum(1 for p, g in pairs if p == "benign" and g == "benign")
    acc = (tp + tn) / len(pairs) if pairs else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return acc, fpr, f1, (tp, fp, fn, tn)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(f"{'condition':24s} {'path':10s} {'n':>4} {'pre-fix Acc/FPR':>18} "
          f"{'post-fix Acc/FPR':>18} {'changed':>8}")
    print("-" * 92)
    for label, rel in RUNS:
        p = root / rel
        if not p.exists():
            print(f"{label:24s} [missing] {rel}")
            continue
        recs = [json.loads(l) for l in p.open() if l.strip()]

        # Instrument self-check: on the LOW_CONF samples the recovered agent verdict must
        # be recoverable and must differ from the post-permission record, otherwise we are
        # reading the rewritten value again and any "no change" result is an artefact.
        lc = [r for r in recs if "LOW_CONF" in (r["result"].get("reasoning") or "")]
        if lc:
            recovered = [agent_own_verdict(r) for r in lc]
            n_null = sum(1 for v in recovered if v is None)
            n_same_as_rewritten = sum(
                1 for r, v in zip(lc, recovered) if v == r["evaluation"]["binary_pred"])
            print(f"{label:24s} [instrument] LOW_CONF={len(lc)} unrecoverable={n_null} "
                  f"equal_to_rewritten={n_same_as_rewritten}")

        pre, post, n_lowconf, n_maxsteps, changed = [], [], 0, 0, 0
        for r in recs:
            gt = r["evaluation"]["binary_gt"]
            pre_v = r["evaluation"]["binary_pred"]
            reasoning = r["result"].get("reasoning") or ""
            own = agent_own_verdict(r)
            if "LOW_CONF" in reasoning:
                n_lowconf += 1
                post_v = own if own else pre_v      # gate short-circuits, agent verdict stands
            elif "MAX_STEPS" in reasoning:
                n_maxsteps += 1
                post_v = "benign"                    # explicit default, same outcome
            else:
                post_v = pre_v
            pre.append((pre_v, gt))
            post.append((post_v, gt))
            if pre_v != post_v:
                changed += 1
        a0, f0, _, cm0 = metrics(pre)
        a1, f1_, _, cm1 = metrics(post)
        path = f"L{n_lowconf}/M{n_maxsteps}"
        flag = "" if changed == 0 else f"**{changed}**"
        print(f"{label:24s} {path:10s} {len(recs):>4} "
              f"{a0*100:7.1f}% /{f0*100:6.1f}% "
              f"{a1*100:8.1f}% /{f1_*100:6.1f}% {flag:>8}")
        if changed:
            print(f"{'':24s} confusion pre  TP/FP/FN/TN = {cm0}")
            print(f"{'':24s} confusion post TP/FP/FN/TN = {cm1}")
    print("\nL = samples on the LOW_CONF path (verdict can change), "
          "M = MAX_STEPS path (verdict identical, log honesty only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
