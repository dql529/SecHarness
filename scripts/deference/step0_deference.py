"""Step 0: per-sample tool deference across every audit log. CPU only, no GPU, no re-inference.

Deference = fraction of samples where the agent's final verdict equals the binarised output of the
check_anomaly tool it called. It answers "how often is the LLM just forwarding the tool?" -- which
decides whether a deployed LLM+tool stack has one layer of defence or two.

Why this script exists: the number previously quoted as agent-tool agreement
(0.992 / 0.820 / 0.171) was actually pearsonr(cov_knn_attack, harness_zd_rate) from
run_harness_zd.py:210 -- a cross-fold RATE correlation between kNN coverage and zero-day detection,
which has nothing to do with whether the agent trusts the tool. This computes the real per-sample
quantity from the raw audit logs.

Only logs with a non-empty tool_chain can be measured. E1_rf_context_* injects the RF prediction
into the prompt instead of exposing it as a tool, so its tool_chain is empty and the RF prediction
was never persisted -- measuring that arm needs the RF model re-run against input.traffic_text
(deferred to step 1).

Usage: python3 step0_deference.py [--logs DIR] [--csv OUT]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

LOG_DIRS = ["v2", "v2_tdsc"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval. Correct at the boundaries where the normal approximation fails --
    several of these cells sit at or near 100%."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def binarise_tool(pred: str) -> str | None:
    """check_anomaly returns a class name; anything that is not Normal counts as an attack."""
    if not pred:
        return None
    p = pred.strip().lower()
    if p in ("normal", "benign"):
        return "benign"
    return "malicious"


def binarise_verdict(v: str) -> str | None:
    if not v:
        return None
    v = v.strip().lower()
    if v in ("benign", "normal"):
        return "benign"
    if v in ("malicious", "attack", "anomaly"):
        return "malicious"
    return None


def tool_prediction(rec: dict) -> str | None:
    """First check_anomaly call: the evidence the agent had when it started reasoning."""
    for step in rec.get("tool_chain") or []:
        if step.get("tool") != "check_anomaly":
            continue
        out = step.get("output")
        if isinstance(out, dict):
            return out.get("prediction")
        if isinstance(out, str):
            try:
                return json.loads(out).get("prediction")
            except json.JSONDecodeError:
                m = re.search(r'"prediction"\s*:\s*"([^"]+)"', out)
                if m:
                    return m.group(1)
        return None
    return None


def classify(name: str) -> tuple:
    """(condition, dataset) from the filename."""
    n = name.lower()
    ds = "CIC" if "cic" in n else ("UNSW" if "unsw" in n else "?")
    if n.startswith("e1_rf_context"):
        cond = "E1-RF (prompt channel)"
    elif n.startswith("e1_"):
        cond = "E1 zero-shot"
    elif n.startswith("e2_"):
        cond = "E2 finetuned"
    elif n.startswith("e3_norf"):
        cond = "E3-noRF"
    elif n.startswith("e3_degraded"):
        cond = "E3-degraded"
    elif n.startswith("e3_"):
        cond = "E3 (tool channel)"
    elif n.startswith("e4_"):
        cond = "E4 (tool channel, FT)"
    elif n.startswith("tau_"):
        cond = "tau RQ7"
    elif n.startswith("baseline"):
        cond = "baseline"
    else:
        cond = "other"
    return cond, ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path.home() / "Paper_project/SecHarness/project/logs"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.logs)
    files = sorted(f for d in LOG_DIRS for f in (root / d).glob("*.jsonl"))
    print(f"scanning {len(files)} log files under {root}\n")

    rows, skipped = [], []
    for f in files:
        agree = total = no_tool = bad = 0
        models: set = set()
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            models.add((rec.get("agent_config") or {}).get("model", "?"))
            tp = binarise_tool(tool_prediction(rec))
            av = binarise_verdict((rec.get("result") or {}).get("verdict"))
            if tp is None:
                no_tool += 1
                continue
            if av is None:
                bad += 1
                continue
            total += 1
            agree += (tp == av)
        cond, ds = classify(f.name)
        model = sorted(models)[0] if len(models) == 1 else f"MIXED({len(models)})"
        if total == 0:
            skipped.append((f.name, cond, ds, model, no_tool, bad))
            continue
        p, lo, hi = wilson(agree, total)
        rows.append({"file": f.name, "cond": cond, "dataset": ds, "model": model,
                     "n": total, "agree": agree, "deference": round(p, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "no_tool_calls": no_tool, "unparsable": bad})

    rows.sort(key=lambda r: (r["dataset"], r["cond"], r["model"]))
    print("=" * 118)
    print("PER-SAMPLE TOOL DEFERENCE  (agent final verdict == binarised check_anomaly output)")
    print("=" * 118)
    hdr = f"{'dataset':<6}{'condition':<24}{'model':<34}{'n':>5}{'defer':>8}{'95% Wilson CI':>18}"
    print(hdr)
    print("-" * len(hdr))
    last = None
    for r in rows:
        key = (r["dataset"], r["cond"])
        if last and key != last:
            print()
        last = key
        m = r["model"].replace("Qwen/", "").replace("meta-llama/", "").replace("unsloth/", "")
        print(f"{r['dataset']:<6}{r['cond']:<24}{m[:33]:<34}{r['n']:>5}"
              f"{r['deference']:>8.3f}   [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

    if skipped:
        print(f"\n{len(skipped)} logs had no measurable check_anomaly call (expected for the "
              f"prompt-channel and no-tool arms):")
        for name, cond, ds, model, nt, bad in skipped[:12]:
            print(f"  {name:<52} {cond:<24} no_tool={nt} unparsable={bad}")
        if len(skipped) > 12:
            print(f"  ... and {len(skipped) - 12} more")

    # capability sweep within the tool channel -- the claim under test
    print("\n" + "=" * 118)
    print("CAPABILITY SWEEP within the tool channel (is deference a function of scale?)")
    print("=" * 118)
    for cond in ("E3 (tool channel)", "E4 (tool channel, FT)", "tau RQ7"):
        sel = [r for r in rows if r["cond"] == cond and r["n"] >= 100]
        if len(sel) < 2:
            continue
        print(f"\n{cond}:")
        for r in sorted(sel, key=lambda r: r["model"]):
            m = r["model"].replace("Qwen/", "").replace("meta-llama/", "").replace("unsloth/", "")
            bar = "#" * int(r["deference"] * 40)
            print(f"  {m[:36]:<38}{r['n']:>5}  {r['deference']:.3f}  {bar}")
        ds = [r["deference"] for r in sel]
        print(f"  -> spread {max(ds) - min(ds):.3f} across {len(sel)} cells "
              f"(min {min(ds):.3f}, max {max(ds):.3f})")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
