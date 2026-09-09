"""Step 1: recover tool deference for the PROMPT channel by re-running the RF offline.

Step 0 could only measure the tool channel (E3/E4/tau): those logs record the check_anomaly output
in tool_chain. The prompt-channel arm (E1_rf_context_*) injects the RF prediction into the prompt
instead of exposing it as a tool, so tool_chain is empty and the RF prediction was never persisted.

The RF is deterministic and the logs keep input.traffic_text, so the prediction can be recovered
exactly. This matters because the only existing numbers for this arm carry no provenance (no
generating script) -- they must not be cited until reproduced.

Gate first: verify that check_anomaly in the tool channel IS this same RF. If the two channels use
different tools, the channel comparison is meaningless and everything downstream is void.

Usage: .venv/bin/python scripts/deference/step1_prompt_channel.py [--csv OUT]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import types
from pathlib import Path

sys.modules.setdefault("requests", types.ModuleType("requests"))  # src/agents/__init__ pulls it
PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from src.agents.beta_agent_ml import BetaAgentML  # noqa: E402

LOGS = PROJ / "logs"
RF_UNSW = LOGS / "E2_beta_ml_model.pkl"
RF_CIC = LOGS / "E2_beta_ml_cic_full_model.pkl"


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def binv(v) -> str | None:
    if not v:
        return None
    v = str(v).strip().lower()
    if v in ("benign", "normal"):
        return "benign"
    if v in ("malicious", "attack", "anomaly"):
        return "malicious"
    return None


def bintool(p) -> str | None:
    if not p:
        return None
    return "benign" if str(p).strip().lower() in ("normal", "benign") else "malicious"


def read(path: Path) -> list:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tool_pred(rec: dict):
    for s in rec.get("tool_chain") or []:
        if s.get("tool") == "check_anomaly":
            o = s.get("output")
            if isinstance(o, dict):
                return o.get("prediction")
            if isinstance(o, str):
                try:
                    return json.loads(o).get("prediction")
                except json.JSONDecodeError:
                    m = re.search(r'"prediction"\s*:\s*"([^"]+)"', o)
                    return m.group(1) if m else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print("loading RF models...")
    rf = {"UNSW": BetaAgentML.load(str(RF_UNSW))}
    if RF_CIC.exists():
        rf["CIC"] = BetaAgentML.load(str(RF_CIC))
    print(f"  loaded: {', '.join(rf)}\n")

    # ---- GATE: is check_anomaly the same RF? -------------------------------------------------
    print("=" * 104)
    print("GATE  does the tool channel's check_anomaly agree with this RF? "
          "(if not, the channel comparison is void)")
    print("=" * 104)
    gate_ok = True
    for name, ds in (("E3_unsw_sub1000_audit.jsonl", "UNSW"),
                     ("E3_cic_full_sub1000_audit.jsonl", "CIC")):
        p = LOGS / "v2" / name
        if not p.exists() or ds not in rf:
            continue
        recs = [r for r in read(p) if tool_pred(r) and r.get("input", {}).get("traffic_text")]
        if not recs:
            continue
        texts = [r["input"]["traffic_text"] for r in recs]
        mine = [binv(v.verdict) for v in rf[ds].analyze_batch(texts)]
        logged = [bintool(tool_pred(r)) for r in recs]
        agree = sum(a == b for a, b in zip(mine, logged))
        pct = agree / len(recs)
        flag = "OK" if pct >= 0.99 else "MISMATCH"
        if pct < 0.99:
            gate_ok = False
        print(f"  {ds:<5} {name:<38} n={len(recs):<5} match={pct:.4f}  [{flag}]")
    if not gate_ok:
        print("\n  ! check_anomaly does not reproduce from this RF pickle. The prompt-vs-tool")
        print("    channel comparison below is NOT valid until this is resolved.")
    print()

    # ---- prompt channel ----------------------------------------------------------------------
    print("=" * 104)
    print("PROMPT CHANNEL  (E1_rf_context_*): RF prediction injected into the prompt, recovered offline")
    print("=" * 104)
    hdr = f"{'model':<34}{'ds':<6}{'n':>6}{'defer':>8}{'95% Wilson CI':>20}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for p in sorted((LOGS / "v2").glob("E1_rf_context_*.jsonl")):
        recs = [r for r in read(p) if r.get("input", {}).get("traffic_text")]
        if len(recs) < 10:
            continue
        ds = "CIC" if "cic" in p.name.lower() else "UNSW"
        if ds not in rf:
            continue
        models = {(r.get("agent_config") or {}).get("model", "?") for r in recs}
        model = sorted(models)[0] if len(models) == 1 else f"MIXED({len(models)})"
        preds = [binv(v.verdict) for v in rf[ds].analyze_batch([r["input"]["traffic_text"] for r in recs])]
        verds = [binv((r.get("result") or {}).get("verdict")) for r in recs]
        pairs = [(a, b) for a, b in zip(preds, verds) if a and b]
        agree = sum(a == b for a, b in pairs)
        pr, lo, hi = wilson(agree, len(pairs))
        m = model.replace("Qwen/", "").replace("meta-llama/", "").replace("unsloth/", "")
        print(f"{m[:33]:<34}{ds:<6}{len(pairs):>6}{pr:>8.3f}   [{lo:.3f}, {hi:.3f}]")
        rows.append({"channel": "prompt", "file": p.name, "model": model, "dataset": ds,
                     "n": len(pairs), "agree": agree, "deference": round(pr, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    print("\nCompare against the tool-channel numbers from step0_deference.py. If the prompt column")
    print("spreads across models while the tool column stays flat, the tool channel is flattening")
    print("model differences -- i.e. a stronger backbone buys no extra independent judgement once")
    print("the evidence arrives through a tool call.")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
