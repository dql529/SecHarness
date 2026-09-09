"""Step 1a: root-cause the RF reproduction gate failure that blocks step1_prompt_channel.py.

step1's gate replays the on-disk RF pickle against the E3 tool-channel logs and matches only
96.1% (UNSW) / 87.0% (CIC), so every prompt-channel deference number derived from an offline
replay is currently suspect. The cause was previously recorded as "E3 runtime feature
pipeline inconsistent with current state; disk code/pickle/data cannot reproduce it".

That diagnosis is wrong. The pipeline reproduces the runtime EXACTLY. What step1 got wrong is
WHICH log field it replays. Two independent field-level defects make ``input.traffic_text`` a
different string from the argument check_anomaly was actually called with:

  CIC   audit.py:129 writes ``self.traffic_text[:500]``. A CIC record is 1785-1980 chars, so
        every CIC ``input.traffic_text`` is truncated mid-token (e.g. "...flow_iat_mean=very_"),
        silently deleting ~60 of 77 features, which the offline parser then fills with
        "missing". The untruncated string survives in ``tool_chain[].input.traffic_text``.

  UNSW  the argument-override at agent_loop.py:458-460 ("Always override traffic_text for
        analysis tools to prevent LLM from passing truncated or hallucinated feature strings")
        did not exist when the two oldest UNSW E3 logs were produced. In those runs the string
        the LLM typed into the tool call was dispatched verbatim, so the RF scored an
        LLM-mangled record -- dropped features, reordered keys -- while ``input.traffic_text``
        holds the true one. The argument that was really scored is, again, in
        ``tool_chain[].input.traffic_text``.

Both are recoverable because the E3 logs record the FULL check_anomaly JSON (prediction,
confidence, anomaly_score, top3), so the comparison is exact rather than binarised: this script
scores each candidate text source on byte-equality of all four fields.

It also rules out the two competing hypotheses. (1) "A different pickle ran at E3": every UNSW
config points at logs/E2_beta_ml_model.pkl and every CIC config at
logs/E2_beta_ml_cic_full_model.pkl (grep ml_model_path configs/), there is no third candidate
anywhere in the repo, and the wrong-dataset pickle is scored here as a control. (2) "batch vs
single-sample encoding": precompute() has no caller in the repo, so runtime used the one-row path
CheckAnomalyTool._predict_one while step1 used BetaAgentML.analyze_batch; a one-row DataFrame
never materialises a column for an absent kv key (beta_agent_ml.py:111-115) whereas a batch one
gets NaN then "UNK"/"missing" (:123/:140). All three paths are scored separately here.

Usage: .venv/bin/python scripts/deference/step1a_rf_drift_forensics.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from collections import Counter
from pathlib import Path

sys.modules.setdefault("requests", types.ModuleType("requests"))  # src/agents/__init__ pulls it
PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

warnings.filterwarnings("ignore")  # pickle was written by sklearn 1.7.2, venv has 1.8.0

from src.agents.beta_agent_ml import BetaAgentML, parse_kv_text  # noqa: E402
from src.v2.tools.check_anomaly import CheckAnomalyTool  # noqa: E402

LOGS = PROJ / "logs"
PKL = {"UNSW": LOGS / "E2_beta_ml_model.pkl", "CIC": LOGS / "E2_beta_ml_cic_full_model.pkl"}
GATE_LOGS = {"UNSW": LOGS / "v2/E3_unsw_sub1000_audit.jsonl",
             "CIC": LOGS / "v2/E3_cic_full_sub1000_audit.jsonl"}
# every tool-channel UNSW log, to date the argument-override fix
EXTRA_UNSW = ["v2/E3_unsw_sub200_audit.jsonl", "v2/E3_8B_unsw_sub200_audit.jsonl",
              "v2/E3_14B_unsw_sub200_audit.jsonl", "v2/E3_70B_unsw_sub200_audit.jsonl",
              "v2/E3_32B_local_unsw_sub200_audit.jsonl",
              "v2_tdsc/tau_32B_full_unsw_sub200_audit.jsonl",
              "v2_tdsc/tau_13B_full_unsw_sub200_audit.jsonl",
              "v2/E4_unsw_sub1000_audit.jsonl", "multiseed/E3_seed42_sub1000_audit.jsonl"]


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def tool_step(rec: dict) -> dict | None:
    """The first check_anomaly step. Its `input` is the argument that was really dispatched."""
    for step in rec.get("tool_chain") or []:
        if step.get("tool") == "check_anomaly":
            return step
    return None


def logged_output(step: dict) -> dict | None:
    out = step.get("output")
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None
    return None


def binarise(label: str | None) -> str | None:
    if not label:
        return None
    return "benign" if str(label).strip().lower() in ("normal", "benign") else "malicious"


FIELDS = ("prediction", "confidence", "anomaly_score", "top3")


def score(name: str, got: list[dict], logged: list[dict]) -> tuple[int, int]:
    n = len(logged)
    exact = sum(all(g.get(f) == l.get(f) for f in FIELDS) for g, l in zip(got, logged))
    lab = sum(g.get("prediction") == l.get("prediction") for g, l in zip(got, logged))
    bin_ = sum(binarise(g.get("prediction")) == binarise(l.get("prediction"))
               for g, l in zip(got, logged))
    print(f"    {name:<42} exact4={exact:>5}/{n}={exact/n:.4f}  label={lab/n:.4f}  "
          f"binary={bin_/n:.4f}")
    return exact, lab


def field_divergence(recs: list[dict]) -> None:
    """How often does input.traffic_text differ from the dispatched tool argument, and why?"""
    n = len(recs)
    diff = trunc = mangled = 0
    lost_keys: Counter = Counter()
    for r in recs:
        true_t = (r.get("input") or {}).get("traffic_text") or ""
        arg_t = (tool_step(r).get("input") or {}).get("traffic_text") or ""
        if true_t == arg_t:
            continue
        diff += 1
        if arg_t.startswith(true_t):
            trunc += 1  # the log field is a prefix of the real argument -> audit.py:129
        else:
            mangled += 1
            lost_keys.update(set(parse_kv_text(true_t)) - set(parse_kv_text(arg_t)))
    print(f"    input.traffic_text != dispatched tool argument: {diff}/{n} ({diff/n:.1%})"
          f"   [prefix-of-argument (audit truncation): {trunc}   "
          f"not-a-prefix (LLM-authored argument): {mangled}]")
    if lost_keys:
        print(f"    features the LLM dropped from the argument, most often first: "
              f"{lost_keys.most_common(6)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap samples per log (0 = all)")
    args = ap.parse_args()

    for ds in ("UNSW", "CIC"):
        pkl, log = PKL[ds], GATE_LOGS[ds]
        if not pkl.exists() or not log.exists():
            print(f"[{ds}] missing artifact, skipped")
            continue

        print("=" * 100)
        print(f"{ds}   pkl={pkl.name}   gate log={log.name}")
        print("=" * 100)

        recs = [r for r in read_jsonl(log)
                if tool_step(r) and logged_output(tool_step(r))
                and (r.get("input") or {}).get("traffic_text")]
        if args.limit:
            recs = recs[: args.limit]
        logged = [logged_output(tool_step(r)) for r in recs]
        true_texts = [r["input"]["traffic_text"] for r in recs]          # what step1 replays
        arg_texts = [tool_step(r)["input"]["traffic_text"] for r in recs]  # what really ran
        print(f"  records with a logged check_anomaly output: {len(recs)}")
        print(f"  len(input.traffic_text)          min={min(map(len, true_texts))} "
              f"max={max(map(len, true_texts))}   (audit.py:129 caps this field at 500)")
        print(f"  len(tool_chain[].input.traffic_text) min={min(map(len, arg_texts))} "
              f"max={max(map(len, arg_texts))}")
        field_divergence(recs)

        agent = BetaAgentML.load(str(pkl))
        print(f"  model: {len(agent.cat_features)} categorical + {len(agent.num_features)} "
              f"binned-numeric features; classes={list(agent.label_encoder.classes_)}")

        # was the runtime even a deterministic function of its argument?
        by_arg: dict[str, set] = {}
        for r in recs:
            s = tool_step(r)
            by_arg.setdefault(s["input"]["traffic_text"], set()).add(s.get("output"))
        nondet = sum(1 for v in by_arg.values() if len(v) > 1)
        print(f"  determinism: {len(by_arg)} distinct arguments, {nondet} of them logged more "
              f"than one distinct output  ({'DETERMINISTIC' if nondet == 0 else 'NONDETERMINISTIC'})")

        print("\n  replay of the on-disk pickle, by text source and code path:")
        for src_name, texts in (("input.traffic_text  [step1 uses this]", true_texts),
                                ("tool_chain[].input  [really dispatched]", arg_texts)):
            tool = CheckAnomalyTool(str(pkl), degraded=False)
            single = [json.loads(tool(t)) for t in texts]
            score(f"single _predict_one | {src_name}", single, logged)
        # batch paths, on the correct source only
        tool2 = CheckAnomalyTool(str(pkl), degraded=False)
        tool2.precompute(arg_texts)
        pre = [json.loads(tool2._cache[hash(t)]) for t in arg_texts]
        score("precompute batch  | tool_chain[].input", pre, logged)
        bv = agent.analyze_batch(arg_texts)
        batch = [{"prediction": (v.attack_type if v.verdict == "attack"
                                 else list(agent.label_encoder.classes_)[
                                     [c.lower() in ("normal", "benign")
                                      for c in agent.label_encoder.classes_].index(True)]),
                  "confidence": round(v.confidence, 3)} for v in bv]
        lab = sum(g["prediction"] == l["prediction"] for g, l in zip(batch, logged))
        print(f"    {'analyze_batch     | tool_chain[].input':<42} "
              f"label={lab/len(logged):.4f}   (no anomaly_score/top3 on this path)")

        # control: the other dataset's pickle must be far worse
        other = "CIC" if ds == "UNSW" else "UNSW"
        if PKL[other].exists():
            try:
                oth = BetaAgentML.load(str(PKL[other]))
                ov = oth.analyze_batch(arg_texts)
                names = list(oth.label_encoder.classes_)
                norm = names[[c.lower() in ("normal", "benign") for c in names].index(True)]
                og = [(v.attack_type if v.verdict == "attack" else norm) for v in ov]
                m = sum(a == l["prediction"] for a, l in zip(og, logged))
                print(f"    {'CONTROL: ' + other + ' pickle on ' + ds + ' text':<42} "
                      f"label={m/len(logged):.4f}   (must be far worse; confirms model identity)")
            except Exception as exc:  # noqa: BLE001 -- wrong-schema pickle is an expected outcome
                print(f"    CONTROL: {other} pickle -> {type(exc).__name__}: {exc}")

        # residual mismatches on the winning source
        tool3 = CheckAnomalyTool(str(pkl), degraded=False)
        best = [json.loads(tool3(t)) for t in arg_texts]
        bad = [i for i, (g, l) in enumerate(zip(best, logged))
               if any(g.get(f) != l.get(f) for f in FIELDS)]
        print(f"\n  residual exact-match failures on tool_chain[].input: {len(bad)}/{len(logged)}")
        for i in bad[:5]:
            print(f"    idx={recs[i].get('sample_index')}  runtime={json.dumps(logged[i])}")
            print(f"                recovered={json.dumps(best[i])}")
        print()

    # ---- how far does the LLM-authored-argument defect spread? --------------------------------
    print("=" * 100)
    print("SPREAD OF THE LLM-AUTHORED-ARGUMENT DEFECT across UNSW tool-channel logs")
    print("(agent_loop.py:458-460 forces the true text into the argument; logs written before")
    print(" that line existed dispatched whatever the LLM typed)")
    print("=" * 100)
    agent = BetaAgentML.load(str(PKL["UNSW"]))
    hdr = (f"  {'log':<46}{'mtime':<18}{'n':>5}{'LLM-authored':>14}"
           f"{'match@input':>12}{'match@arg':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rel in EXTRA_UNSW:
        p = LOGS / rel
        if not p.exists():
            print(f"  {rel:<46}MISSING")
            continue
        recs = [r for r in read_jsonl(p)
                if tool_step(r) and logged_output(tool_step(r))
                and (r.get("input") or {}).get("traffic_text")]
        if not recs:
            print(f"  {rel:<46}no measurable check_anomaly step")
            continue
        if args.limit:
            recs = recs[: args.limit]
        logged = [logged_output(tool_step(r)) for r in recs]
        true_t = [r["input"]["traffic_text"] for r in recs]
        arg_t = [tool_step(r)["input"]["traffic_text"] for r in recs]
        mang = sum(1 for a, b in zip(true_t, arg_t) if a != b and not b.startswith(a))
        t1 = CheckAnomalyTool(str(PKL["UNSW"]), degraded=False)
        m1 = sum(all(json.loads(t1(t)).get(f) == l.get(f) for f in FIELDS)
                 for t, l in zip(true_t, logged))
        t2 = CheckAnomalyTool(str(PKL["UNSW"]), degraded=False)
        m2 = sum(all(json.loads(t2(t)).get(f) == l.get(f) for f in FIELDS)
                 for t, l in zip(arg_t, logged))
        n = len(recs)
        mt = __import__("datetime").datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")
        print(f"  {Path(rel).name:<46}{mt:<18}{n:>5}{mang/n:>13.1%}"
              f"{m1/n:>12.4f}{m2/n:>11.4f}")


if __name__ == "__main__":
    main()
