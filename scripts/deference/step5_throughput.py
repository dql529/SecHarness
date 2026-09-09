"""Step 5: empirical inference throughput from existing audit logs, a two-proportion power
calculation, and an A800-hour cost projection for a follow-up scale study. CPU only, no GPU,
no re-inference -- every number in TASK A is read directly off logs that already exist on disk.

Why this script exists: sizing an A800 budget for a follow-up model-scale study needs real
samples/hour per model-size class. The audit logs under logs/v2/ and logs/v2_tdsc/ already
contain a timestamp per record and a full llm_calls[] breakdown (latency_ms, input_tokens,
output_tokens per call), so throughput can be measured empirically instead of guessed.

The one thing that makes this non-trivial: agent_config.model in the log is NOT a reliable
serving-stack signal by itself. src/v2/llm_engine.py:56 shows the actual dispatch rule --
`self._api_mode = base_model.startswith("api://")` -- and when api mode is on, the recorded
model string is whatever comes after host:port/, which can look exactly like a bare HF repo id
(e.g. "Qwen/Qwen2.5-32B-Instruct-AWQ" recorded from api://localhost:8000/Qwen/Qwen2.5-32B...).
A *bare* "Qwen/Qwen2.5-7B-Instruct" with no api:// prefix at all loads through HuggingFace
transformers locally (this machine has zero GPU, so that is Mac inference, not A800 inference) --
and that string is indistinguishable from the A800 case by looking at the log alone. This script
resolves the ambiguity by reading model.base out of every configs/v2*/**/*.yaml and matching it
against the model string recorded in each log, then buckets every log into one of three stacks:
Ollama-Mac (api://localhost:11434), vLLM-A800-GPU (api://localhost:8000, the AutoDL A800 vLLM
port), or HF-transformers-Mac (no api:// prefix). Only the middle one prices A800 hours; the
other two are Mac-local and must never be averaged into a GPU cost estimate.

TASK A: per-log and per-size-class throughput, with gap detection for paused/resumed runs.
TASK B: two-proportion two-sided power calculation (required n, and achieved power at n=200).
TASK C: A800-hour cost table for a 4-model matrix, combining A's GPU-served throughput and B's n.

Usage: .venv/bin/python scripts/deference/step5_throughput.py [--logs DIR] [--configs DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

LOG_DIRS = ["v2", "v2_tdsc"]
CONFIG_DIRS = ["v2", "v2_tdsc"]

# ---------------------------------------------------------------------------
# recorded agent_config.model string -> (size_class, family_label)
# Built from a full manual survey of every distinct model string across logs/v2/*.jsonl and
# logs/v2_tdsc/*.jsonl (see the "SCHEMA / MODEL SURVEY" section this script prints at runtime
# for the live version of this list). Unknown strings are NOT silently dropped -- see
# analyze_log(), which flags them as unresolved instead of guessing a class.
# ---------------------------------------------------------------------------
SIZE_CLASS: dict[str, tuple[str, str]] = {
    "unsloth/Llama-3.2-3B-Instruct": ("3B", "Llama-3.2"),
    "Qwen/Qwen2.5-7B-Instruct": ("7B", "Qwen2.5"),
    "llama3.1:8b": ("8B", "Llama-3.1"),
    "unsloth/Meta-Llama-3.1-8B-Instruct": ("8B", "Llama-3.1"),
    "gemma4:e4b": ("12B/14B", "Gemma4 [tag=e4b -- Google's 'effective-4B' elastic notation, "
                              "NOT a dense 12B despite the E1_rf_context_12B_unsw filename]"),
    "qwen2.5:14b": ("12B/14B", "Qwen2.5"),
    "Qwen/Qwen2.5-14B-Instruct": ("12B/14B", "Qwen2.5"),
    "Qwen/Qwen2.5-14B-Instruct-AWQ": ("12B/14B", "Qwen2.5-AWQ"),
    "gemma4:31b": ("31B/32B", "Gemma4"),
    "qwen2.5:32b": ("31B/32B", "Qwen2.5"),
    "Qwen/Qwen2.5-32B-Instruct-AWQ": ("31B/32B", "Qwen2.5-AWQ"),
    "casperhansen/llama-3.3-70b-instruct-awq": ("70B", "Llama-3.3-AWQ"),
    "llama-3.3-70b-instruct-awq": ("70B", "Llama-3.3-AWQ"),
}
CLASS_ORDER = ["3B", "7B", "8B", "12B/14B", "31B/32B", "70B"]
# representative parameter count per class, used only to pick the nearest class for TASK C's
# explicitly-labelled proxy substitution when a class has no GPU-served evidence at all.
CLASS_PARAMS = {"3B": 3, "7B": 7, "8B": 8, "12B/14B": 13, "31B/32B": 31.5, "70B": 70}
NON_LLM_MODELS = {"RandomForest"}  # classical baseline, not an LLM -- reported, never classed

STACK_OLLAMA = "Ollama-Mac"
STACK_VLLM_A800 = "vLLM-A800-GPU"
STACK_HF_LOCAL = "HF-transformers-Mac"
BASE_RE = re.compile(r'^\s*base:\s*"([^"]*)"\s*$', re.MULTILINE)


def fmt(x, spec: str = "{:.2f}", na: str = "n/a") -> str:
    return na if x is None else spec.format(x)


def fmt_rate(x, na: str = "n/a") -> str:
    return na if x is None else f"{x:,.1f}"


def hours(x: float | None) -> float | None:
    """Seconds -> hours, None-safe (avoids the `x and x/3600` short-circuit footgun where a
    genuine 0.0 span happens to survive only because 0/3600 is also 0)."""
    return None if x is None else x / 3600


# ---------------------------------------------------------------------------
# TASK A
# ---------------------------------------------------------------------------

def load_stack_table(configs_root: Path) -> tuple[dict[str, tuple[str, str, str]], int]:
    """model_name -> (stack_label, raw model.base string, source config path).

    Ground truth for the split is src/v2/llm_engine.py:56 `base_model.startswith("api://")`.
    api://host:port/model -> API backend (host:port 11434 = Ollama, 8000 = the AutoDL A800 vLLM
    port per this user's standing device notes); no api:// prefix -> HuggingFace transformers
    loaded locally in-process (Mac, since this machine has zero GPU).
    """
    table: dict[str, tuple[str, str, str]] = {}
    n_yaml = 0
    for d in CONFIG_DIRS:
        for path in sorted((configs_root / d).rglob("*.yaml")):
            n_yaml += 1
            m = BASE_RE.search(path.read_text())
            if not m or not m.group(1) or m.group(1) == "null":
                continue
            base = m.group(1)
            rel = str(path.relative_to(configs_root.parent))
            if base.startswith("api://"):
                uri = base[len("api://"):]
                slash = uri.find("/")
                if slash == -1:
                    continue
                host_port, model_name = uri[:slash], uri[slash + 1:]
                if host_port.endswith(":11434"):
                    stack = STACK_OLLAMA
                elif host_port.endswith(":8000"):
                    stack = STACK_VLLM_A800
                else:
                    stack = f"UNKNOWN-API-PORT({host_port})"
            else:
                model_name, stack = base, STACK_HF_LOCAL
            prev = table.get(model_name)
            if prev and prev[0] != stack:
                print(f"WARNING: conflicting stack for '{model_name}': "
                      f"{prev[0]} ({prev[2]}) vs {stack} ({rel}) -- keeping first")
                continue
            table[model_name] = (stack, base, rel)
    return table, n_yaml


def analyze_log(path: Path, stack_table: dict[str, tuple[str, str, str]]) -> dict:
    """One log file -> a stats dict. Every field that cannot be computed is left absent/None
    and the reason is appended to row['unresolved'] -- callers must use .get() throughout."""
    raw_lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    records, bad = [], 0
    for ln in raw_lines:
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    row: dict = {"file": path.name, "dir": path.parent.name, "n": len(records),
                 "n_bad_json": bad, "unresolved": []}
    n = len(records)
    if n == 0:
        row["unresolved"].append("empty file (0 parsable JSON records)")
        return row

    models = sorted({(r.get("agent_config") or {}).get("model", "?") for r in records})
    row["models"] = models
    model_key = models[0] if len(models) == 1 else None
    if model_key is None:
        row["unresolved"].append(f"mixed/inconsistent model strings in one file: {models}")
        row["stack"] = f"MIXED/UNKNOWN{tuple(models)}"
    elif model_key == "?":
        row["unresolved"].append("no agent_config.model field on any record")
        row["stack"] = "UNKNOWN (no agent_config field)"
    elif model_key in NON_LLM_MODELS:
        row["class_label"] = "N/A (non-LLM baseline)"
        row["family"] = model_key
        row["stack"] = "N/A (classical ML, not served)"
    else:
        cls = SIZE_CLASS.get(model_key)
        if cls is None:
            row["unresolved"].append(f"model string '{model_key}' not in the SIZE_CLASS table")
        else:
            row["class_label"], row["family"] = cls
        info = stack_table.get(model_key)
        if info:
            row["stack"], row["stack_config"] = info[0], info[2]
        else:
            row["stack"] = f"UNKNOWN (no configs/**/*.yaml model.base matches '{model_key}')"
            row["unresolved"].append(f"could not cross-check serving stack for '{model_key}'")

    # --- wall-clock span, gap detection, gap-trimmed span (source: record['timestamp']) ---
    ts_all = [r.get("timestamp") for r in records if r.get("timestamp") is not None]
    row["has_timestamp"] = len(ts_all) > 0
    if not ts_all:
        row["unresolved"].append(
            "no `timestamp` field on any record -- wall-clock span / samples-per-hour cannot "
            "be computed from this log; would need an external run-log with start/end wall time")
    else:
        if len(ts_all) < n:
            row["unresolved"].append(f"only {len(ts_all)}/{n} records carry `timestamp`; "
                                      f"span computed from that subset")
        ts = sorted(ts_all)
        raw_span = ts[-1] - ts[0]
        row["raw_span_s"] = raw_span
        if len(ts) >= 2:
            diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            row["median_wall_s"] = statistics.median(diffs)
            row["mean_wall_s"] = statistics.mean(diffs)
            med_diff = row["median_wall_s"]
            gaps = [d for d in diffs if med_diff > 0 and d > 10 * med_diff]
            row["n_gaps"] = len(gaps)
            row["gap_total_s"] = sum(gaps)
            row["gap_trimmed_span_s"] = raw_span - row["gap_total_s"]
            if gaps:
                row["unresolved"].append(
                    f"{len(gaps)} gap(s) > 10x median inter-record delta totalling "
                    f"{row['gap_total_s']:.0f}s excluded from gap-trimmed span (paused/resumed run)")
        else:
            row["n_gaps"], row["gap_total_s"], row["gap_trimmed_span_s"] = 0, 0.0, raw_span
            row["unresolved"].append("only 1 timestamped record -- span/gap stats degenerate")
        row["raw_per_hour"] = (n / (raw_span / 3600)) if raw_span > 0 else None
        gts = row.get("gap_trimmed_span_s")
        row["gaptrim_per_hour"] = (n / (gts / 3600)) if gts and gts > 0 else None

    # --- per-call latency / tokens (source: record['llm_calls'][].{latency_ms,in,out}) ---
    has_calls_field = any("llm_calls" in r for r in records)
    row["has_llm_calls"] = has_calls_field
    if not has_calls_field:
        row["unresolved"].append(
            "no `llm_calls` field on any record -- cannot derive per-call latency/token/"
            "tokens-per-sec stats for this log")
    else:
        lat_sums, n_calls_l, out_sums, in_sums, tok_per_sec = [], [], [], [], []
        for r in records:
            calls = r.get("llm_calls") or []
            lat = sum((c.get("latency_ms") or 0) for c in calls)
            out = sum((c.get("output_tokens") or 0) for c in calls)
            inp = sum((c.get("input_tokens") or 0) for c in calls)
            lat_sums.append(lat)
            n_calls_l.append(len(calls))
            out_sums.append(out)
            in_sums.append(inp)
            if lat > 0:
                tok_per_sec.append(out / (lat / 1000))
        row["med_llm_latency_ms"] = statistics.median(lat_sums)
        row["med_n_calls"] = statistics.median(n_calls_l)
        row["med_out_tokens"] = statistics.median(out_sums)
        row["med_in_tokens"] = statistics.median(in_sums)
        row["med_tokens_per_sec"] = statistics.median(tok_per_sec) if tok_per_sec else None
        if not tok_per_sec:
            row["unresolved"].append("all records have zero llm latency -- tokens/sec undefined")

        # cheap cross-check against the precomputed efficiency.llm_latency_ms (sanity, not a
        # second source of truth -- TASK A numbers come from llm_calls[] per the task spec)
        eff = [(r.get("efficiency") or {}).get("llm_latency_ms") for r in records]
        pairs = [(a, b) for a, b in zip(lat_sums, eff) if b is not None]
        mism = sum(1 for a, b in pairs if abs(a - b) > max(1.0, 0.01 * max(a, b, 1.0)))
        if mism:
            row["unresolved"].append(
                f"{mism}/{len(pairs)} records: sum(llm_calls[].latency_ms) disagrees with "
                f"efficiency.llm_latency_ms by >1% -- used the former (task spec) but flagging it")
    return row


def aggregate_by_class_stack(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(class_label, stack) -> {median_per_hour, files, n_files}, gap-trimmed samples/hour only."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cls = r.get("class_label")
        if cls is None or cls not in CLASS_ORDER:
            continue
        if r.get("gaptrim_per_hour") is None:
            continue
        buckets.setdefault((cls, r["stack"]), []).append(r)
    out = {}
    for key, sub in buckets.items():
        vals = [r["gaptrim_per_hour"] for r in sub]
        out[key] = {
            "median_per_hour": statistics.median(vals),
            "min_per_hour": min(vals),
            "max_per_hour": max(vals),
            "files": [r["file"] for r in sub],
            "n_files": len(sub),
        }
    return out


def print_task_a(logs_root: Path, configs_root: Path) -> dict[tuple[str, str], dict]:
    print("#" * 100)
    print("TASK A -- EMPIRICAL THROUGHPUT FROM AUDIT LOGS")
    print("#" * 100)

    files = sorted(f for d in LOG_DIRS for f in (logs_root / d).glob("*.jsonl"))
    stack_table, n_yaml = load_stack_table(configs_root)
    print(f"\nscanning {len(files)} log files under {logs_root}")
    print(f"cross-checked {n_yaml} config yamls under {configs_root} "
          f"(configs/v2/**/*.yaml, configs/v2_tdsc/**/*.yaml) for model.base -> serving stack")

    # schema check -- read straight from data, do not assume the shape (per task instructions)
    sample_path = next((f for f in files if f.stat().st_size > 0), None)
    if sample_path:
        with open(sample_path) as fh:
            sample_rec = next(json.loads(ln) for ln in fh if ln.strip())
        print(f"\nSCHEMA CHECK -- first record of {sample_path.relative_to(logs_root.parent)}")
        print(f"  top-level keys : {list(sample_rec.keys())}")
        print(f"  efficiency keys: {list((sample_rec.get('efficiency') or {}).keys())}")
        lc = sample_rec.get("llm_calls") or []
        print(f"  llm_calls[0] keys: {list(lc[0].keys()) if lc else '(empty in this record)'}")

    print("\nfield provenance: n = parsable JSON lines; model = agent_config.model; "
          "span = max(timestamp)-min(timestamp); gap = inter-record diff > 10x median diff; "
          "llm latency/tokens = sum over llm_calls[] per record; "
          "stack = configs/**/*.yaml model.base cross-check (see src/v2/llm_engine.py:56)")

    rows = [analyze_log(f, stack_table) for f in files]

    classed = sorted(
        (r for r in rows if r.get("class_label") in CLASS_ORDER),
        key=lambda r: (CLASS_ORDER.index(r["class_label"]), r.get("stack") or "", r["file"]),
    )
    other = [r for r in rows if r.get("class_label") not in CLASS_ORDER]

    # --- Table A1: span / throughput ---
    print("\n" + "=" * 118)
    print("TABLE A1 -- WALL-CLOCK SPAN & THROUGHPUT (gap = inter-record delta > 10x median)")
    print("=" * 118)
    hdr = (f"{'file':<46}{'class':<9}{'stack':<14}{'n':>5}{'raw_hr':>9}{'gtrim_hr':>10}"
           f"{'gaps':>6}{'raw/hr':>11}{'gtrim/hr':>11}")
    print(hdr)
    print("-" * len(hdr))
    last_cls = None
    for r in classed:
        if last_cls is not None and r["class_label"] != last_cls:
            print()
        last_cls = r["class_label"]
        stack_short = r.get("stack", "?").replace(STACK_VLLM_A800, "vLLM-A800").replace(
            STACK_OLLAMA, "Ollama-Mac").replace(STACK_HF_LOCAL, "HF-local-Mac")
        print(f"{r['file'][:45]:<46}{r['class_label']:<9}{stack_short[:13]:<14}{r['n']:>5}"
              f"{fmt(hours(r.get('raw_span_s'))):>9}"
              f"{fmt(hours(r.get('gap_trimmed_span_s'))):>10}"
              f"{r.get('n_gaps', 0):>6}{fmt_rate(r.get('raw_per_hour')):>11}"
              f"{fmt_rate(r.get('gaptrim_per_hour')):>11}")

    if other:
        print("\n-- non-LLM / unresolved-class logs (excluded from class table above; "
              "these do not fit the fixed-width table, printed as key=value instead) --")
        for r in other:
            tag = r.get("class_label", "UNRESOLVED")
            stack = r.get("stack", "?")
            print(f"  {r['file']}  n={r['n']}  class={tag}  stack={stack}  "
                  f"raw_hr={fmt(hours(r.get('raw_span_s')))}  "
                  f"gtrim_hr={fmt(hours(r.get('gap_trimmed_span_s')))}  "
                  f"gaps={r.get('n_gaps', 0)}  raw/hr={fmt_rate(r.get('raw_per_hour'))}  "
                  f"gtrim/hr={fmt_rate(r.get('gaptrim_per_hour'))}")

    # --- Table A2: latency / tokens ---
    print("\n" + "=" * 118)
    print("TABLE A2 -- PER-RECORD LATENCY & TOKENS (all medians)")
    print("=" * 118)
    hdr2 = (f"{'file':<46}{'med_wall_s':>11}{'mean_wall_s':>12}{'llm_lat_ms':>11}"
            f"{'n_calls':>8}{'out_tok':>9}{'in_tok':>8}{'tok/s':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    last_cls = None
    for r in classed:
        if last_cls is not None and r["class_label"] != last_cls:
            print()
        last_cls = r["class_label"]
        print(f"{r['file'][:45]:<46}{fmt(r.get('median_wall_s')):>11}"
              f"{fmt(r.get('mean_wall_s')):>12}{fmt(r.get('med_llm_latency_ms'), '{:.0f}'):>11}"
              f"{fmt(r.get('med_n_calls'), '{:.1f}'):>8}"
              f"{fmt(r.get('med_out_tokens'), '{:.0f}'):>9}"
              f"{fmt(r.get('med_in_tokens'), '{:.0f}'):>8}"
              f"{fmt(r.get('med_tokens_per_sec')):>9}")

    # --- unresolved summary ---
    unresolved_rows = [r for r in rows if r.get("unresolved")]
    print(f"\n{len(unresolved_rows)}/{len(rows)} log files have at least one UNRESOLVED note:")
    for r in unresolved_rows:
        print(f"  {r['file']}:")
        for u in r["unresolved"]:
            print(f"    - {u}")

    # --- stack legend, from data ---
    print("\n" + "=" * 118)
    print("SERVING-STACK DETERMINATION (configs/**/*.yaml model.base, api:// dispatch = "
          "src/v2/llm_engine.py:56)")
    print("=" * 118)
    seen = set()
    for r in rows:
        key = (tuple(r.get("models") or []), r.get("stack"))
        if key in seen or not r.get("models"):
            continue
        seen.add(key)
        cfg = r.get("stack_config", "-")
        print(f"  {r['models'][0]:<42} -> {r.get('stack','?'):<44} (config: {cfg})")

    # --- class aggregation ---
    agg = aggregate_by_class_stack(rows)
    print("\n" + "=" * 118)
    print("AGGREGATE BY MODEL-SIZE CLASS -- median gap-trimmed samples/hour, STACKS KEPT SEPARATE")
    print("=" * 118)
    print("(never average Ollama-Mac / HF-local-Mac / vLLM-A800-GPU together -- only vLLM-A800-GPU")
    print(" rows below are valid for pricing A800 hours in TASK C)\n")
    for cls in CLASS_ORDER:
        stacks_here = sorted({k[1] for k in agg if k[0] == cls})
        if not stacks_here:
            print(f"{cls}: UNRESOLVED -- no log in logs/v2*/*.jsonl classified into this size "
                  f"class at all. Would need a run of a {cls} model logged with the v2 audit schema.")
            continue
        print(f"{cls}:")
        for stack in stacks_here:
            info = agg[(cls, stack)]
            print(f"  {stack:<20} median={fmt_rate(info['median_per_hour'])}/hr  "
                  f"(range {fmt_rate(info['min_per_hour'])}-{fmt_rate(info['max_per_hour'])}, "
                  f"n_files={info['n_files']})")
            print(f"    backed by: {', '.join(info['files'])}")
        gpu_present = any(s == STACK_VLLM_A800 for s in stacks_here)
        if not gpu_present:
            print(f"    UNRESOLVED: no GPU-served (vLLM-A800) log for class {cls}. "
                  f"Would need a run of this class through the api://localhost:8000 vLLM endpoint.")
        print()
    print("NOTE on 12B/14B composition: this class mixes gemma4:e4b (Ollama; Google's "
          "'effective-4B' elastic tag, not a dense 12B) with true ~14B dense models "
          "(qwen2.5:14b Ollama; Qwen2.5-14B-Instruct[-AWQ]). Check the backing-file list above, "
          "not just the median, if the distinction matters.")
    print("NOTE on per-class medians generally: each class's backing files mix harness "
          "conditions (E1 direct-verdict / single LLM call vs E3-E4/tau multi-step ReAct tool "
          "loops), which differ in samples/hour by roughly an order of magnitude on their own -- "
          "the class median is a rough central tendency across conditions, not a controlled "
          "single-condition measurement.")

    return agg


# ---------------------------------------------------------------------------
# TASK B -- two-proportion two-sided power calculation
# ---------------------------------------------------------------------------

# z_{1-alpha/2}=1.959964 and z_{1-beta}=0.841621 are standard normal quantiles
# (== scipy.stats.norm.ppf(0.975) and ppf(0.80)): the two-sided alpha=0.05 critical value
# (each tail alpha/2=0.025) and the power=0.80 (beta=0.20) quantile used in the standard
# two-proportion sample-size formula (e.g. Fleiss, Levin & Paik, "Statistical Methods for
# Rates and Proportions"; Chow, Shao & Wang, "Sample Size Calculations in Clinical Research").
Z_ALPHA2 = 1.959964
Z_BETA = 0.841621

N_FORMULA = ("n_per_group = ( z_(1-a/2)*sqrt(2*pbar*(1-pbar)) "
             "+ z_(1-b)*sqrt(p1*(1-p1)+p2*(1-p2)) )^2 / (p1-p2)^2,  pbar=(p1+p2)/2")
POWER_FORMULA = ("power = Phi( (|p1-p2|*sqrt(n) - z_(1-a/2)*sqrt(2*pbar*(1-pbar))) "
                  "/ sqrt(p1*(1-p1)+p2*(1-p2)) ),  Phi = standard normal CDF")


def phi(x: float) -> float:
    """Standard normal CDF via math.erf (stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_n(p1: float, p2: float) -> float:
    pbar = (p1 + p2) / 2.0
    num = Z_ALPHA2 * math.sqrt(2 * pbar * (1 - pbar)) + Z_BETA * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2))
    return (num ** 2) / ((p1 - p2) ** 2)


def achieved_power(p1: float, p2: float, n: int) -> float:
    pbar = (p1 + p2) / 2.0
    delta = abs(p1 - p2)
    se1 = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    z_b = (delta * math.sqrt(n) - Z_ALPHA2 * math.sqrt(2 * pbar * (1 - pbar))) / se1
    return phi(z_b)


TASK_B_CASES = [
    ("(i)   observed 13pp gap  ", 0.990, 0.860),
    ("(ii)  5pp gap, high end  ", 0.990, 0.940),
    ("(iii) 5pp gap, mid-scale ", 0.910, 0.860),
]


def print_task_b() -> dict[str, int]:
    print("\n" + "#" * 100)
    print("TASK B -- TWO-PROPORTION TWO-SIDED POWER CALCULATION (alpha=0.05, power=0.80)")
    print("#" * 100)
    print(f"\nz_(1-alpha/2) = {Z_ALPHA2}  (Phi^-1(0.975), two-sided alpha=0.05)")
    print(f"z_(1-beta)    = {Z_BETA}  (Phi^-1(0.80), power=0.80)")
    print(f"\nformula: {N_FORMULA}\n")

    n_ceils: dict[str, int] = {}
    hdr = f"{'case':<28}{'p1':>7}{'p2':>7}{'gap_pp':>8}{'n_per_group (exact)':>22}{'n_per_group (ceil)':>21}"
    print(hdr)
    print("-" * len(hdr))
    for label, p1, p2 in TASK_B_CASES:
        n_exact = two_proportion_n(p1, p2)
        n_ceil = math.ceil(n_exact)
        n_ceils[label.strip()] = n_ceil
        print(f"{label:<28}{p1:>7.3f}{p2:>7.3f}{(p1-p2)*100:>7.1f}p{n_exact:>22.2f}{n_ceil:>21d}")

    label_i, p1_i, p2_i = TASK_B_CASES[0]
    print(f"\nachieved power at n=200 for case (i) [p1={p1_i}, p2={p2_i}]:")
    print(f"formula: {POWER_FORMULA}")
    pw = achieved_power(p1_i, p2_i, 200)
    print(f"power(n=200) = {pw:.6f}  ({pw*100:.2f}%)")
    return n_ceils


# ---------------------------------------------------------------------------
# TASK C -- A800-hour cost table
# ---------------------------------------------------------------------------

MODEL_SET = [
    # (display_name, size_class, family, one representative recorded model string for reference)
    ("Llama-3.2-3B", "3B", "Llama", "unsloth/Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B", "8B", "Llama", "llama3.1:8b"),
    ("Qwen2.5-7B", "7B", "Qwen", "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B", "12B/14B", "Qwen", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
]


def nearest_gpu_class(cls: str, gpu_by_class: dict[str, dict]) -> str | None:
    candidates = [c for c in CLASS_ORDER if c in gpu_by_class]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(CLASS_PARAMS[c] - CLASS_PARAMS[cls]))


def print_task_c(agg: dict[tuple[str, str], dict], n_ceils: dict[str, int]) -> None:
    print("\n" + "#" * 100)
    print("TASK C -- A800-HOUR COST TABLE (two families x two scale points each)")
    print("#" * 100)

    gpu_by_class = {cls: info for (cls, stack), info in agg.items() if stack == STACK_VLLM_A800}
    print("\nGPU-served (vLLM-A800) median gap-trimmed samples/hour by class, from TASK A:")
    for cls in CLASS_ORDER:
        if cls in gpu_by_class:
            print(f"  {cls:<10} {fmt_rate(gpu_by_class[cls]['median_per_hour'])}/hr "
                  f"(n_files={gpu_by_class[cls]['n_files']})")
        else:
            print(f"  {cls:<10} UNRESOLVED: no GPU-served log for {cls}")

    print("\nper-model resolution (assigning each of the 4 models a GPU samples/hour figure):")
    resolved: dict[str, tuple[float, bool, str]] = {}  # name -> (rate, is_proxy, note)
    for name, cls, family, model_str in MODEL_SET:
        if cls in gpu_by_class:
            rate = gpu_by_class[cls]["median_per_hour"]
            resolved[name] = (rate, False, f"direct: class {cls} GPU-served, "
                                            f"n_files={gpu_by_class[cls]['n_files']}")
            print(f"  {name:<14} (class {cls:<8}, family {family:<5}): "
                  f"{fmt_rate(rate)}/hr -- MEASURED ({resolved[name][2]})")
        else:
            proxy_cls = nearest_gpu_class(cls, gpu_by_class)
            print(f"  {name:<14} (class {cls:<8}, family {family:<5}): "
                  f"UNRESOLVED: no GPU-served log for {cls} (recorded string '{model_str}' "
                  f"only ever runs Ollama-Mac or HF-local-Mac in these logs -- see TASK A "
                  f"per-file table). Resolving this needs a short pilot of this exact model "
                  f"through the api://localhost:8000 vLLM endpoint.")
            if proxy_cls is None:
                print(f"    no GPU-served class exists anywhere -- cannot even proxy. SKIPPED.")
                continue
            rate = gpu_by_class[proxy_cls]["median_per_hour"]
            note = (f"PROXY: nearest GPU-served class by param count is {proxy_cls} "
                    f"({fmt_rate(rate)}/hr, n_files={gpu_by_class[proxy_cls]['n_files']}) -- "
                    f"NOT a measurement of {cls}, likely UNDERSTATES this model's real "
                    f"throughput since {cls} has fewer parameters to decode than {proxy_cls}")
            resolved[name] = (rate, True, note)
            print(f"    -> using proxy: {note}")

    print("\ncost model: each of the 4 models runs `n` samples once under one identical "
          "condition (4 cells @ n samples); GPU-hours(model) = n / samples_per_hour(model); "
          "total A800-hours = sum over the 4 models.\n")

    for case_label, n in n_ceils.items():
        print(f"--- n_per_group = {n}  [TASK B case {case_label}] ---")
        hdr = f"{'model':<14}{'class':<10}{'rate/hr':>12}{'GPU-hours':>12}  note"
        print(hdr)
        print("-" * 80)
        total = 0.0
        for name, cls, family, model_str in MODEL_SET:
            if name not in resolved:
                print(f"{name:<14}{cls:<10}{'--':>12}{'--':>12}  SKIPPED (no proxy available)")
                continue
            rate, is_proxy, note = resolved[name]
            hours = n / rate
            total += hours
            flag = "PROXY" if is_proxy else "measured"
            print(f"{name:<14}{cls:<10}{fmt_rate(rate):>12}{hours:>12.2f}  {flag}")
        print(f"{'TOTAL':<14}{'':<10}{'':<12}{total:>12.2f}  A800-hours for this n\n")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path(__file__).resolve().parents[2] / "logs"))
    ap.add_argument("--configs",
                     default=str(Path(__file__).resolve().parents[2] / "configs"))
    args = ap.parse_args()

    agg = print_task_a(Path(args.logs), Path(args.configs))
    n_ceils = print_task_b()
    print_task_c(agg, n_ceils)


if __name__ == "__main__":
    main()
