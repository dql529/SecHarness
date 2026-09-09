#!/usr/bin/env python3
"""D3 — operational-quality evaluation of the harness beyond accuracy.

Evaluates the harness on five dimensions other than accuracy: audit completeness,
analyst workload, error recovery, escalation quality, explanation usefulness.

Every number is derived from audit logs that already exist on disk; no model is run.

Instrument discipline (each metric declares the layer it reads):
  L-model   llm_calls[].raw_output      what the model itself emitted
  L-tool    tool_chain[].output         what a tool returned
  L-final   result.*                    the harness's final record, which the
                                        permission layer may have rewritten

Reading L-final where L-model is meant produced a false "nothing changed" result on
2026-08-17; the gates below refuse to emit numbers unless the reader demonstrably
separates the two layers.

Usage: python3 d3_operational_quality.py <project_root> [--out results/tables/d3]
Exit code 0 only if every gate passes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(".")  # set from argv in main(); see the other analysis/ scripts

# --------------------------------------------------------------------------------------
# Condition allowlist (M5: only registry-backed runs; superseded runs are excluded here,
# not filtered downstream). `patched` marks runs made after the 2026-08-17 fallback fix.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    key: str
    log: str
    family: str
    capacity: str
    harness: bool
    stress: str  # "none" | "-K" | "-O" | "-P" | "-T" | "-RF" | "-A"
    patched: bool
    note: str = ""


CONDITIONS: tuple[Condition, ...] = (
    # --- same-hardware batch, post-fix (A800, 2026-08-17/18) -------------------------
    Condition("E1-3B", "logs/v2/lat_E1_unsw_sub200_audit.jsonl", "Llama-3.2", "3B", False, "none", True, "zero-shot, no harness"),
    Condition("E1RF-3B", "logs/v2/lat_E1_rf_context_unsw_sub200_audit.jsonl", "Llama-3.2", "3B", False, "none", True, "RF verdict+confidence in prompt, no loop"),
    Condition("E3-3B", "logs/v2/lat_E3_unsw_sub200_audit.jsonl", "Llama-3.2", "3B", True, "none", True, "full harness"),
    Condition("E4-3B", "logs/v2/lat_E4_unsw_sub200_audit.jsonl", "Llama-3.2", "3B", True, "none", True, "fine-tuned + full harness"),
    Condition("E4-3B-A", "logs/v2/E4_ablation_noActions_unsw_sub200_audit.jsonl", "Llama-3.2", "3B", True, "-A", True, ""),
    Condition("E3-7B", "logs/v2/E3_7B_ablation_full_unsw_sub200_audit.jsonl", "Qwen2.5", "7B", True, "none", True, ""),
    Condition("E3-7B-A", "logs/v2/E3_7B_ablation_noActions_unsw_sub200_audit.jsonl", "Qwen2.5", "7B", True, "-A", True, ""),
    Condition("E3-14B", "logs/v2_tdsc/tau_13B_full_unsw_sub200_audit.jsonl", "Qwen2.5", "14B", True, "none", True, "registry label '13B' = Qwen2.5-14B"),
    Condition("E3-14B-A", "logs/v2_tdsc/tau_13B_noActions_unsw_sub200_audit.jsonl", "Qwen2.5", "14B", True, "-A", True, ""),
    Condition("E3-32B", "logs/v2_tdsc/tau_32B_full_unsw_sub200_audit.jsonl", "Qwen2.5", "32B", True, "none", True, ""),
    Condition("E3-32B-A", "logs/v2_tdsc/tau_32B_noActions_unsw_sub200_audit.jsonl", "Qwen2.5", "32B", True, "-A", True, ""),
    # --- capacity sweep of the unharnessed prompt-context baseline (pre-fix, but the
    #     fallback path cannot fire without a harness loop) ------------------------------
    Condition("E1RF-7B", "logs/v2/E1_rf_context_7B_unsw_sub200_audit.jsonl", "Qwen2.5", "7B", False, "none", False, ""),
    Condition("E1RF-14B", "logs/v2/E1_rf_context_14B_unsw_sub200_audit.jsonl", "Qwen2.5", "14B", False, "none", False, ""),
    Condition("E1RF-32B", "logs/v2/E1_rf_context_32B_local_unsw_sub200_audit.jsonl", "Qwen2.5", "32B", False, "none", False, ""),
    # --- stress conditions (pre-fix; used for recovery/escalation/explanation, and the
    #     attribution gate needs the pre-fix misattributions to prove it can see them) ---
    Condition("E3-7B-T", "logs/v2_tdsc/tau_7B_noTools_unsw_sub200_audit.jsonl", "Qwen2.5", "7B", True, "-T", False, ""),
    Condition("E3-14B-T", "logs/v2_tdsc/tau_13B_noTools_unsw_sub200_audit.jsonl", "Qwen2.5", "14B", True, "-T", False, ""),
    Condition("E3-32B-T", "logs/v2_tdsc/tau_32B_noTools_unsw_sub200_audit.jsonl", "Qwen2.5", "32B", True, "-T", False, ""),
    Condition("E4-3B-T", "logs/v2/E4_ablation_noTools_unsw_sub200_audit.jsonl", "Llama-3.2", "3B", True, "-T", False, "fine-tuned"),
    Condition("E3noRF-8B", "logs/v2/E3_noRF_8B_unsw_sub200_audit.jsonl", "Llama-3.1", "8B", True, "-RF", False, ""),
    Condition("E3noRF-14B", "logs/v2/E3_noRF_14B_unsw_sub200_audit.jsonl", "Qwen2.5", "14B", True, "-RF", False, ""),
    Condition("E3noRF-32B", "logs/v2/E3_noRF_32B_unsw_sub200_audit.jsonl", "Qwen2.5", "32B", True, "-RF", False, ""),
)

# Harness-authored reasoning strings carry a bracketed tag; anything else is the model's.
HARNESS_TAG = re.compile(r"^\s*\[(ESCALATED|MAX_STEPS|LOW_CONF_FALLBACK|FALLBACK|ERROR)[^\]]*\]")
# A harness-authored record naming the ML tool as the decision source. Two wordings exist
# ("Fallback to ML prediction", "Using ML prediction"); matching only the first missed the
# 21 low-confidence rewrites in the fine-tuned 3B tool-ablation run.
ML_SOURCE_CLAIM = re.compile(r"(fallback to|using)\s+ml\s+prediction", re.I)
ML_EVIDENCE_CLAIM = re.compile(
    r"\bML\b|machine[- ]learning|anomaly detect|anomaly score|classifier|random.?forest|\bRF\b", re.I
)

Record = dict[str, Any]


def load_records(path: Path) -> list[Record]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------------------
# Layer accessors (M1)
# --------------------------------------------------------------------------------------


def model_classify_payload(rec: Record) -> Optional[dict[str, Any]]:
    """L-model: the last structured decision the model itself emitted.

    Reads llm_calls[].raw_output, never result.*, so a permission-layer rewrite of the
    final record cannot masquerade as the model's own output.
    """
    for call in reversed(rec.get("llm_calls") or []):
        raw = call.get("raw_output") or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                continue
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue
        if payload.get("tool") in {"classify", "escalate"}:
            args = payload.get("args")
            return args if isinstance(args, dict) else None
        if "verdict" in payload:  # no-harness conditions emit the verdict directly
            return payload
    return None


def model_verdict(rec: Record) -> Optional[str]:
    payload = model_classify_payload(rec)
    if not payload:
        return None
    verdict = payload.get("verdict")
    return verdict.strip().lower() if isinstance(verdict, str) else None


REASONING_FIELD = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)


def model_reasoning(rec: Record) -> str:
    """L-model: the reason the model itself stated.

    Whole-payload JSON parsing recovers only part of the corpus: on the unharnessed
    conditions roughly 60% of raw outputs are truncated mid-string by the generation
    cap and carry no closing brace, which the pipeline's own field-level parser
    tolerates. Two fallbacks mirror it, in order of decreasing directness:
      1. field-level extraction from the raw output;
      2. result.reasoning, but only when it carries no harness tag -- a tagged string
         is harness-authored and must never be scored as the model's words (M2).
    """
    payload = model_classify_payload(rec) or {}
    text = payload.get("reasoning") or payload.get("reason") or ""
    if isinstance(text, str) and text:
        return text
    for call in reversed(rec.get("llm_calls") or []):
        match = REASONING_FIELD.search(call.get("raw_output") or "")
        if match and match.group(1):
            return match.group(1)
    final = final_reasoning(rec)
    return "" if HARNESS_TAG.search(final) else final


def final_reasoning(rec: Record) -> str:
    return (rec.get("result") or {}).get("reasoning") or ""


def valid_ml_payloads(rec: Record) -> list[dict[str, Any]]:
    """L-tool: check_anomaly returns that actually carry a prediction.

    An ablated tool returns a well-formed JSON *error object*; treating that as a
    prediction is precisely the 2026-08-17 defect, so membership requires the key.
    """
    out: list[dict[str, Any]] = []
    for call in rec.get("tool_chain") or []:
        if call.get("tool") != "check_anomaly":
            continue
        try:
            payload = json.loads(call.get("output") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "prediction" in payload:
            out.append(payload)
    return out


def tool_errors(rec: Record) -> int:
    n = 0
    for call in rec.get("tool_chain") or []:
        try:
            payload = json.loads(call.get("output") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "error" in payload:
            n += 1
    return n


def is_correct(rec: Record) -> Optional[bool]:
    value = (rec.get("evaluation") or {}).get("detection_correct")
    return None if value is None else bool(value)


# --------------------------------------------------------------------------------------
# D3-A audit completeness and attribution faithfulness
# --------------------------------------------------------------------------------------

REQUIRED_TOOL_FIELDS = ("tool", "input", "output", "latency_ms", "timestamp")


def audit_metrics(recs: list[Record]) -> dict[str, float]:
    n = len(recs)
    prov_ok = attr_ok = 0
    step_fields = model_out = evidence_link = terminal = 0
    misattributed = silent_override = disclosed_override = 0
    for rec in recs:
        result = rec.get("result") or {}
        chain = rec.get("tool_chain") or []
        calls = rec.get("llm_calls") or []

        p_steps = all(all(f in c and c[f] is not None for f in REQUIRED_TOOL_FIELDS) for c in chain)
        p_model = bool(calls) and all((c.get("raw_output") or "").strip() for c in calls)
        # The verdict is evidence-linked when the trace itself records what produced it:
        # a tool return, or the model's own structured decision.
        p_evidence = bool(chain) or model_classify_payload(rec) is not None
        # No-harness runs record neither a tool call nor the ML opinion they were shown,
        # so the verdict cannot be traced to the evidence that drove it.
        if not chain:
            p_evidence = False
        p_terminal = bool(result.get("termination"))

        step_fields += p_steps
        model_out += p_model
        evidence_link += p_evidence
        terminal += p_terminal
        prov_ok += p_steps and p_model and p_evidence and p_terminal

        # Attribution violations.
        v1 = bool(ML_SOURCE_CLAIM.search(final_reasoning(rec))) and not valid_ml_payloads(rec)
        mv, fv = model_verdict(rec), (result.get("verdict") or "").strip().lower()
        overridden = bool(mv and fv and mv != fv)
        v3 = overridden and not HARNESS_TAG.search(final_reasoning(rec))
        misattributed += v1
        silent_override += v3
        disclosed_override += overridden and not v3
        attr_ok += not (v1 or v3)
    return {
        "n": n,
        "acs_prov": prov_ok / n,
        "prov_step_fields": step_fields / n,
        "prov_model_output": model_out / n,
        "prov_evidence_linked": evidence_link / n,
        "prov_terminal_action": terminal / n,
        "acs_attr": attr_ok / n,
        "attr_false_ml_claim": misattributed,
        "attr_silent_override": silent_override,
        "attr_disclosed_override": disclosed_override,
    }


# --------------------------------------------------------------------------------------
# D3-B analyst workload
# --------------------------------------------------------------------------------------


def tie_aware_risk_coverage(pairs: list[tuple[float, bool]]) -> dict[str, float]:
    """Selective-prediction cost of catching the system's errors by manual review.

    `pairs` are (reported confidence, correct?). Reviewing proceeds from the least
    confident alert upward. Ties are reviewed as whole groups: a coarse confidence
    scale (E1-RF emits 11 distinct values) must not be credited with resolution it
    does not have (M4).
    """
    n = len(pairs)
    errors = sum(1 for _, ok in pairs if not ok)
    if n == 0 or errors == 0:
        return {"rc_review_for_all_errors": float("nan"), "aurc": float("nan"),
                "n_errors": errors, "distinct_conf": len({c for c, _ in pairs})}
    worst_error_conf = max(c for c, ok in pairs if not ok)
    review = sum(1 for c, _ in pairs if c <= worst_error_conf) / n

    # AURC over tie groups: accept the most confident groups first.
    groups: dict[float, list[bool]] = {}
    for c, ok in pairs:
        groups.setdefault(c, []).append(ok)
    points: list[tuple[float, float]] = []
    accepted_n = accepted_err = 0
    for conf in sorted(groups, reverse=True):
        oks = groups[conf]
        accepted_n += len(oks)
        accepted_err += sum(1 for ok in oks if not ok)
        points.append((accepted_n / n, accepted_err / accepted_n))
    aurc = 0.0
    prev_cov, prev_risk = 0.0, points[0][1]
    for cov, risk in points:
        aurc += (cov - prev_cov) * (risk + prev_risk) / 2
        prev_cov, prev_risk = cov, risk
    return {"rc_review_for_all_errors": review, "aurc": aurc, "n_errors": errors,
            "distinct_conf": len(groups)}


def tool_only_workload(recs: list[Record]) -> dict[str, float]:
    """The control with no language model at all: the tool's own verdict and confidence.

    Comparing the harness only against an unharnessed LLM measures it against a condition we
    built, in which the model is handed a calibrated confidence and discards it. The operator's
    real alternative is to route the detector's own output to the alert queue at no inference
    cost, so that row belongs in the table. Extracted from the harnessed run's own tool returns,
    which makes it the same detector on the same samples.
    """
    pairs = []
    for rec in recs:
        mls = valid_ml_payloads(rec)
        gt = (rec.get("evaluation") or {}).get("binary_gt")
        if not mls or gt is None:
            continue
        ml = mls[0]
        pred = "attack" if ml.get("prediction") != "Normal" else "benign"
        pairs.append((float(ml["confidence"]), pred == gt))
    out = tie_aware_risk_coverage(pairs)
    out["n_rows"] = len(pairs)
    out["accuracy"] = sum(1 for _, ok in pairs if ok) / max(1, len(pairs))
    return out


def workload_metrics(recs: list[Record]) -> dict[str, float]:
    pairs: list[tuple[float, bool]] = []
    for rec in recs:
        conf = (rec.get("result") or {}).get("confidence")
        ok = is_correct(rec)
        if conf is None or ok is None:
            continue
        pairs.append((float(conf), ok))
    out = tie_aware_risk_coverage(pairs)
    toks = [(r.get("efficiency") or {}).get("total_tokens") for r in recs]
    steps = [(r.get("result") or {}).get("total_steps") for r in recs]
    out["read_tokens_per_verdict"] = sum(t for t in toks if t) / max(1, len([t for t in toks if t]))
    out["steps_per_verdict"] = sum(s for s in steps if s is not None) / max(1, len([s for s in steps if s is not None]))
    return out


# --------------------------------------------------------------------------------------
# D3-C error recovery
# --------------------------------------------------------------------------------------


def recovery_metrics(recs: list[Record]) -> dict[str, float]:
    n = len(recs)
    with_err = [r for r in recs if tool_errors(r) > 0]
    recovered = sum(
        1 for r in with_err
        if (r.get("result") or {}).get("termination") in {"classify", "escalate"}
        and (r.get("result") or {}).get("verdict") in {"attack", "benign"}
    )
    term = Counter((r.get("result") or {}).get("termination") for r in recs)
    return {
        "frac_traces_with_tool_error": len(with_err) / n,
        "recovery_rate": (recovered / len(with_err)) if with_err else float("nan"),
        "parse_fail_rate": term.get("parse_fail", 0) / n,
        "max_steps_rate": term.get("max_steps", 0) / n,
        "termination": dict(term),
    }


# --------------------------------------------------------------------------------------
# D3-D escalation quality
# --------------------------------------------------------------------------------------


def escalation_metrics(recs: list[Record]) -> dict[str, float]:
    n = len(recs)
    esc = [r for r in recs if (r.get("result") or {}).get("termination") == "escalate"]
    rest = [r for r in recs if (r.get("result") or {}).get("termination") != "escalate"]
    model_initiated = sum(
        1 for r in esc
        if any('"tool": "escalate"' in (c.get("raw_output") or "").replace("'", '"')
               for c in (r.get("llm_calls") or []))
    )

    def err_rate(rs: list[Record]) -> float:
        vals = [is_correct(r) for r in rs]
        vals = [v for v in vals if v is not None]
        return (sum(1 for v in vals if not v) / len(vals)) if vals else float("nan")

    e_esc, e_rest = err_rate(esc), err_rate(rest)
    lift = (e_esc / e_rest) if (e_rest and not math.isnan(e_esc) and e_rest > 0) else float("nan")
    return {
        "escalation_rate": len(esc) / n,
        "escalations_model_initiated": model_initiated,
        "err_rate_escalated": e_esc,
        "err_rate_not_escalated": e_rest,
        "escalation_error_lift": lift,
    }


# --------------------------------------------------------------------------------------
# D3-E explanation quality
# --------------------------------------------------------------------------------------


def explanation_metrics(recs: list[Record], harness: bool) -> dict[str, float]:
    """Faithfulness, diversity and error-discriminativeness of the stated reasons.

    Faithfulness is evaluated on model-authored text only (L-model). For harnessed
    runs an ML claim is unfaithful when no check_anomaly return carried a prediction.
    For no-harness runs the ML opinion legitimately arrives in the prompt, so an ML
    claim there is not a fabrication and the rate is reported as not applicable.
    """
    n = len(recs)
    # A model statement exists only where the agent actually terminated by stating one.
    # Traces that exhaust the step budget or fail to parse carry no model explanation by
    # construction, so they belong in the denominator of neither recovery nor faithfulness.
    stated = [r for r in recs
              if (r.get("result") or {}).get("termination") in {"classify", "escalate", "direct_verdict"}]
    texts = [(r, model_reasoning(r)) for r in stated]
    recovered = [(r, t) for r, t in texts if t and not HARNESS_TAG.search(t)]

    # Faithfulness is only meaningful for a verdict that cites evidence for itself.
    # An escalation reason such as "Anomaly detection tools not available" names the ML
    # tool in order to report that it is missing; counting that as a fabricated citation
    # scored the 14B agent's honesty as dishonesty (101 of 101 flags at the -T condition).
    justified = [(r, t) for r, t in recovered
                 if (r.get("result") or {}).get("termination") in {"classify", "direct_verdict"}]
    hallucinated = ml_claims = 0
    for rec, text in justified:
        claims_ml = bool(ML_EVIDENCE_CLAIM.search(text))
        ml_claims += claims_ml
        if harness and claims_ml and not valid_ml_payloads(rec):
            hallucinated += 1
    n_justified = max(1, len(justified))

    finals = [final_reasoning(r) for r in recs]
    counts = Counter(t for _, t in recovered)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values()) if total else float("nan")
    denom = max(1, len(recovered))

    # Reader coverage vs the pipeline's own parser: wherever the pipeline recovered an
    # untagged reason, this reader must recover one too. A shortfall means the metrics
    # below describe a subset selected by parser behaviour rather than by condition.
    reader_miss = sum(
        1 for r, t in texts
        if not t and final_reasoning(r).strip() and not HARNESS_TAG.search(final_reasoning(r))
    )

    return {
        "n_model_statements": len(recovered),
        "reader_miss_vs_pipeline": reader_miss,
        # Verdicts the agent delivered with no reason field at all. A harness property:
        # the action schema accepts a classify call that omits `reasoning`.
        "explanation_omission_rate": sum(1 for _, t in texts if not t) / max(1, len(stated)),
        "traces_without_model_statement": (n - len(stated)) / n,
        "n_evidence_claiming_verdicts": len(justified),
        "explanations_claiming_ml": ml_claims / n_justified,
        "hallucinated_evidence_rate": (hallucinated / n_justified) if harness else float("nan"),
        "distinct_explanations_model": len(counts) / denom,
        "distinct_explanations_final": len({t for t in finals if t}) / n,
        "explanation_entropy_bits": entropy,
        "harness_authored_final_reasoning": sum(1 for t in finals if HARNESS_TAG.search(t)) / n,
    }


def explanation_discriminativeness(recs: list[Record], seed: int = 42) -> float:
    """Can an analyst tell right from wrong verdicts by reading the reason alone?

    Bag-of-words logistic regression, 5-fold stratified CV, ROC-AUC. 0.5 means the
    text carries no signal about whether the verdict is correct.

    Not computed where the explanation column is near-constant (fewer than 10 distinct
    strings): fitting a text model to a column the design forces to be constant produces
    a number that describes the design, not the explanations.
    """
    texts, labels = [], []
    for rec in recs:
        text, ok = model_reasoning(rec), is_correct(rec)
        if text and ok is not None:
            texts.append(text)
            labels.append(int(not ok))  # predict "this verdict is wrong"
    if len(set(labels)) < 2 or min(Counter(labels).values()) < 10 or len(set(texts)) < 10:
        return float("nan")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # error_score="raise": a fold that fails must surface, not become a silent NaN.
    scores = cross_val_score(pipe, texts, labels, cv=cv, scoring="roc_auc", error_score="raise")
    return float(scores.mean())


# --------------------------------------------------------------------------------------
# Gates (M3): the reader must be shown to separate layers and to see differences.
# --------------------------------------------------------------------------------------


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


def run_gates(rows: dict[str, dict[str, Any]]) -> list[Gate]:
    gates: list[Gate] = []

    def get(key: str, metric: str) -> Any:
        return rows.get(key, {}).get(metric)

    # G1 the attribution reader sees the known pre-fix misattributions, and only those.
    # Counts established independently by audit-log replay on 2026-08-17.
    known = {"E3noRF-8B": 1, "E3noRF-14B": 18, "E3noRF-32B": 39}
    detail = ", ".join(f"{k}: {get(k, 'attr_false_ml_claim')} (expect {v})" for k, v in known.items())
    gates.append(Gate("G1 attribution reader reproduces known pre-fix misattributions",
                      all(get(k, "attr_false_ml_claim") == v for k, v in known.items()), detail))

    # G2 a post-fix run must show none, so G1 is not passing on a constant.
    post = {k: get(k, "attr_false_ml_claim") for k in ("E3-3B", "E3-14B", "E3-32B")}
    gates.append(Gate("G2 post-fix runs show zero misattribution",
                      all(v == 0 for v in post.values()), str(post)))

    # G3 the tool-error reader separates healthy from ablated conditions.
    healthy = get("E3-3B", "frac_traces_with_tool_error")
    ablated = get("E3-14B-T", "frac_traces_with_tool_error")
    gates.append(Gate("G3 tool-error reader distinguishes healthy from -T",
                      healthy == 0.0 and ablated == 1.0, f"full={healthy}, -T={ablated}"))

    # G4 escalation is attributed to the model, not to a harness rule.
    esc_n = round((get("E3-14B-T", "escalation_rate") or 0) * (get("E3-14B-T", "n") or 0))
    gates.append(Gate("G4 escalations are model-initiated",
                      get("E3-14B-T", "escalations_model_initiated") == esc_n and esc_n > 0,
                      f"model-initiated={get('E3-14B-T', 'escalations_model_initiated')} of {esc_n}"))

    # G5 the explanation reader separates model-authored from harness-authored text.
    gates.append(Gate("G5 harness-authored reasoning is detected where it exists",
                      (get("E3noRF-32B", "harness_authored_final_reasoning") or 0) > 0
                      and (get("E3-3B", "harness_authored_final_reasoning") or 0) == 0,
                      f"E3noRF-32B={get('E3noRF-32B', 'harness_authored_final_reasoning')}, "
                      f"E3-3B={get('E3-3B', 'harness_authored_final_reasoning')}"))

    # G6 the confidence reader sees the coarseness of the unharnessed baseline.
    gates.append(Gate("G6 risk-coverage reader sees confidence granularity gap",
                      (get("E3-3B", "distinct_conf") or 0) > 3 * (get("E1RF-3B", "distinct_conf") or 1),
                      f"E3-3B={get('E3-3B', 'distinct_conf')} vs E1RF-3B={get('E1RF-3B', 'distinct_conf')}"))

    # G7 the false-ML-claim reader covers both harness wordings, not just one. The
    # fine-tuned 3B tool-ablation run rewrites 21 verdicts with "Using ML prediction"
    # while no tool is available; a reader matching only "Fallback to ..." reports 0.
    gates.append(Gate("G7 false-ML-claim reader covers the low-confidence wording",
                      get("E4-3B-T", "attr_false_ml_claim") == 21,
                      f"E4-3B-T={get('E4-3B-T', 'attr_false_ml_claim')} (expect 21)"))

    # G9 the explanation reader never recovers less than the pipeline's own parser.
    # Whole-payload JSON parsing alone missed 118 of 200 unharnessed records (truncated
    # generations with no closing brace) against 0 of the harnessed ones, which would
    # have compared explanation metrics across two differently-selected subsets.
    misses = {k: v.get("reader_miss_vs_pipeline", 0) for k, v in rows.items()}
    worst = max(misses.items(), key=lambda kv: kv[1])
    gates.append(Gate("G9 explanation reader loses nothing the pipeline parser recovered",
                      worst[1] == 0, f"worst: {worst[0]}={worst[1]} records"))

    # G10 the faithfulness reader flags fabricated citations without flagging honest
    # reports of tool unavailability, which are escalation reasons, not evidence claims.
    gates.append(Gate("G10 faithfulness reader separates fabrication from honest failure report",
                      get("E4-3B-T", "hallucinated_evidence_rate") == 1.0
                      and (get("E3-14B-T", "hallucinated_evidence_rate") or 0) < 1.0,
                      f"E4-3B-T(fine-tuned, fabricates)={get('E4-3B-T', 'hallucinated_evidence_rate')}, "
                      f"E3-14B-T(escalates)={get('E3-14B-T', 'hallucinated_evidence_rate')}"))

    # G8 the override reader is shown to detect verdict overrides at all before its
    # zero count for undisclosed overrides is reported as a finding.
    disclosed = get("E4-3B-T", "attr_disclosed_override")
    gates.append(Gate("G8 override reader detects known disclosed overrides",
                      bool(disclosed and disclosed > 0),
                      f"E4-3B-T disclosed={disclosed}, undisclosed across corpus="
                      f"{sum(r.get('attr_silent_override', 0) for r in rows.values())}"))
    return gates


# --------------------------------------------------------------------------------------


def main() -> int:
    global PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", default=".",
                        help="project/ directory containing logs/ and results/")
    parser.add_argument("--out", default="results/tables/d3")
    args = parser.parse_args()
    PROJECT_ROOT = Path(args.project_root).resolve()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for cond in CONDITIONS:
        path = PROJECT_ROOT / cond.log
        if not path.exists():
            missing.append(cond.log)
            continue
        recs = load_records(path)
        row: dict[str, Any] = {
            "condition": cond.key, "family": cond.family, "capacity": cond.capacity,
            "harness": cond.harness, "stress": cond.stress, "patched": cond.patched,
            "log": cond.log, "note": cond.note, "n": len(recs),
        }
        row.update(audit_metrics(recs))
        row.update(workload_metrics(recs))
        row.update(recovery_metrics(recs))
        row.update(escalation_metrics(recs))
        row.update(explanation_metrics(recs, harness=cond.harness))
        row["explanation_error_auc"] = explanation_discriminativeness(recs)
        rows[cond.key] = row

    if missing:
        print("MISSING LOGS:", *missing, sep="\n  ", file=sys.stderr)
        return 2

    gates = run_gates(rows)
    for gate in gates:
        print(f"[{'PASS' if gate.passed else 'FAIL'}] {gate.name} :: {gate.detail}")
    if not all(g.passed for g in gates):
        print("GATES FAILED — refusing to emit numbers", file=sys.stderr)
        return 1

    tool_only = tool_only_workload(load_records(PROJECT_ROOT / "logs/v2/lat_E3_unsw_sub200_audit.jsonl"))
    print(f"\ntool-only control (no LLM): acc={tool_only['accuracy']:.3f} "
          f"review={tool_only['rc_review_for_all_errors']:.3f} aurc={tool_only['aurc']:.4f} "
          f"distinct_conf={tool_only['distinct_conf']} n={tool_only['n_rows']}")

    import pandas as pd

    df = pd.DataFrame(list(rows.values()))
    df.to_csv(out_dir / "operational_quality_percondition.csv", index=False)
    with (out_dir / "operational_quality_gates.json").open("w") as fh:
        json.dump([{"gate": g.name, "passed": g.passed, "detail": g.detail} for g in gates], fh, indent=2)

    cols = ["condition", "capacity", "harness", "stress", "acs_prov", "acs_attr",
            "rc_review_for_all_errors", "aurc", "distinct_conf", "read_tokens_per_verdict",
            "recovery_rate", "parse_fail_rate", "escalation_rate", "escalation_error_lift", "explanation_omission_rate",
            "hallucinated_evidence_rate", "distinct_explanations_model", "explanation_error_auc"]
    print()
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_dir/'operational_quality_percondition.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
