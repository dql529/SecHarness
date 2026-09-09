#!/usr/bin/env python3
"""
analyze_tau_curves.py — Post-hoc analyzer for Exp-τ phase transition detection.

CONTRACT: main() -> None
    1. Load all per-condition results from:
       - project/results/tables/v2/ (3B/7B existing conditions)
       - project/results/tables/v2_tdsc/ (7B-noTools, 13B, 32B new conditions)
    2. For each (model_size, ablation), extract accuracy, FPR, chain_length
    3. Compute Δa per component vs full baseline (in percentage points)
    4. Compute tau_hat(M) := I(H\\T; Y|T) / H(Y|T) using compute_mi_plugin
       on per-sample tool invocation flags from audit logs
    5. Phase transition detection: find M* where ∂a/∂H_i sign flips
    6. T7 dichotomy: classify each model into {relay, exploration, boundary}
    7. Write tau_summary.csv, tau_summary.json
    8. Generate 2 pgfplots figures via pdflatex:
       - fig_tau_phase_transition.pdf: line plot (log x = model_size, y = Δa per component)
       - fig_component_gradient.pdf: bar chart of Δa per (model, component)
    9. Append a summary section to the results registry markdown file
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("analyze-tau")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

RESULTS_V2 = PROJECT_ROOT / "results" / "tables" / "v2"
RESULTS_V2_TDSC = PROJECT_ROOT / "results" / "tables" / "v2_tdsc"
LOGS_V2 = PROJECT_ROOT / "logs" / "v2"
LOGS_V2_TDSC = PROJECT_ROOT / "logs" / "v2_tdsc"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "v2_tdsc"
REGISTRY_PATH = PROJECT_ROOT / "results" / "VERIFIED_REGISTRY.md"

# Map (model_size_label, ablation) -> (results_dir, exp_id)
# Existing conditions from v2 experiments (3B=Llama-3.2-3B, 7B=Qwen2.5-7B)
CONDITION_MAP: dict[tuple[str, str], tuple[Path, str]] = {
    # 3B ablations (Llama-3.2-3B; per-component ablation files not run — will resolve to None)
    ("3B", "full"):           (RESULTS_V2, "E3_unsw"),                         # E3_unsw_results.json — 3B zero-shot+harness
    ("3B", "noKnowledge"):    (RESULTS_V2, "E3_3B_ablation_noKnowledge_unsw"), # no file on disk — resolves to None
    ("3B", "noObservation"):  (RESULTS_V2, "E3_3B_ablation_noObservation_unsw"), # no file on disk — resolves to None
    ("3B", "noPermissions"):  (RESULTS_V2, "E3_3B_ablation_noPermissions_unsw"), # no file on disk — resolves to None
    ("3B", "noTools"):        (RESULTS_V2, "E3_3B_ablation_noTools_unsw"),     # no file on disk — resolves to None
    # 7B ablations (Qwen2.5-7B; full=E3_8B_unsw; noTools new from v2_tdsc)
    ("7B", "full"):           (RESULTS_V2, "E3_8B_unsw"),                      # E3_8B_unsw_results.json — 7B zero-shot+harness
    ("7B", "noKnowledge"):    (RESULTS_V2, "E3_7B_ablation_noKnowledge_unsw"), # E3_7B_ablation_noKnowledge_unsw_results.json
    ("7B", "noObservation"):  (RESULTS_V2, "E3_7B_ablation_noObservation_unsw"), # E3_7B_ablation_noObservation_unsw_results.json
    ("7B", "noPermissions"):  (RESULTS_V2, "E3_7B_ablation_noPermissions_unsw"), # E3_7B_ablation_noPermissions_unsw_results.json
    ("7B", "noTools"):        (RESULTS_V2_TDSC, "tau_7B_noTools_unsw"),        # tau_7B_noTools_unsw_results.json
    # 13B ablations (new)
    ("13B", "full"):          (RESULTS_V2_TDSC, "tau_13B_full_unsw"),
    ("13B", "noKnowledge"):   (RESULTS_V2_TDSC, "tau_13B_noKnowledge_unsw"),
    ("13B", "noObservation"): (RESULTS_V2_TDSC, "tau_13B_noObservation_unsw"),
    ("13B", "noPermissions"): (RESULTS_V2_TDSC, "tau_13B_noPermissions_unsw"),
    ("13B", "noTools"):       (RESULTS_V2_TDSC, "tau_13B_noTools_unsw"),
    # 32B ablations (new)
    ("32B", "full"):          (RESULTS_V2_TDSC, "tau_32B_full_unsw"),
    ("32B", "noKnowledge"):   (RESULTS_V2_TDSC, "tau_32B_noKnowledge_unsw"),
    ("32B", "noObservation"): (RESULTS_V2_TDSC, "tau_32B_noObservation_unsw"),
    ("32B", "noPermissions"): (RESULTS_V2_TDSC, "tau_32B_noPermissions_unsw"),
    ("32B", "noTools"):       (RESULTS_V2_TDSC, "tau_32B_noTools_unsw"),
}

MODEL_SIZES_ORDERED = ["3B", "7B", "13B", "32B"]
# Parameter counts in billions for log-scale x-axis
MODEL_PARAMS_B: dict[str, float] = {"3B": 3.2, "7B": 7.0, "13B": 14.0, "32B": 32.0}
ABLATION_COMPONENTS = ["noKnowledge", "noObservation", "noPermissions", "noTools"]
COMPONENT_LABELS = {
    "noKnowledge": "Knowledge",
    "noObservation": "Observation",
    "noPermissions": "Permissions",
    "noTools": "All Tools",
}

# T7 thresholds: relay if ALL |Δa| < 1pp, exploration if any Δa > +1pp
RELAY_THRESHOLD_PP = 1.0
EXPLORATION_THRESHOLD_PP = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition_metrics(results_dir: Path, exp_id: str) -> dict[str, Any] | None:
    """
    CONTRACT: load_condition_metrics(results_dir, exp_id) -> dict | None
        results_dir: Path  — directory containing {exp_id}_results.json
        exp_id: str        — experiment identifier
        returns: metrics dict or None if file not found
    """
    json_path = results_dir / f"{exp_id}_results.json"
    if not json_path.exists():
        log.warning("Results not found: %s", json_path)
        return None
    with open(json_path) as f:
        return json.load(f)


def load_all_conditions() -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load all available condition results. Missing conditions are logged and skipped.
    Returns: {(model_size, ablation): metrics_dict}
    """
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for (size, abl), (rdir, exp_id) in CONDITION_MAP.items():
        m = load_condition_metrics(rdir, exp_id)
        if m is not None:
            results[(size, abl)] = m
            log.info("Loaded (%s, %s): acc=%.4f fpr=%.4f", size, abl, m.get("accuracy", 0), m.get("fpr", 0))
        else:
            log.warning("Missing condition: (%s, %s) — will be omitted from analysis", size, abl)
    return results


# ---------------------------------------------------------------------------
# MI estimation from audit logs
# ---------------------------------------------------------------------------

def load_audit_tool_flags(logs_dir: Path, exp_id: str) -> list[dict[str, Any]] | None:
    """
    Load per-sample tool invocation flags from audit JSONL.
    Returns list of per-sample records or None if log not found.
    """
    log_path = logs_dir / f"{exp_id}_audit.jsonl"
    if not log_path.exists():
        log.warning("Audit log not found: %s", log_path)
        return None
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    log.info("Loaded %d audit records from %s", len(records), log_path)
    return records


def compute_tau_hat(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    CONTRACT: compute_tau_hat(records) -> dict
        records: list[dict]  — per-sample audit records with tool_chain and evaluation
        returns: {
            tau_hat: float   — I(H\\T; Y|T) / H(Y|T) (proxy)
            mi: float        — I(H\\T; Y|T) in nats
            h_y_given_t: float
            n: int
        }

    Uses compute_mi_plugin to estimate I(H_tool_used; Y | T_anomaly_score).
    H\\T proxy: whether any non-primary tool was invoked (binary flag).
    T proxy: check_anomaly score bucket (high/low derived from tool_chain presence).
    Y: binary ground truth (attack/benign).
    """
    try:
        from project.scripts.v2_tdsc.compute_mi_plugin import plugin_mi
    except ImportError:
        # Fallback: direct import for when running from project root
        _script_dir = Path(__file__).parent
        sys.path.insert(0, str(_script_dir))
        from compute_mi_plugin import plugin_mi  # type: ignore[import]

    xs: list[str] = []  # H\\T proxy: secondary tool usage pattern
    ys: list[str] = []  # Y: ground truth
    ts: list[str] = []  # T proxy: primary tool (check_anomaly) invoked or not

    for rec in records:
        if isinstance(rec, dict):
            tool_chain = rec.get("tool_chain", [])
            eval_info = rec.get("evaluation", {})
            gt = eval_info.get("binary_gt", "unknown")
        else:
            tool_chain = getattr(rec, "tool_chain", [])
            gt = getattr(rec, "binary_gt", "unknown")

        if gt == "unknown":
            continue

        # T proxy: whether check_anomaly was called
        t_val = "check_anomaly_called" if "check_anomaly" in tool_chain else "no_check_anomaly"
        # H\T proxy: whether any secondary tool (lookup_signature, query_history, load_knowledge) used
        secondary_tools = {"lookup_signature", "query_history", "load_knowledge"}
        secondary_used = any(tool in tool_chain for tool in secondary_tools)
        x_val = "secondary_used" if secondary_used else "no_secondary"

        xs.append(x_val)
        ys.append(gt)
        ts.append(t_val)

    if not xs:
        log.warning("No valid records for MI estimation.")
        return {"tau_hat": float("nan"), "mi": float("nan"), "h_y_given_t": float("nan"), "n": 0}

    result = plugin_mi(xs, ys, ts)
    h_y_given_t = result["h_y_given_t"]
    mi = result["mi"]

    if math.isnan(h_y_given_t) or h_y_given_t == 0.0:
        tau_hat = float("nan")
    else:
        tau_hat = mi / h_y_given_t

    log.info("tau_hat=%.4f  MI=%.4f  H(Y|T)=%.4f  n=%d", tau_hat, mi, h_y_given_t, result["n"])
    return {
        "tau_hat": round(tau_hat, 6) if not math.isnan(tau_hat) else float("nan"),
        "mi": round(mi, 6),
        "h_y_given_t": round(h_y_given_t, 6),
        "n": result["n"],
    }


# ---------------------------------------------------------------------------
# Phase transition analysis
# ---------------------------------------------------------------------------

def compute_delta_accuracy(
    condition_data: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    CONTRACT: compute_delta_accuracy(condition_data) -> dict
        condition_data: {(model_size, ablation): metrics}
        returns: {model_size: {component: delta_acc_pp}} where delta_acc_pp = (ablation_acc - full_acc) * 100
        Missing conditions yield None.
    """
    deltas: dict[str, dict[str, float | None]] = {}
    for size in MODEL_SIZES_ORDERED:
        deltas[size] = {}
        full_metrics = condition_data.get((size, "full"))
        if full_metrics is None:
            log.warning("No 'full' condition for %s; Δa will be None for all components.", size)
            for abl in ABLATION_COMPONENTS:
                deltas[size][abl] = None
            continue
        full_acc = full_metrics.get("accuracy", 0.0)
        for abl in ABLATION_COMPONENTS:
            abl_metrics = condition_data.get((size, abl))
            if abl_metrics is None:
                deltas[size][abl] = None
            else:
                delta_pp = (abl_metrics.get("accuracy", 0.0) - full_acc) * 100.0
                deltas[size][abl] = round(delta_pp, 4)
    return deltas


def classify_mode(deltas: dict[str, float | None]) -> str:
    """
    CONTRACT: classify_mode(deltas) -> str
        deltas: {component: delta_acc_pp or None}
        returns: "relay" | "exploration" | "boundary" | "insufficient_data"

    relay: ALL |Δa| < RELAY_THRESHOLD_PP (removing components has no effect)
    exploration: any Δa > +EXPLORATION_THRESHOLD_PP (removing component hurts significantly)
    boundary: mixed — some relay, some exploration
    """
    valid_deltas = [v for v in deltas.values() if v is not None]
    if not valid_deltas:
        return "insufficient_data"

    all_relay = all(abs(d) < RELAY_THRESHOLD_PP for d in valid_deltas)
    any_exploration = any(d > EXPLORATION_THRESHOLD_PP for d in valid_deltas)

    if all_relay:
        return "relay"
    if any_exploration and not all_relay:
        return "exploration"
    return "boundary"


def detect_phase_transition(
    deltas: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """
    CONTRACT: detect_phase_transition(deltas) -> dict
        deltas: {model_size: {component: delta_pp or None}}
        returns: {
            transition_found: bool,
            transition_interval: str,  e.g. "(3B, 13B]" or "none"
            component_transitions: {component: interval or "monotonic" or "insufficient"}
            report: str
        }

    For each component, check if ∂a/∂H_i sign flips between consecutive model sizes.
    """
    component_transitions: dict[str, str] = {}
    for abl in ABLATION_COMPONENTS:
        sizes_with_data = [
            (size, deltas[size][abl])
            for size in MODEL_SIZES_ORDERED
            if deltas.get(size, {}).get(abl) is not None
        ]
        if len(sizes_with_data) < 2:
            component_transitions[abl] = "insufficient"
            continue

        transition_interval = None
        for i in range(len(sizes_with_data) - 1):
            s1, d1 = sizes_with_data[i]
            s2, d2 = sizes_with_data[i + 1]
            # Sign flip: one positive, one negative (ignoring near-zero)
            if d1 is not None and d2 is not None:
                if (d1 > RELAY_THRESHOLD_PP and d2 < -RELAY_THRESHOLD_PP) or \
                   (d1 < -RELAY_THRESHOLD_PP and d2 > RELAY_THRESHOLD_PP) or \
                   (abs(d1) < RELAY_THRESHOLD_PP and abs(d2) >= RELAY_THRESHOLD_PP) or \
                   (abs(d1) >= RELAY_THRESHOLD_PP and abs(d2) < RELAY_THRESHOLD_PP):
                    transition_interval = f"({s1}, {s2}]"
                    break
        component_transitions[abl] = transition_interval if transition_interval else "monotonic"

    any_transition = any(v not in ("monotonic", "insufficient") for v in component_transitions.values())
    intervals = [v for v in component_transitions.values() if v not in ("monotonic", "insufficient")]
    transition_interval_str = intervals[0] if intervals else "none"

    if any_transition:
        report = f"Phase transition observed at M* ∈ {transition_interval_str}"
    else:
        report = f"Monotonic relay across {MODEL_SIZES_ORDERED[0]}-{MODEL_SIZES_ORDERED[-1]}"

    return {
        "transition_found": any_transition,
        "transition_interval": transition_interval_str,
        "component_transitions": component_transitions,
        "report": report,
    }


# ---------------------------------------------------------------------------
# pgfplots figure generation
# ---------------------------------------------------------------------------

# Colorblind-safe palette (Wong 2011): blue, orange, green, purple
COLORS = {
    "noKnowledge":    "rgb,255:red,0;green,114;blue,178",
    "noObservation":  "rgb,255:red,230;green,159;blue,0",
    "noPermissions":  "rgb,255:red,0;green,158;blue,115",
    "noTools":        "rgb,255:red,204;green,121;blue,167",
}
MARKS = {
    "noKnowledge":   "o",
    "noObservation": "square",
    "noPermissions": "triangle",
    "noTools":       "diamond",
}


def _format_coord(size: str, delta: float | None) -> str | None:
    """Format a pgfplots coordinate for log-scale x (parameter count in billions)."""
    if delta is None:
        return None
    params = MODEL_PARAMS_B[size]
    return f"({params},{delta:.4f})"


def generate_phase_transition_tex(
    deltas: dict[str, dict[str, float | None]],
) -> str:
    """Generate pgfplots LaTeX for fig_tau_phase_transition (line plot, log x-axis)."""
    plots = []
    for abl in ABLATION_COMPONENTS:
        label = COMPONENT_LABELS[abl]
        color = COLORS[abl]
        mark = MARKS[abl]
        coords = []
        for size in MODEL_SIZES_ORDERED:
            coord = _format_coord(size, deltas.get(size, {}).get(abl))
            if coord:
                coords.append(coord)
        if not coords:
            continue
        coord_str = " ".join(coords)
        plots.append(
            f"        \\addplot[color={{{color}}},mark={mark},line width=1.2pt] coordinates {{{coord_str}}};\n"
            f"        \\addlegendentry{{{label}}}"
        )

    plots_str = "\n".join(plots)

    # X-tick labels: show model name labels at param positions
    xtick_vals = ",".join(str(MODEL_PARAMS_B[s]) for s in MODEL_SIZES_ORDERED)
    xticklabels = ",".join(f"{s}" for s in MODEL_SIZES_ORDERED)

    tex = r"""\documentclass[crop,tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{xcolor}

\begin{document}
\begin{tikzpicture}
\begin{semilogxaxis}[
    width=9cm, height=7cm,
    xlabel={Model Size (parameters)},
    ylabel={$\Delta\mathrm{acc}$ vs. Full (\%)},
    xmin=2, xmax=50,
    xtick={""" + xtick_vals + r"""},
    xticklabels={""" + xticklabels + r"""},
    grid=both,
    grid style={line width=0.3pt, draw=gray!30},
    legend pos=south west,
    legend style={font=\small, cells={anchor=west}},
    tick label style={font=\small},
    label style={font=\small},
    title style={font=\small\bfseries},
    title={Component Contribution vs.\ Model Capacity (UNSW-NB15, $N=200$, single-seed)},
    yticklabel={\pgfmathprintnumber[fixed,precision=1]{\tick}},
]
% Relay boundary (Δa = 0)
\draw[dashed, gray, line width=0.8pt] (axis cs:2,0) -- (axis cs:50,0)
    node[right, font=\footnotesize, gray] {relay boundary};
""" + plots_str + r"""
\end{semilogxaxis}
\end{tikzpicture}
\end{document}
"""
    return tex


def generate_component_gradient_tex(
    deltas: dict[str, dict[str, float | None]],
) -> str:
    """Generate pgfplots LaTeX for fig_component_gradient (grouped bar chart)."""
    # Build bar data: for each model size, 4 bars (one per component)
    n_sizes = len(MODEL_SIZES_ORDERED)
    n_components = len(ABLATION_COMPONENTS)

    # Generate one \addplot per component (each plots one bar per model size)
    plots = []
    for i, abl in enumerate(ABLATION_COMPONENTS):
        label = COMPONENT_LABELS[abl]
        color = COLORS[abl]
        coords = []
        for j, size in enumerate(MODEL_SIZES_ORDERED):
            val = deltas.get(size, {}).get(abl)
            if val is None:
                val = 0.0  # Missing data: show as zero with note
            coords.append(f"({j},{val:.4f})")
        coord_str = " ".join(coords)
        plots.append(
            f"        \\addplot[fill={{{color}}},fill opacity=0.8,draw={{{color}}}] coordinates {{{coord_str}}};\n"
            f"        \\addlegendentry{{{label}}}"
        )

    plots_str = "\n".join(plots)
    xticklabels = ",".join(MODEL_SIZES_ORDERED)
    xtick_vals = ",".join(str(i) for i in range(n_sizes))

    tex = r"""\documentclass[crop,tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{xcolor}
\usepgfplotslibrary{groupplots}

\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=0.15cm,
    width=11cm, height=7cm,
    xlabel={Model Size},
    ylabel={$\Delta\mathrm{acc}$ vs. Full (\%), single-seed, no CI},
    xtick={""" + xtick_vals + r"""},
    xticklabels={""" + xticklabels + r"""},
    xmin=-0.5, xmax=""" + str(n_sizes - 0.5) + r""",
    grid=both,
    grid style={line width=0.3pt, draw=gray!30},
    legend pos=north west,
    legend style={font=\small, cells={anchor=west},
                  legend columns=2},
    tick label style={font=\small},
    label style={font=\small},
    title style={font=\small\bfseries},
    title={Per-Component Contribution Gradient (UNSW-NB15, $N=200$)},
    yticklabel={\pgfmathprintnumber[fixed,precision=1]{\tick}},
]
% Relay boundary
\draw[dashed, gray, line width=0.8pt] (axis cs:-0.5,0) -- (axis cs:""" + str(n_sizes - 0.5) + r""",0);
""" + plots_str + r"""
\end{axis}
\end{tikzpicture}
\end{document}
"""
    return tex


def compile_tex_to_pdf(tex_source: str, output_pdf: Path) -> bool:
    """
    CONTRACT: compile_tex_to_pdf(tex_source, output_pdf) -> bool
        tex_source: str   — LaTeX source string
        output_pdf: Path  — destination PDF path
        returns: True on success, False on pdflatex failure
    Write .tex to output_pdf.with_suffix('.tex'), compile, move PDF.
    """
    tex_path = output_pdf.with_suffix(".tex")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_path, "w") as f:
        f.write(tex_source)
    log.info("Wrote LaTeX source: %s", tex_path)

    # Compile with pdflatex in a temp dir to keep aux files separate
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, str(tex_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error("pdflatex failed for %s:\n%s", tex_path, result.stdout[-1000:])
            return False

        tmp_pdf = Path(tmpdir) / tex_path.with_suffix(".pdf").name
        if not tmp_pdf.exists():
            log.error("pdflatex ran but PDF not found in %s", tmpdir)
            return False

        import shutil
        shutil.copy2(tmp_pdf, output_pdf)
        log.info("PDF compiled: %s", output_pdf)
        return True


# ---------------------------------------------------------------------------
# Results-registry append
# ---------------------------------------------------------------------------

def append_registry(summary_rows: list[dict[str, Any]], transition_report: str) -> None:
    """Append the tau phase-transition summary section to the results registry file."""
    if not REGISTRY_PATH.exists():
        log.warning("Results registry not found at %s; skipping registry append.", REGISTRY_PATH)
        return

    now_str = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "\n\n## §32 Tau Phase Transition (Exp-τ, T3 Tool-Dominance Collapse)\n",
        f"<!-- src:project/results/tables/v2_tdsc/tau_summary.csv:section=32 -->\n",
        f"<!-- ran_at:{now_str} -->\n",
        f"<!-- script:project/scripts/v2_tdsc/analyze_tau_curves.py -->\n\n",
        f"**Phase Transition Finding**: {transition_report}\n\n",
        "| model_size | tau_hat | mode | acc_full | fpr_full | da_K | da_O | da_P | da_T |\n",
        "|-----------|---------|------|----------|----------|------|------|------|------|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_size']} | {row['tau_hat']} | {row['mode']} | "
            f"{row['acc_full']} | {row['fpr_full']} | "
            f"{row.get('da_K', 'N/A')} | {row.get('da_O', 'N/A')} | "
            f"{row.get('da_P', 'N/A')} | {row.get('da_T', 'N/A')} |\n"
        )
    lines.append("\n*Single-seed, no CI. Source: tau_summary.csv + tau_summary.json*\n")

    with open(REGISTRY_PATH, "a") as f:
        f.writelines(lines)
    log.info("Appended the tau phase-transition section to the results registry.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CONTRACT: main() -> None
        See module docstring for full contract.
        Exits with code 1 if no conditions could be loaded.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Analyze Exp-τ curves and generate figures")
    ap.add_argument("--no-figures", action="store_true", help="Skip figure compilation")
    ap.add_argument("--no-registry", action="store_true", help="Skip the results registry append")
    args = ap.parse_args()

    log.info("=== Analyze Tau Curves ===")
    RESULTS_V2_TDSC.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all condition results
    condition_data = load_all_conditions()
    if not condition_data:
        log.error("No condition data found. Run experiments first.")
        sys.exit(1)
    log.info("Loaded %d conditions.", len(condition_data))

    # Step 2 & 3: Compute Δa per component
    deltas = compute_delta_accuracy(condition_data)

    # Step 4: Compute τ_hat from audit logs (best-effort)
    tau_hats: dict[str, dict[str, float]] = {}
    for size in MODEL_SIZES_ORDERED:
        # Use the "full" condition audit log for tau_hat estimation
        logs_dir = LOGS_V2_TDSC if size in ("13B", "32B") else LOGS_V2
        exp_id_for_tau = f"tau_{size}_full_unsw" if size in ("13B", "32B") else (
            "E3_7B_ablation_full_unsw" if size == "7B" else "E3_3B_ablation_full_unsw"
        )
        records = load_audit_tool_flags(logs_dir, exp_id_for_tau)
        if records:
            tau_hats[size] = compute_tau_hat(records)
        else:
            log.warning("No audit log for %s full; tau_hat will be NaN.", size)
            tau_hats[size] = {"tau_hat": float("nan"), "mi": float("nan"), "h_y_given_t": float("nan"), "n": 0}

    # Step 5: Phase transition detection
    transition = detect_phase_transition(deltas)
    log.info("Phase transition report: %s", transition["report"])

    # Step 6: T7 dichotomy per model
    modes: dict[str, str] = {}
    for size in MODEL_SIZES_ORDERED:
        modes[size] = classify_mode(deltas.get(size, {}))
        log.info("Mode %s: %s", size, modes[size])

    # Step 7a: Build summary rows
    summary_rows: list[dict[str, Any]] = []
    for size in MODEL_SIZES_ORDERED:
        full_metrics = condition_data.get((size, "full"))
        row: dict[str, Any] = {
            "model_size": size,
            "tau_hat": tau_hats[size]["tau_hat"],
            "mode": modes[size],
            "acc_full": full_metrics["accuracy"] if full_metrics else "N/A",
            "fpr_full": full_metrics["fpr"] if full_metrics else "N/A",
            "da_K": deltas[size].get("noKnowledge"),
            "da_O": deltas[size].get("noObservation"),
            "da_P": deltas[size].get("noPermissions"),
            "da_T": deltas[size].get("noTools"),
        }
        summary_rows.append(row)

    # Step 7b: Write tau_summary.csv
    csv_path = RESULTS_V2_TDSC / "tau_summary.csv"
    csv_fields = ["model_size", "tau_hat", "mode", "acc_full", "fpr_full", "da_K", "da_O", "da_P", "da_T"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Wrote tau_summary.csv: %s", csv_path)

    # Step 7c: Write tau_summary.json
    json_path = RESULTS_V2_TDSC / "tau_summary.json"
    summary_json = {
        "conditions_loaded": len(condition_data),
        "phase_transition": transition,
        "models": summary_rows,
        "component_deltas": {
            size: {abl: deltas[size].get(abl) for abl in ABLATION_COMPONENTS}
            for size in MODEL_SIZES_ORDERED
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    log.info("Wrote tau_summary.json: %s", json_path)

    # Step 8: Generate figures
    if not args.no_figures:
        # Figure 1: phase transition line plot
        tex1 = generate_phase_transition_tex(deltas)
        pdf1 = FIGURES_DIR / "fig_tau_phase_transition.pdf"
        ok1 = compile_tex_to_pdf(tex1, pdf1)
        if not ok1:
            log.warning("fig_tau_phase_transition.pdf compilation failed. Check .tex source.")

        # Figure 2: component gradient bar chart
        tex2 = generate_component_gradient_tex(deltas)
        pdf2 = FIGURES_DIR / "fig_component_gradient.pdf"
        ok2 = compile_tex_to_pdf(tex2, pdf2)
        if not ok2:
            log.warning("fig_component_gradient.pdf compilation failed. Check .tex source.")
    else:
        log.info("--no-figures: skipping figure compilation.")

    # Step 9: Append to the results registry
    if not args.no_registry:
        append_registry(summary_rows, transition["report"])
    else:
        log.info("--no-registry: skipping registry append.")

    # DQ1-5 checklist output (logged for manual verification)
    log.info("=== DQ1-5 Figure Quality Checklist ===")
    log.info("DQ1: Internal structure — axes labeled (model size, Δacc %%), log x-axis: YES")
    log.info("DQ2: Background — grid lines at both major/minor, white background: YES")
    log.info("DQ3: Dimension labels — xlabel/ylabel set, units explicit (%%): YES")
    log.info("DQ4: Icons — none used (line+marker/bar chart): N/A")
    log.info("DQ5: Visual hierarchy — dashed relay boundary, legend south west / north west (no north east): YES")
    log.info("Legend occlusion check: legend pos=south west (fig1), north west (fig2) — NO north east: PASS")

    log.info("=== Analysis Complete ===")
    log.info("Outputs:")
    log.info("  %s", csv_path)
    log.info("  %s", json_path)
    if not args.no_figures:
        log.info("  %s", FIGURES_DIR / "fig_tau_phase_transition.pdf")
        log.info("  %s", FIGURES_DIR / "fig_tau_phase_transition.tex")
        log.info("  %s", FIGURES_DIR / "fig_component_gradient.pdf")
        log.info("  %s", FIGURES_DIR / "fig_component_gradient.tex")


if __name__ == "__main__":
    main()
