#!/usr/bin/env python3
"""Draw the risk--coverage curves behind Section 4.9's review-budget numbers.

The section reports three numbers -- 40.5% of the alert stream reviewed under the harness,
43.5% for the detector alone, 53.5% for the same model without a harness -- and one comparison
that only a curve makes legible: the harness and the tool nearly coincide, because the harness
relays the tool's confidence, while the unharnessed model replaces 111 distinct confidence
values with 11 and pays for it across the whole curve.

Everything is recomputed from the same audit logs and the same tie-aware procedure the gated
table uses, by importing that module rather than reimplementing it: a figure drawn from a
second implementation of the statistic would not be evidence about the first.

Usage: python3 d3_risk_coverage_curve.py <project_root> [--out <pdf>]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_d3():
    spec = importlib.util.spec_from_file_location("d3", HERE / "d3_operational_quality.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves its own module through sys.modules, and a
    # module absent from it makes every dataclass in the file raise on definition.
    sys.modules["d3"] = mod
    spec.loader.exec_module(mod)
    return mod


def curve(pairs: list[tuple[float, bool]]) -> tuple[list[float], list[float]]:
    """Risk as a function of coverage, accepting the most confident tie groups first."""
    groups: dict[float, list[bool]] = {}
    for conf, ok in pairs:
        groups.setdefault(conf, []).append(ok)
    n = len(pairs)
    cov, risk = [], []
    acc_n = acc_err = 0
    for conf in sorted(groups, reverse=True):
        oks = groups[conf]
        acc_n += len(oks)
        acc_err += sum(1 for ok in oks if not ok)
        cov.append(acc_n / n)
        risk.append(acc_err / acc_n)
    return cov, risk


def harness_pairs(d3, recs) -> list[tuple[float, bool]]:
    pairs = []
    for rec in recs:
        conf = (rec.get("result") or {}).get("confidence")
        ok = d3.is_correct(rec)
        if conf is not None and ok is not None:
            pairs.append((float(conf), ok))
    return pairs


def tool_pairs(d3, recs) -> list[tuple[float, bool]]:
    pairs = []
    for rec in recs:
        mls = d3.valid_ml_payloads(rec)
        gt = (rec.get("evaluation") or {}).get("binary_gt")
        if not mls or gt is None:
            continue
        ml = mls[0]
        pred = "attack" if ml.get("prediction") != "Normal" else "benign"
        pairs.append((float(ml["confidence"]), pred == gt))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_root", nargs="?", default=".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    d3 = load_d3()
    d3.PROJECT_ROOT = root

    logs = {
        "E3": "logs/v2/lat_E3_unsw_sub200_audit.jsonl",
        "E1RF": "logs/v2/lat_E1_rf_context_unsw_sub200_audit.jsonl",
        "E4": "logs/v2/lat_E4_unsw_sub200_audit.jsonl",
    }
    recs = {}
    for key, rel in logs.items():
        path = root / rel
        if not path.exists():
            print(f"MISSING LOG: {path}", file=sys.stderr)
            return 2
        recs[key] = d3.load_records(path)

    series = {
        "RF only": tool_pairs(d3, recs["E3"]),
        "E3": harness_pairs(d3, recs["E3"]),
        "E4": harness_pairs(d3, recs["E4"]),
        "E1-RF": harness_pairs(d3, recs["E1RF"]),
    }

    out_pdf = Path(args.out) if args.out else root.parent / "submission/figures/fig_risk_coverage.pdf"
    out_csv = out_pdf.with_suffix(".csv")

    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "coverage", "risk"])
        for name, pairs in series.items():
            for c, r in zip(*curve(pairs)):
                w.writerow([name, f"{c:.6f}", f"{r:.6f}"])

    # The number the section quotes: the fraction that must be read to reach zero residual risk.
    budgets = {name: d3.tie_aware_risk_coverage(p)["rc_review_for_all_errors"]
               for name, p in series.items()}
    for name, b in budgets.items():
        n_conf = len({c for c, _ in series[name]})
        print(f"{name:32s} review={b:.3f}  distinct_conf={n_conf}  n={len(series[name])}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "RF only": dict(color="#AA3377", ls="--", lw=1.6, zorder=3),
        "E3": dict(color="#4477AA", ls="-", lw=2.0, zorder=4),
        "E4": dict(color="#66CCEE", ls="-", lw=1.3, zorder=2),
        "E1-RF": dict(color="#BBBBBB", ls="-", lw=1.8, zorder=1),
    }

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    for name, pairs in series.items():
        c, r = curve(pairs)
        ax.plot([100 * x for x in c], [100 * y for y in r], label=name, **style[name])

    for name in ("E3", "RF only", "E1-RF"):
        x = 100 * (1 - budgets[name])
        ax.axvline(x, color=style[name]["color"], lw=0.7, ls=":", alpha=0.8, zorder=0)

    ax.set_xlabel("Alert stream accepted without review (%)", fontsize=8)
    ax.set_ylabel("Errors among accepted (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")


    fig.tight_layout(pad=0.3)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    print(f"\nwrote {out_pdf}\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
