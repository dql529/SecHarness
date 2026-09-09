"""
generate_figure.py — Candidate B coverage-vs-ZD scatter (Tool 1 pgfplots).

Emits a standalone TikZ/pgfplots figure: 1x3 panels (LogReg | HistGB | RF),
x = kNN-attack-fraction coverage, y = zero-day detection rate, CIC (blue circles)
vs UNSW (red triangles), per-dataset OLS regression lines + Pearson r/p annotation.

Reads:  results/tables/candidate_b/multidetector_validation.csv
        results/tables/candidate_b/multidetector_validation_summary.json
Writes: results/figures/candidate_b_coverage_zd.tex  (standalone, compile to PDF)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables" / "candidate_b"
FIGDIR = ROOT / "results" / "figures"

PANELS = [("zd_logreg", "Logistic Reg.\\ (linear)"),
          ("zd_histgb", "Hist.\\ Grad.\\ Boost (boosting)"),
          ("zd_rf", "Random Forest (bagging)")]
DSETS = [("CIC-IDS2017", "plotblue", "*", "CIC-IDS2017"),
         ("UNSW-NB15", "plotred", "triangle*", "UNSW-NB15")]


# Which cells earn a star is the manuscript's criterion, not an uncorrected p-value.
#
# This starred on raw Pearson p < 0.05, which is the pass criterion we pre-registered and then
# ABANDONED after seeing the six correlations (Section 4.8 says so in as many words). Under it
# both UNSW-NB15 tree cells star -- Pearson p = 6.2e-4 (RF) and 3.1e-4 (GB) -- while the text
# they illustrate says those two do not pass: their bootstrap CIs cross zero and neither
# survives Holm correction on Spearman. A reader comparing figure to text would have found the
# figure asserting what the text withdraws.
#
# The criterion of Table 10, and therefore of this figure, stated verbatim from its caption:
# Pearson AND Spearman both Holm-significant, AND the Pearson CI excluding zero. Read from the
# same CSV the table is built from, so the two cannot drift apart.
HOLM = ROOT / "results" / "tables" / "candidate_b" / "table8_spearman_ci_holm.csv"

_FAMILY_BY_COL = {
    "zd_rf": "Random forest (bagging)",
    "zd_histgb": "Grad. boosting (boosting)",
    "zd_logreg": "Logistic reg. (linear)",
}


def passes_manuscript_criterion(dataset: str, col: str) -> bool:
    fam = _FAMILY_BY_COL.get(col)
    if fam is None:
        raise KeyError(f"no family mapping for column {col!r} -- add it before starring it")
    h = pd.read_csv(HOLM)
    row = h[(h.dataset == dataset) & (h.family == fam)]
    if len(row) != 1:
        raise LookupError(f"{dataset}/{fam}: expected 1 row in {HOLM.name}, got {len(row)}")
    r = row.iloc[0]
    return (bool(r.pearson_sig_holm_05) and bool(r.spearman_sig_holm_05)
            and bool(r.ci_excludes_zero_pearson))


def ols_line(x: np.ndarray, y: np.ndarray):
    """Return endpoints (x0,y0,x1,y1) of the OLS fit over the data x-range,
    clipped so the drawn segment stays within the valid rate band y in [0,1]
    (detection rate is bounded; the line must not extrapolate past 0 or 1)."""
    b, a = np.polyfit(x, y, 1)  # y = b*x + a
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = a + b * x0, a + b * x1

    def x_at(yb):
        return (yb - a) / b if abs(b) > 1e-9 else x0

    if y0 < 0.0:
        x0, y0 = x_at(0.0), 0.0
    elif y0 > 1.0:
        x0, y0 = x_at(1.0), 1.0
    if y1 < 0.0:
        x1, y1 = x_at(0.0), 0.0
    elif y1 > 1.0:
        x1, y1 = x_at(1.0), 1.0
    return x0, y0, x1, y1


def main() -> None:
    df = pd.read_csv(TAB / "multidetector_validation.csv")
    with open(TAB / "multidetector_validation_summary.json") as f:
        summ = json.load(f)

    L = [
        r"\documentclass[border={4pt 4pt 4pt 4pt}]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{libertine}",
        r"\usepackage{pgfplots}\pgfplotsset{compat=1.18}",
        r"\usepgfplotslibrary{groupplots}",
        r"\definecolor{plotblue}{HTML}{2166AC}",
        r"\definecolor{plotred}{HTML}{D6604D}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 1, horizontal sep=0.55cm},",
        r"  width=5.2cm, height=5.0cm,",
        r"  xmin=-0.02, xmax=1.02, ymin=-0.02, ymax=1.02,",
        r"  xlabel={coverage $\mathrm{cov}(c)$}, xlabel style={font=\small},",
        r"  tick label style={font=\footnotesize},",
        r"  xtick={0,0.5,1}, ytick={0,0.5,1},",
        r"  grid=major, grid style={gray!15},",
        r"  axis line style={gray!60},",
        r"]",
    ]

    for i, (col, title) in enumerate(PANELS):
        ylab = r"ylabel={zero-day detection rate}, ylabel style={font=\small}," if i == 0 \
            else "yticklabel=\\empty,"
        # legend only on last (RF) panel, in the empty top-left region
        leg = (r"legend style={at={(0.03,0.97)}, anchor=north west, font=\scriptsize, "
               r"draw=gray!50, fill=white, fill opacity=0.85, text opacity=1, "
               r"row sep=-1pt}, legend cell align=left,") if i == len(PANELS) - 1 \
            else "legend style={draw=none},"
        L.append(f"\\nextgroupplot[title={{\\footnotesize {title}}}, {ylab} {leg}]")
        ann = []
        for dname, color, mark, leglabel in DSETS:
            g = df[df.dataset == dname]
            x = g["cov_knn_attack"].to_numpy()
            y = g[col].to_numpy()
            coords = " ".join(f"({xi:.4f},{yi:.4f})" for xi, yi in zip(x, y))
            L.append(f"\\addplot[only marks, mark={mark}, mark size=1.6pt, "
                     f"color={color}] coordinates {{{coords}}};")
            if i == len(PANELS) - 1:
                L.append(f"\\addlegendentry{{{leglabel}}}")
            x0, y0, x1, y1 = ols_line(x, y)
            L.append(f"\\addplot[{color}, thick, forget plot] coordinates "
                     f"{{({x0:.4f},{y0:.4f}) ({x1:.4f},{y1:.4f})}};")
            b = summ["detectors"][col][dname]
            star = "$^{*}$" if passes_manuscript_criterion(dname, col) else ""
            tag = "CIC" if dname == "CIC-IDS2017" else "UNSW"
            ann.append(f"{tag} $r{{=}}{b['pearson_r']:.2f}${star}")
        # per-dataset r annotation, lower-right (sparse for these positive trends)
        L.append(r"\node[anchor=south east, font=\scriptsize, align=right, "
                 r"fill=white, fill opacity=0.8, text opacity=1, inner sep=1.5pt] "
                 r"at (rel axis cs:0.98,0.03) {" + r"\\".join(ann) + "};")

    L += [r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}"]

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "candidate_b_coverage_zd.tex"
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
