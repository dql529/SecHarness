"""
generate_scaling_figure.py — coverage vs deployed-agent ZD across model capacity.

1x3 panels (Llama-3.2-3B | Qwen2.5-7B | Qwen2.5-14B), x = kNN coverage,
y = harness zero-day detection rate, CIC (blue circles) vs UNSW (red triangles),
per-dataset OLS + pooled Pearson r annotation. Visualises the relay->exploration
decoupling: coverage predicts the 3B relay tightly and the 14B agent barely.

Reads results/tables/candidate_b/harness_zd_{3b,qwen7b,qwen14b}.csv + coverage_variants.csv
Writes results/figures/candidate_b_scaling.tex (standalone -> PDF).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables" / "candidate_b"
FIGDIR = ROOT / "results" / "figures"

# The claim this figure carries is capacity *within one family* (Section 4.8: "identical harness
# and coverage"). The first panel was Llama-3.2-3B, which crosses the family boundary the text
# holds fixed and makes the declining series unreadable as a capacity effect. Qwen2.5-3B is the
# matched cell and its pooled r (0.924) is the value the text quotes; the Llama agent (r=0.938)
# is the separate family-at-fixed-capacity control, reported in the text, not plotted here.
PANELS = [("qwen3b", "Qwen2.5-3B"), ("qwen7b", "Qwen2.5-7B"), ("qwen14b", "Qwen2.5-14B")]
DSETS = [("CIC-IDS2017", "plotblue", "*", "CIC-IDS2017"),
         ("UNSW-NB15", "plotred", "triangle*", "UNSW-NB15")]


def ols(x, y):
    """OLS endpoints clipped to the valid rate band y in [0,1] (no extrapolation past 0/1)."""
    b, a = np.polyfit(x, y, 1)
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
    cov = pd.read_csv(TAB / "coverage_variants.csv")[["dataset", "held_class", "cov_knn_attack"]]
    L = [
        r"\documentclass[border={4pt 4pt 4pt 4pt}]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{libertine}",
        r"\usepackage{pgfplots}\pgfplotsset{compat=1.18}",
        r"\usepgfplotslibrary{groupplots}",
        r"\definecolor{plotblue}{HTML}{2166AC}", r"\definecolor{plotred}{HTML}{D6604D}",
        r"\begin{document}", r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.55cm},",
        r"  width=5.2cm, height=5.0cm, xmin=-0.02, xmax=1.02, ymin=-0.02, ymax=1.02,",
        r"  xlabel={coverage $\mathrm{cov}(c)$}, xlabel style={font=\small},",
        r"  tick label style={font=\footnotesize}, xtick={0,0.5,1}, ytick={0,0.5,1},",
        r"  grid=major, grid style={gray!15}, axis line style={gray!60},]",
    ]
    for i, (tag, name) in enumerate(PANELS):
        df = pd.read_csv(TAB / f"harness_zd_{tag}.csv")[["dataset", "held_class", "harness_zd_rate"]]
        df = df.merge(cov, on=["dataset", "held_class"])
        summ = json.load(open(TAB / f"harness_zd_{tag}_summary.json"))
        pooled_r = summ["pooled"]["pearson_r"]
        ylab = (r"ylabel={zero-day detection rate}, ylabel style={font=\small},"
                if i == 0 else "yticklabel=\\empty,")
        leg = (r"legend style={at={(0.03,0.03)}, anchor=south west, font=\scriptsize,"
               r" draw=gray!50, fill=white, fill opacity=0.85, text opacity=1, row sep=-1pt},"
               r" legend cell align=left,") if i == len(PANELS) - 1 else "legend style={draw=none},"
        L.append(f"\\nextgroupplot[title={{\\footnotesize {name} (pooled $r{{=}}{pooled_r:.2f}$)}}, {ylab} {leg}]")
        for dname, color, mark, leglabel in DSETS:
            g = df[df.dataset == dname]
            coords = " ".join(f"({r.cov_knn_attack:.4f},{r.harness_zd_rate:.4f})"
                              for r in g.itertuples())
            L.append(f"\\addplot[only marks, mark={mark}, mark size=1.6pt, color={color}] "
                     f"coordinates {{{coords}}};")
            if i == len(PANELS) - 1:
                L.append(f"\\addlegendentry{{{leglabel}}}")
            x0, y0, x1, y1 = ols(g.cov_knn_attack.values, g.harness_zd_rate.values)
            L.append(f"\\addplot[{color}, thick, forget plot] coordinates "
                     f"{{({x0:.4f},{y0:.4f}) ({x1:.4f},{y1:.4f})}};")
        # pooled r is carried in the panel title (an in-axis node sat on the UNSW fit line at 7B/14B)
    L += [r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}"]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "candidate_b_scaling.tex").write_text("\n".join(L))
    print("wrote", FIGDIR / "candidate_b_scaling.tex")


if __name__ == "__main__":
    main()
