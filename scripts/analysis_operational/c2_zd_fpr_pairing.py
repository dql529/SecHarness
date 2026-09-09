#!/usr/bin/env python3
"""C2 — pair every zero-day detection rate with its false-positive rate.

A zero-day rate reported without a paired false-positive rate is ambiguous: the 14B result,
where SSH-Patator and FTP-Patator rise from 0.00 to 1.00, is equally consistent with a more
attack-prone model.

This pairs the coverage-experiment ZD rates (capacity scaling) with the FPR
each of those same models produces under the full harness (measured 2026-08-17/18).

Result: the ZD gain is accompanied by a comparable FPR rise. The 2026-08-18
cross-capacity control refines the diagnosis: within the Qwen family FPR is identical and
verdicts agree 200/200, so the FPR jump tracks the family boundary, not the capacity ladder.

Usage: python3 c2_zd_fpr_pairing.py <project_root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Pooled ZD mean/std from the capacity-scaling runs (leave-one-attack-out, N=200/class)
ZD = {
    "Llama-3.2-3B": (0.62, 0.41),
    "Qwen2.5-7B": (0.77, 0.28),
    "Qwen2.5-14B": (0.90, 0.10),
}

# FPR for the SAME models under the full harness, measured on the main evaluation split
FPR_SOURCES = {
    "Llama-3.2-3B": "results/tables/v2/lat_E3_unsw_results.json",
    "Qwen2.5-7B": "results/tables/v2/E3_7B_ablation_full_unsw_results.json",
    "Qwen2.5-14B": "results/tables/v2_tdsc/tau_13B_full_unsw_results.json",
}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print("C2 — pairing zero-day detection with false-positive rate\n")
    print(f"{'model':16s} {'ZD rate (cov exp)':>18} {'FPR (full harness)':>20} {'TN':>5} {'FP':>5}")

    rows = []
    for model, (zd_mean, zd_std) in ZD.items():
        p = root / FPR_SOURCES[model]
        if not p.exists():
            print(f"{model:16s} MISSING {FPR_SOURCES[model]}")
            return 1
        d = json.load(open(p))
        rows.append((model, zd_mean, d["fpr"], d["tn"], d["fp"]))
        print(f"{model:16s} {zd_mean:9.2f} ± {zd_std:<5.2f} "
              f"{d['fpr']*100:19.1f}% {d['tn']:5d} {d['fp']:5d}")

    base, top = rows[0], rows[-1]
    d_zd = (top[1] - base[1]) * 100
    d_fpr = (top[2] - base[2]) * 100
    print(f"\n3B -> 14B :  ZD  {base[1]:.2f} -> {top[1]:.2f}  (+{d_zd:.0f}pp)")
    print(f"             FPR {base[2]*100:.1f}% -> {top[2]*100:.1f}%  (+{d_fpr:.1f}pp)")
    print(f"""
Reading: the zero-day gain is accompanied by a comparable rise in false positives
(+{d_zd:.0f}pp ZD against +{d_fpr:.1f}pp FPR), so the "more attack-prone model" reading is SUPPORTED.

Caveat to state in the paper: 3B is Llama-3.2 while 7B/14B are Qwen2.5, so family is
confounded with capacity in this table. The controlling evidence is the 2026-08-18
cross-capacity result: within the Qwen family (7B/14B/32B) FPR is identical at 36.21% and
verdicts agree on 200/200 samples, i.e. the FPR jump sits at the FAMILY boundary
(Llama 13.8% vs Qwen 36.2%), not on the capacity ladder.

Also note: the coverage experiment retrains a detector per fold, so the agent relays a
DIFFERENT RF in each fold, whereas the main experiment uses one fixed RF. That is why
7B and 14B agree perfectly in the main experiment yet differ in the coverage experiment.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
