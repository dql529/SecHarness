#!/usr/bin/env python3
"""C1 + C4 — trivial baselines and precision at an operational prior.

C1: the paper reports a 64.5pp gain of E3 over E1, but E1 (27.9%) is worse
than a constant classifier. Report every trivial baseline so the effect size is honest.

C4: the evaluation set is 68% attack, which inverts real base rates.
Recompute precision at realistic priors from the measured TPR/FPR, which are prior-free.

Confusion matrix source: the published main table, E3/E4 on UNSW-NB15 N=1000
(TP 642 / TN 282 / FP 38 / FN 38), identical to the RF baseline.
"""
from __future__ import annotations

TP, TN, FP, FN = 642, 282, 38, 38
N = TP + TN + FP + FN
ATTACK = TP + FN
BENIGN = TN + FP
TPR = TP / ATTACK
FPR = FP / BENIGN


def f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


print(f"evaluation set: N={N}  attack={ATTACK} ({ATTACK/N:.1%})  benign={BENIGN} ({BENIGN/N:.1%})")
print(f"system operating point: TPR={TPR:.3f}  FPR={FPR:.3f}\n")

print("=== C1: trivial baselines ===")
rows = [
    ("always-attack (= majority)", ATTACK / N, 1.0, f1(ATTACK, BENIGN, 0), 1.0),
    ("always-benign",              BENIGN / N, 0.0, 0.0, 0.0),
    ("E1 zero-shot (paper)",       0.279, None, None, None),
    ("E1-RF / RF alone / E3 / E4", (TP + TN) / N, FPR, f1(TP, FP, FN), TPR),
]
print(f"{'baseline':30s} {'Acc':>7} {'FPR':>7} {'F1':>7} {'recall':>7}")
for name, acc, fpr, f, rec in rows:
    fs = f"{fpr:.3f}" if fpr is not None else "   -"
    ff = f"{f:.3f}" if f is not None else "   -"
    rs = f"{rec:.3f}" if rec is not None else "   -"
    print(f"{name:30s} {acc:7.3f} {fs:>7} {ff:>7} {rs:>7}")

print(f"""
Reading: the always-attack constant scores {ATTACK/N:.1%} accuracy and F1 {f1(ATTACK,BENIGN,0):.3f}
with zero intelligence. E1's 27.9% is BELOW BOTH constants ({ATTACK/N:.1%} and {BENIGN/N:.1%}),
so "+64.5pp over E1" measures the distance from a degenerate baseline, not system value.
Honest effect sizes are vs the always-attack constant (+{(TP+TN)/N - ATTACK/N:.3f}) and vs
RF alone / E1-RF (0.000 accuracy difference).""")

print("\n=== C4: precision at operational priors ===")
print(f"{'attack prior':>13} {'precision':>10} {'alerts/1000 flows':>19} {'true alerts':>12}")
for p in (0.68, 0.20, 0.10, 0.05, 0.01, 0.001):
    tps = p * TPR
    fps = (1 - p) * FPR
    prec = tps / (tps + fps) if (tps + fps) else 0.0
    alerts = (tps + fps) * 1000
    print(f"{p:12.1%} {prec:10.3f} {alerts:19.1f} {tps*1000:12.1f}")

p1 = 0.01
prec1 = (p1 * TPR) / (p1 * TPR + (1 - p1) * FPR)
print(f"""
Reading: at the evaluated 68% prior precision looks strong, but at a 1% operational prior
the same operating point yields precision {prec1:.1%} — roughly {1/prec1:.0f} false alerts per true
detection. The FPR of {FPR:.1%} is the binding constraint in deployment, not accuracy.
This must be stated when the paper claims deployment readiness.""")
