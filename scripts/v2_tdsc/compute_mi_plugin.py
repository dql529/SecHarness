#!/usr/bin/env python3
"""
compute_mi_plugin.py — Plug-in mutual information estimator with Laplace smoothing.

CONTRACT:
    plugin_mi(xs, ys, ts=None, laplace_alpha=0.5) -> dict
        xs: list[Hashable]        — categorical observations of component output
        ys: list[Hashable]        — categorical observations of ground-truth label
        ts: list[Hashable] | None — conditioning variable (tool output); None = unconditional MI
        laplace_alpha: float      — Dirichlet smoothing parameter (default 0.5)
        returns: dict with keys {mi, h_x, h_y, h_y_given_t, n}
            mi: float             — I(X;Y) or I(X;Y|T) in nats
            h_x: float            — H(X) in nats
            h_y: float            — H(Y) in nats
            h_y_given_t: float    — H(Y|T) in nats (nan if ts is None)
            n: int                — number of observations

    conditional_entropy(xs, ys, laplace_alpha=0.5) -> float
        Computes H(Y|X) in nats with Laplace smoothing.

    joint_entropy(xs, ys, laplace_alpha=0.5) -> float
        Computes H(X,Y) in nats with Laplace smoothing.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Hashable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core entropy helpers
# ---------------------------------------------------------------------------

def _marginal_entropy(xs: list[Hashable], laplace_alpha: float) -> float:
    """
    CONTRACT: _marginal_entropy(xs, laplace_alpha) -> float
        xs: list[Hashable]     — categorical sequence
        laplace_alpha: float   — Dirichlet smoothing (>=0)
        returns: H(X) in nats with Laplace smoothing
    """
    if not xs:
        return 0.0
    n = len(xs)
    counts = Counter(xs)
    vocab_size = len(counts)
    denom = n + laplace_alpha * vocab_size
    h = 0.0
    for cnt in counts.values():
        p = (cnt + laplace_alpha) / denom
        h -= p * math.log(p)
    return h


def joint_entropy(
    xs: list[Hashable],
    ys: list[Hashable],
    laplace_alpha: float = 0.5,
) -> float:
    """
    CONTRACT: joint_entropy(xs, ys, laplace_alpha=0.5) -> float
        xs: list[Hashable]     — first variable
        ys: list[Hashable]     — second variable
        laplace_alpha: float   — Dirichlet smoothing
        returns: H(X,Y) in nats
    Raises ValueError if len(xs) != len(ys).
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have equal length, got {len(xs)} vs {len(ys)}")
    if not xs:
        return 0.0

    n = len(xs)
    joint_counts: Counter[tuple[Hashable, Hashable]] = Counter(zip(xs, ys))
    vocab_size = len(joint_counts)
    denom = n + laplace_alpha * vocab_size
    h = 0.0
    for cnt in joint_counts.values():
        p = (cnt + laplace_alpha) / denom
        h -= p * math.log(p)
    return h


def conditional_entropy(
    xs: list[Hashable],
    ys: list[Hashable],
    laplace_alpha: float = 0.5,
) -> float:
    """
    CONTRACT: conditional_entropy(xs, ys, laplace_alpha=0.5) -> float
        xs: list[Hashable]     — conditioning variable X
        ys: list[Hashable]     — target variable Y
        laplace_alpha: float   — Dirichlet smoothing
        returns: H(Y|X) = H(X,Y) - H(X) in nats
    Raises ValueError if len(xs) != len(ys).
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have equal length, got {len(xs)} vs {len(ys)}")
    if not xs:
        return 0.0
    h_joint = joint_entropy(xs, ys, laplace_alpha)
    h_x = _marginal_entropy(xs, laplace_alpha)
    return h_joint - h_x


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def plugin_mi(
    xs: list[Hashable],
    ys: list[Hashable],
    ts: list[Hashable] | None = None,
    laplace_alpha: float = 0.5,
) -> dict[str, float | int]:
    """
    CONTRACT: plugin_mi(xs, ys, ts=None, laplace_alpha=0.5) -> dict
        xs: list[Hashable]        — categorical observations of component output H_i
        ys: list[Hashable]        — categorical observations of ground-truth label Y
        ts: list[Hashable] | None — conditioning tool output T; None = unconditional MI
        laplace_alpha: float      — Dirichlet smoothing (0.5 = Jeffreys prior)
        returns: {
            mi: float             — I(X;Y) or I(X;Y|T) in nats
            h_x: float            — H(X) in nats
            h_y: float            — H(Y) in nats
            h_y_given_t: float    — H(Y|T) in nats (nan if ts is None)
            n: int                — number of observations
        }

    Formulas:
        Unconditional: I(X;Y) = H(X) + H(Y) - H(X,Y)
        Conditional:   I(X;Y|T) = H(Y|T) - H(Y|X,T)
                       H(Y|T) = sum_t p(t) * H(Y|T=t)
                       H(Y|X,T) = sum_t p(t) * H(Y|X,T=t)

    Raises ValueError if input lengths mismatch or if laplace_alpha < 0.
    """
    if laplace_alpha < 0:
        raise ValueError(f"laplace_alpha must be >= 0, got {laplace_alpha}")
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have equal length, got {len(xs)} vs {len(ys)}")
    if ts is not None and len(ts) != len(xs):
        raise ValueError(f"ts must have same length as xs, got {len(ts)} vs {len(xs)}")
    if not xs:
        log.warning("plugin_mi called with empty sequences; returning zeros")
        return {"mi": 0.0, "h_x": 0.0, "h_y": 0.0, "h_y_given_t": float("nan"), "n": 0}

    n = len(xs)
    h_x = _marginal_entropy(xs, laplace_alpha)
    h_y = _marginal_entropy(ys, laplace_alpha)

    if ts is None:
        # Unconditional: I(X;Y) = H(X) + H(Y) - H(X,Y)
        h_xy = joint_entropy(xs, ys, laplace_alpha)
        mi = max(0.0, h_x + h_y - h_xy)
        log.debug("Unconditional MI: H(X)=%.4f H(Y)=%.4f H(X,Y)=%.4f I(X;Y)=%.4f", h_x, h_y, h_xy, mi)
        return {"mi": mi, "h_x": h_x, "h_y": h_y, "h_y_given_t": float("nan"), "n": n}

    # Conditional: I(X;Y|T) = H(Y|T) - H(Y|X,T)
    # Group by t value
    t_counts = Counter(ts)
    t_vocab_size = len(t_counts)
    t_denom = n + laplace_alpha * t_vocab_size

    h_y_given_t = 0.0
    h_y_given_xt = 0.0

    # Build slices per t value
    slices: dict[Hashable, tuple[list[Hashable], list[Hashable]]] = {}
    for x, y, t in zip(xs, ys, ts):
        if t not in slices:
            slices[t] = ([], [])
        slices[t][0].append(x)
        slices[t][1].append(y)

    for t_val, (xs_t, ys_t) in slices.items():
        # Smoothed marginal probability p(T=t)
        p_t = (t_counts[t_val] + laplace_alpha) / t_denom
        # H(Y|T=t)
        # NOTE: vocab_size computed per T-slice (standard plugin estimator); global vocab would reduce bias for small slices but change estimator class
        h_y_t = _marginal_entropy(ys_t, laplace_alpha)
        # H(Y|X,T=t) = H(Y|X) within slice
        h_y_x_t = conditional_entropy(xs_t, ys_t, laplace_alpha)
        h_y_given_t += p_t * h_y_t
        h_y_given_xt += p_t * h_y_x_t

    mi = max(0.0, h_y_given_t - h_y_given_xt)
    log.debug(
        "Conditional MI: H(Y|T)=%.4f H(Y|X,T)=%.4f I(X;Y|T)=%.4f",
        h_y_given_t, h_y_given_xt, mi,
    )
    return {"mi": mi, "h_x": h_x, "h_y": h_y, "h_y_given_t": h_y_given_t, "n": n}


# ---------------------------------------------------------------------------
# CLI entry point (for quick inspection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    # Simple smoke test on synthetic data
    import random
    rng = random.Random(42)
    n = 200
    xs = [rng.choice(["a", "b", "c"]) for _ in range(n)]
    ys = [rng.choice(["attack", "benign"]) for _ in range(n)]
    ts = [rng.choice(["high", "low"]) for _ in range(n)]

    result_unconditional = plugin_mi(xs, ys)
    result_conditional = plugin_mi(xs, ys, ts)
    print("Unconditional MI:", json.dumps(result_unconditional, indent=2))
    print("Conditional MI:  ", json.dumps(result_conditional, indent=2))
    sys.exit(0)
