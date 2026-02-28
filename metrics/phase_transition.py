"""
Phase transition detection metrics.

Given a performance curve M(alpha) — a list of (alpha, score) pairs —
this module:
  1. Fits a logistic sigmoid to detect a sharp transition.
  2. Extracts the critical threshold alpha* and sharpness beta.
  3. Computes the Area Under the Curve (AUC).

The sigmoid model is:
    M(alpha) = L / (1 + exp(-beta * (alpha - alpha_star))) + offset

where:
  - alpha_star : inflection point (critical threshold)
  - beta       : steepness (large beta = sharp phase transition)
  - L          : scale (≈ 1 for normalised metrics)
  - offset     : floor value (≈ 0 for normalised metrics)

scipy.optimize is imported lazily so the module can be imported without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SigmoidFit:
    """Result of fitting a sigmoid to a performance curve."""
    alpha_star: float          # critical threshold (inflection point)
    beta: float                # steepness / sharpness
    L: float                   # scale parameter
    offset: float              # floor parameter
    r_squared: float           # goodness of fit [0, 1]
    converged: bool            # whether the optimiser converged
    error: Optional[str] = None

    @property
    def is_sharp(self) -> bool:
        """Heuristic: beta > 10 suggests a sharp phase transition."""
        return self.beta > 10.0


# ---------------------------------------------------------------------------
# Sigmoid function
# ---------------------------------------------------------------------------

def _sigmoid(alpha: float, alpha_star: float, beta: float, L: float, offset: float) -> float:
    """Logistic sigmoid evaluated at a single alpha value."""
    try:
        return L / (1.0 + math.exp(-beta * (alpha - alpha_star))) + offset
    except OverflowError:
        return offset if beta * (alpha - alpha_star) < 0 else L + offset


def _sigmoid_vec(alphas: List[float], alpha_star: float, beta: float, L: float, offset: float) -> List[float]:
    return [_sigmoid(a, alpha_star, beta, L, offset) for a in alphas]


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fit_sigmoid(
    alphas: List[float],
    scores: List[float],
) -> SigmoidFit:
    """
    Fit a logistic sigmoid to the (alpha, score) performance curve.

    Parameters
    ----------
    alphas : list of coherence ratio values (x-axis), e.g. [0.0, 0.05, ..., 1.0]
    scores : list of metric values (y-axis), same length as alphas

    Returns
    -------
    SigmoidFit dataclass with fitted parameters.
    """
    if len(alphas) != len(scores):
        raise ValueError("alphas and scores must have the same length")
    if len(alphas) < 4:
        return SigmoidFit(
            alpha_star=0.5, beta=1.0, L=1.0, offset=0.0,
            r_squared=0.0, converged=False,
            error="Need at least 4 data points to fit sigmoid"
        )

    try:
        from scipy.optimize import curve_fit  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        def model(x, alpha_star, beta, L, offset):
            return L / (1.0 + np.exp(-beta * (x - alpha_star))) + offset

        x = np.array(alphas)
        y = np.array(scores)

        # Initial guess: inflection at midpoint, moderate steepness
        p0 = [0.5, 5.0, max(y) - min(y), min(y)]
        bounds = (
            [0.0, 0.0, 0.0, -0.5],   # lower bounds
            [1.0, 200.0, 2.0, 1.0],  # upper bounds
        )

        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        alpha_star, beta, L, offset = popt

        # R² goodness of fit
        y_pred = model(x, *popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return SigmoidFit(
            alpha_star=float(alpha_star),
            beta=float(beta),
            L=float(L),
            offset=float(offset),
            r_squared=r_squared,
            converged=True,
        )

    except ImportError:
        # scipy not available: use a simple heuristic
        return _heuristic_fit(alphas, scores)
    except Exception as exc:  # noqa: BLE001
        return SigmoidFit(
            alpha_star=0.5, beta=1.0, L=1.0, offset=0.0,
            r_squared=0.0, converged=False,
            error=str(exc),
        )


def _heuristic_fit(alphas: List[float], scores: List[float]) -> SigmoidFit:
    """
    Fallback when scipy is unavailable.
    Estimates alpha* as the alpha where score crosses 50% of its range.
    """
    min_s, max_s = min(scores), max(scores)
    mid = (min_s + max_s) / 2.0
    alpha_star = 0.5
    for a, s in zip(alphas, scores):
        if s >= mid:
            alpha_star = a
            break
    return SigmoidFit(
        alpha_star=alpha_star,
        beta=5.0,   # assumed moderate sharpness
        L=max_s - min_s,
        offset=min_s,
        r_squared=0.0,
        converged=False,
        error="scipy not available; used heuristic estimate",
    )


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def compute_critical_threshold(
    alphas: List[float],
    scores: List[float],
    threshold_fraction: float = 0.5,
) -> Optional[float]:
    """
    Find the alpha at which the score drops below `threshold_fraction` of
    its maximum value (measured from the alpha=1.0 baseline).

    Returns None if the score never drops below the threshold.
    """
    if not scores:
        return None
    baseline = scores[-1]  # assume alphas are sorted ascending; last = alpha=1.0
    cutoff = baseline * threshold_fraction

    # Walk from high alpha to low alpha
    for a, s in zip(reversed(alphas), reversed(scores)):
        if s <= cutoff:
            return a
    return None


def compute_auc(alphas: List[float], scores: List[float]) -> float:
    """
    Compute the Area Under the performance Curve using the trapezoidal rule.
    Normalised to [0, 1] by dividing by the alpha range.
    """
    if len(alphas) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(alphas)):
        da = alphas[i] - alphas[i - 1]
        area += da * (scores[i] + scores[i - 1]) / 2.0
    alpha_range = alphas[-1] - alphas[0]
    return area / alpha_range if alpha_range > 0 else 0.0


def compute_transition_sharpness(
    alphas: List[float],
    scores: List[float],
) -> float:
    """
    Estimate transition sharpness as the maximum absolute slope |dM/dalpha|
    across the curve (finite differences).
    """
    if len(alphas) < 2:
        return 0.0
    max_slope = 0.0
    for i in range(1, len(alphas)):
        da = alphas[i] - alphas[i - 1]
        if da == 0:
            continue
        slope = abs((scores[i] - scores[i - 1]) / da)
        if slope > max_slope:
            max_slope = slope
    return max_slope


# ---------------------------------------------------------------------------
# Convenience: build a performance curve from raw results
# ---------------------------------------------------------------------------

def build_performance_curve(
    results_by_alpha: dict,   # {alpha: List[ProbingResult]}
    metric_fn,                # callable(List[ProbingResult]) -> float
) -> Tuple[List[float], List[float]]:
    """
    Build sorted (alphas, scores) lists from a dict of results keyed by alpha.

    Parameters
    ----------
    results_by_alpha : dict mapping alpha → list of ProbingResult
    metric_fn        : function that takes a list of ProbingResult and returns a float

    Returns
    -------
    (alphas, scores) both sorted by alpha ascending
    """
    pairs = [(alpha, metric_fn(results)) for alpha, results in results_by_alpha.items()]
    pairs.sort(key=lambda x: x[0])
    alphas = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]
    return alphas, scores

# Made with Bob
