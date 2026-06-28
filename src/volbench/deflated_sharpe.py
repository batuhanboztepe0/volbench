"""Probabilistic and Deflated Sharpe ratios (Bailey & López de Prado, 2014).

A raw annualised Sharpe ignores three things that matter for an honest edge claim:

* **non-normality**: fat tails / negative skew make a given Sharpe less impressive,
* **sample length**: a high Sharpe over a short sample is weak evidence,
* **selection**: if the reported strategy was picked as the *best of N* tried, its
  Sharpe is inflated by multiple testing.

The Probabilistic Sharpe Ratio (PSR) handles the first two; the Deflated Sharpe
Ratio (DSR) adds the third. Both return a probability that the *true* (per-period)
Sharpe exceeds a benchmark, so a value near 1 is strong evidence and near 0.5 is
none. All Sharpes here are **per-period** (mean/std of the return series), not
annualised; pass ``n_eff`` to down-weight overlapping returns.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

# Euler–Mascheroni constant, used in the expected-maximum-Sharpe approximation.
_EULER_MASCHERONI: float = 0.5772156649015329


def _moments(returns: ArrayLike) -> tuple[float, float, float, int] | None:
    """Return (per-period Sharpe, skewness, non-excess kurtosis, n) or None."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return None
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return None
    mu = float(np.mean(r))
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # non-excess (Gaussian == 3)
    return mu / sd, skew, kurt, n


def per_period_sharpe(returns: ArrayLike) -> float:
    """Per-period (non-annualised) Sharpe = mean / std(ddof=1); NaN if degenerate."""
    m = _moments(returns)
    return float("nan") if m is None else m[0]


def probabilistic_sharpe_ratio(
    returns: ArrayLike,
    benchmark_sr: float = 0.0,
    n_eff: float | None = None,
) -> float:
    """P(true per-period Sharpe > ``benchmark_sr``), adjusting for skew/kurtosis/length.

    Parameters
    ----------
    returns : np.ndarray
        Per-period strategy return (or P&L) series.
    benchmark_sr : float, default 0.0
        Per-period Sharpe to beat. 0.0 asks "is this better than nothing?"; pass
        :func:`expected_max_sharpe` to deflate for selection (see
        :func:`deflated_sharpe_ratio`).
    n_eff : float, optional
        Effective sample size. Defaults to the number of observations; for
        overlapping ``h``-period returns sampled every period use ``n / h``, since
        the overlap shrinks the independent information.
    """
    m = _moments(returns)
    if m is None:
        return float("nan")
    sr, skew, kurt, n = m
    t = float(n if n_eff is None else n_eff)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if t <= 1.0 or denom <= 0.0:
        return float("nan")
    z = (sr - benchmark_sr) * np.sqrt(t - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(trial_sharpes: ArrayLike) -> float:
    """Expected maximum per-period Sharpe under the null across ``N`` i.i.d. trials.

    This is the bar a strategy must clear *because it was selected as the best of
    ``N``*. With one trial there is no selection, so it is 0.
    """
    s = np.asarray(trial_sharpes, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n_trials = s.size
    if n_trials < 2:
        return 0.0
    var_sr = float(np.var(s, ddof=1))
    if var_sr <= 0.0:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    return float(np.sqrt(var_sr) * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: ArrayLike,
    trial_sharpes: ArrayLike,
    n_eff: float | None = None,
) -> float:
    """Deflated Sharpe: PSR against the expected best-of-``N`` null Sharpe.

    ``trial_sharpes`` are the per-period Sharpes of *all* strategy configurations
    tried (including this one). The more trials, and the more dispersed their
    Sharpes, the higher the bar this strategy must clear to be judged genuinely
    positive. Reduces to ``probabilistic_sharpe_ratio(returns, 0.0)`` when only one
    trial is supplied.
    """
    sr_star = expected_max_sharpe(trial_sharpes)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star, n_eff=n_eff)
