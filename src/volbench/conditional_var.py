"""Return-based conditional-variance engines for Value-at-Risk (risk layer).

These produce *reactive* one-step conditional-variance forecasts from daily
returns — a RiskMetrics EWMA and a (GJR-)GARCH(1,1) — for use as the variance
input to :func:`volbench.economic.var_backtest`. They exist because the realized-
variance point forecast (log-HAR), while accurate on average, is smooth and lags
during turbulence, so a VaR built on it *under-covers* and its violations cluster.

This is a **risk** tool, not a forecast-accuracy track: never rank these by QLIKE
against the realized-variance benchmark (``ROADMAP.md`` invariant 4 — Track 2 and
Track 1 stay siloed). The only quantities compared here are VaR coverage and the
Engle–Manganelli dynamic-quantile test.

All forecasts are look-ahead-free: the variance for day ``t`` uses only returns up
to ``t - 1``. Pass returns in **percent** units (``arch`` is numerically happier
there); VaR coverage is scale-invariant, so the choice does not affect results.

(Track 2's ``scripts/run_garch.py`` carries parallel private copies of these
recursions for the GARCH *forecast-accuracy* benchmark; the public versions here
are for the VaR/risk application.)
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_MIN_TRAIN: int = 1000
RISKMETRICS_LAMBDA: float = 0.94


def ewma_variance_forecast(
    returns: ArrayLike,
    min_train: int = DEFAULT_MIN_TRAIN,
    lam: float = RISKMETRICS_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """RiskMetrics EWMA one-step conditional-variance forecasts.

    ``var[t] = lam * var[t-1] + (1 - lam) * return[t-1]**2`` (no parameters to fit).

    Returns
    -------
    (forecast_var, origins)
        ``forecast_var[k]`` is the variance forecast for the return at
        ``origins[k] + 1``, using only returns up to ``origins[k]``.
    """
    y = np.asarray(returns, dtype=float).ravel()
    n = y.size
    if n < min_train + 2:
        raise ValueError(f"need at least min_train+2={min_train + 2} returns, got {n}")
    var = np.empty(n)
    var[0] = float(y[0] ** 2)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * y[t - 1] ** 2
    origins = np.arange(min_train, n - 1, dtype=int)
    return var[origins + 1], origins


def garch_variance_forecast(
    returns: ArrayLike,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every: int = 63,
    o: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward (GJR-)GARCH(1,1) one-step conditional-variance forecasts.

    ``o = 0`` gives a symmetric GARCH; ``o = 1`` gives GJR-GARCH (an asymmetry term
    so a negative return raises tomorrow's variance more than a positive one — the
    equity leverage effect). Parameters are re-estimated on the expanding window
    every ``refit_every`` origins and then filtered forward with those fixed
    parameters, which is cheap and look-ahead-free (each forecast uses only returns
    up to its origin). Requires the ``arch`` package.

    Returns
    -------
    (forecast_var, origins)
        Same convention as :func:`ewma_variance_forecast`.
    """
    from arch import arch_model

    y = np.asarray(returns, dtype=float).ravel()
    n = y.size
    if n < min_train + 2:
        raise ValueError(f"need at least min_train+2={min_train + 2} returns, got {n}")
    origins = np.arange(min_train, n - 1, dtype=int)
    forecasts = np.full(origins.size, np.nan)
    am_full = arch_model(y, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="normal")

    cond_var: np.ndarray | None = None
    for k, t in enumerate(origins):
        if k % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = arch_model(
                    y[: t + 1], mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="normal"
                ).fit(disp="off").params
            cond_var = am_full.fix(params).conditional_volatility ** 2
        assert cond_var is not None
        forecasts[k] = float(cond_var[t + 1])
    return forecasts, origins
