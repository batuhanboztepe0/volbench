"""CAViaR — Conditional Autoregressive Value at Risk by Regression Quantiles.

Engle & Manganelli (2004, JBES). Models the conditional ``alpha``-quantile of
returns *directly* by minimising the regression-quantile (tick) loss, instead of
routing through a conditional-variance model and a distributional assumption — the
canonical fix for the Dynamic-Quantile failures of normal/Student-t/FHS VaR.

Specifications (``q_t`` = conditional alpha-quantile, negative for a left tail;
``VaR_t = -q_t``):

* ``"SAV"`` — Symmetric Absolute Value::

      q_t = b0 + b1 q_{t-1} + b2 |r_{t-1}|

* ``"AS"`` — Asymmetric Slope (leverage)::

      q_t = b0 + b1 q_{t-1} + b2 max(r_{t-1}, 0) + b3 max(-r_{t-1}, 0)

* ``"REALIZED"`` — Realized-CAViaR: augments the recursion with a realized-volatility
  regressor available at ``t`` (here the project's LogHAR variance forecast)::

      q_t = b0 + b1 q_{t-1} + b2 sqrt(exog_t)

All three are *linear* in the lagged regressors, so each tick-loss evaluation runs
the first-order recursion with :func:`scipy.signal.lfilter` (C-speed). Parameters
are fit by multi-start Nelder-Mead (Engle-Manganelli's recipe for the non-convex
quantile objective).

No look-ahead: :func:`caviar_var_forecasts` walks forward, refitting on an
expanding window every ``refit_every`` origins and rolling the recursion forward
with fixed parameters in between.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

SPECS = ("SAV", "AS", "REALIZED")


def _design(returns: np.ndarray, exog: np.ndarray | None, spec: str) -> np.ndarray:
    """Per-step exogenous columns ``C`` (shape ``(T, m)``) with
    ``q_t = b0 + b1 q_{t-1} + C[t] @ b[2:]``. Lagged terms use ``r_{t-1}``;
    ``REALIZED`` uses the contemporaneous ``exog_t``. Row 0 is zero (``q_0`` seed)."""
    r = np.asarray(returns, dtype=float)
    t = r.size
    if spec == "SAV":
        c = np.zeros((t, 1))
        c[1:, 0] = np.abs(r[:-1])
    elif spec == "AS":
        c = np.zeros((t, 2))
        c[1:, 0] = np.maximum(r[:-1], 0.0)
        c[1:, 1] = np.maximum(-r[:-1], 0.0)
    elif spec == "REALIZED":
        if exog is None:
            raise ValueError("REALIZED spec needs exog (e.g. the variance forecast)")
        x = np.asarray(exog, dtype=float)
        if x.shape != r.shape:
            raise ValueError(f"exog must match returns length, got {x.shape} vs {r.shape}")
        c = np.zeros((t, 1))
        c[:, 0] = np.sqrt(np.maximum(x, 0.0))  # contemporaneous realized vol
    else:
        raise ValueError(f"unknown spec {spec!r}; choose from {SPECS}")
    return c


def _path(params: np.ndarray, design: np.ndarray, q0: float) -> np.ndarray:
    """Quantile path ``q_t = b0 + b1 q_{t-1} + design[t] @ b[2:]`` with ``q_0 = q0``.

    Linear first-order recursion ``y_t = inp_t + b1 y_{t-1}`` evaluated with
    :func:`scipy.signal.lfilter` (the ``inp_0 = q0`` seed gives ``y_0 = q0``)."""
    b0, b1 = float(params[0]), float(params[1])
    inp = b0 + design @ params[2:]
    inp = inp.copy()
    inp[0] = q0
    return lfilter([1.0], [1.0, -b1], inp)


def _tick_loss(params, returns, design, alpha, q0, start) -> float:
    """Regression-quantile (tick) loss over ``returns[start:]``; large penalty for a
    non-stationary persistence ``|b1| >= 1`` or a non-finite path."""
    if abs(params[1]) >= 0.999:
        return 1e12
    q = _path(params, design, q0)
    u = returns[start:] - q[start:]
    val = float(np.sum(u * (alpha - (u < 0.0))))
    return val if np.isfinite(val) else 1e12


def _fit(returns, design, alpha, q0, start, n_starts, rng) -> np.ndarray:
    """Multi-start Nelder-Mead minimisation of the tick loss."""
    k = 2 + design.shape[1]
    starts = [np.r_[q0 * 0.1, 0.9, np.full(k - 2, -0.05)]]  # sensible default
    for _ in range(max(0, n_starts - 1)):
        starts.append(np.r_[
            q0 * rng.uniform(0.0, 0.3),
            rng.uniform(0.5, 0.97),
            rng.uniform(-0.5, 0.0, size=k - 2),
        ])
    best, best_loss = starts[0], np.inf          # best among converged starts
    best_any, best_any_loss = starts[0], np.inf  # fallback if none converge
    for x0 in starts:
        res = minimize(
            _tick_loss, x0, args=(returns, design, alpha, q0, start),
            method="Nelder-Mead",
            options={"maxiter": 1500, "xatol": 1e-7, "fatol": 1e-9},
        )
        if res.fun < best_any_loss:
            best_any_loss, best_any = float(res.fun), res.x
        if res.success and res.fun < best_loss:
            best_loss, best = float(res.fun), res.x
    # Prefer a converged optimum; fall back to the best objective if none converged.
    return best if np.isfinite(best_loss) else best_any


def caviar_var_forecasts(
    returns: np.ndarray,
    alpha: float = 0.05,
    spec: str = "AS",
    *,
    exog: np.ndarray | None = None,
    min_train: int = 500,
    refit_every: int = 125,
    init_window: int = 250,
    n_starts: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Walk-forward one-step CAViaR VaR forecasts (positive loss thresholds).

    Returns an array shaped like ``returns`` with ``np.nan`` for the leading
    ``min_train`` origins and ``VaR_t = -q_t`` thereafter. The quantile seed
    ``q_0`` is the empirical ``alpha``-quantile of ``returns[:init_window]``;
    parameters are refit on the expanding window ``returns[:t]`` every
    ``refit_every`` origins and the recursion rolled forward in between.
    """
    if spec not in SPECS:
        raise ValueError(f"spec={spec!r} not in {SPECS}")
    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    if n <= min_train + 10:
        raise ValueError(f"need more than min_train+10 obs, got n={n}, min_train={min_train}")
    if init_window >= min_train:
        raise ValueError("init_window must be < min_train")

    design = _design(r, exog, spec)
    rng = np.random.default_rng(seed)
    q0 = float(np.quantile(r[:init_window], alpha))
    var = np.full(n, np.nan)
    params: np.ndarray | None = None

    for t in range(min_train, n):
        if params is None or (t - min_train) % refit_every == 0:
            params = _fit(r[:t], design[:t], alpha, q0, init_window, n_starts, rng)
        q_t = _path(params, design[: t + 1], q0)[t]  # one-step-ahead quantile for r_t
        var[t] = -float(q_t)
    return var
