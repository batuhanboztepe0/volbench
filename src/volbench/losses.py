"""Forecast-evaluation loss functions for a noisy volatility proxy.

Realized variance is an imperfect proxy for the latent integrated variance, so
not every loss yields a consistent ranking of forecasts. Patton (2011) shows
that only a sub-class of losses is *robust*: the ranking they induce is
unaffected (in expectation) by the noise in the proxy. The two robust losses
used here are :func:`qlike` and :func:`mse_variance`. The non-robust
:func:`mse_volatility` (squared error on the volatility scale) is provided for
*reference reporting only* and must never drive a ranking or a Model Confidence
Set (see ``ROADMAP.md`` invariant 3).

Every loss function takes ``(realized, forecast)`` arrays on the **variance**
scale and returns a *per-observation* loss array, so the output feeds directly
into the Diebold-Mariano and Model Confidence Set machinery in
:mod:`volbench.evaluation`.

References
----------
- Patton (2011), "Volatility forecast comparison using imperfect volatility
  proxies", *Journal of Econometrics*.
- Mincer & Zarnowitz (1969), forecast-efficiency regression.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Forecasts and proxies are strictly positive variances; this floor keeps the
# logarithm finite if a level-space model emits a non-positive forecast (which
# is then heavily — and correctly — penalised by QLIKE).
_POS_FLOOR: float = 1e-300

# Ranking is permitted only with these proxy-robust losses (Patton 2011).
RANKING_LOSSES: tuple[str, ...] = ("QLIKE", "MSE-var")


def _validate(realized: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to aligned 1-D float arrays.

    Parameters
    ----------
    realized, forecast : np.ndarray
        Realized-variance proxy and variance forecast, equal length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The two arrays as contiguous float64.

    Raises
    ------
    ValueError
        If the lengths differ or an array is empty.
    """
    r = np.asarray(realized, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if r.shape != f.shape:
        raise ValueError(f"realized and forecast must match in length, got {r.shape} and {f.shape}")
    if r.size == 0:
        raise ValueError("loss inputs are empty")
    return r, f


def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss, proxy-robust (Patton 2011).

    ``L = sigma2 / h - log(sigma2 / h) - 1`` where ``sigma2`` is the realized
    proxy and ``h`` the variance forecast. The loss is non-negative and zero
    iff ``h == sigma2``; it penalises under-prediction more than
    over-prediction, matching the asymmetry of variance.

    Parameters
    ----------
    realized, forecast : np.ndarray
        Variance proxy and variance forecast, equal length.

    Returns
    -------
    np.ndarray
        Per-observation QLIKE loss.
    """
    r, f = _validate(realized, forecast)
    f = np.maximum(f, _POS_FLOOR)
    ratio = np.maximum(r, _POS_FLOOR) / f
    return ratio - np.log(ratio) - 1.0


def mse_variance(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Squared error on the **variance** scale, ``(sigma2 - h)^2``.

    Proxy-robust (Patton 2011): it ranks forecasts consistently even though the
    target is the noisy realized variance rather than the latent variance.
    """
    r, f = _validate(realized, forecast)
    return (r - f) ** 2


def mse_volatility(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Squared error on the **volatility** scale, ``(sqrt(sigma2) - sqrt(h))^2``.

    *Reference only.* This loss is **not** robust to proxy noise and must never
    be used to rank models or form an MCS; it is reported because it is the
    quantity practitioners often eyeball.
    """
    r, f = _validate(realized, forecast)
    return (np.sqrt(np.maximum(r, 0.0)) - np.sqrt(np.maximum(f, 0.0))) ** 2


def mean_loss(loss: np.ndarray) -> float:
    """Mean of a per-observation loss array (ignoring non-finite entries)."""
    a = np.asarray(loss, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def rmse_volatility(realized: np.ndarray, forecast: np.ndarray) -> float:
    """Root-mean-squared error on the volatility scale (a scalar reducer).

    Convenience wrapper, ``sqrt(mean(mse_volatility(...)))``, for reference
    tables. Not a ranking loss.
    """
    return float(np.sqrt(mean_loss(mse_volatility(realized, forecast))))


# Mapping of loss name -> per-observation loss function. Only the losses in
# :data:`RANKING_LOSSES` may drive a ranking; ``MSE-vol`` is reference-only.
LOSS_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "QLIKE": qlike,
    "MSE-var": mse_variance,
    "MSE-vol": mse_volatility,
}


def mincer_zarnowitz(realized: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Mincer-Zarnowitz forecast-efficiency regression on the variance scale.

    Fits ``realized = alpha + beta * forecast + e`` by OLS. A well-calibrated
    forecast has ``alpha = 0`` and ``beta = 1``. Returns the slope/intercept,
    the regression ``r2``, individual t-test p-values for ``H0: alpha = 0`` and
    ``H0: beta = 1``, and the joint F-test p-value for ``H0: (alpha, beta) =
    (0, 1)`` (the standard calibration test).

    Parameters
    ----------
    realized, forecast : np.ndarray
        Realized proxy and variance forecast, equal length.

    Returns
    -------
    dict[str, float]
        Keys: ``alpha``, ``beta``, ``r2``, ``p_alpha``, ``p_beta``,
        ``p_joint``, ``n``.
    """
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist

    y, x = _validate(realized, forecast)
    n = y.size
    X = np.column_stack([np.ones(n), x])
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    beta_hat = xtx_inv @ (X.T @ y)
    resid = y - X @ beta_hat
    rss = float(resid @ resid)
    dof = n - 2
    sigma2 = rss / dof if dof > 0 else float("nan")
    cov = sigma2 * xtx_inv

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    se_alpha = float(np.sqrt(cov[0, 0]))
    se_beta = float(np.sqrt(cov[1, 1]))
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")
    t_beta = (beta - 1.0) / se_beta if se_beta > 0 else float("nan")
    p_alpha = float(2.0 * t_dist.sf(abs(t_alpha), df=dof)) if np.isfinite(t_alpha) else float("nan")
    p_beta = float(2.0 * t_dist.sf(abs(t_beta), df=dof)) if np.isfinite(t_beta) else float("nan")

    # Joint Wald/F test of (alpha, beta) = (0, 1).
    diff = beta_hat - np.array([0.0, 1.0])
    try:
        wald = float(diff @ np.linalg.inv(cov) @ diff)
        f_stat = wald / 2.0
        p_joint = float(f_dist.sf(f_stat, 2, dof)) if dof > 0 else float("nan")
    except np.linalg.LinAlgError:
        p_joint = float("nan")

    return {
        "alpha": alpha,
        "beta": beta,
        "r2": float(r2),
        "p_alpha": p_alpha,
        "p_beta": p_beta,
        "p_joint": p_joint,
        "n": float(n),
    }
