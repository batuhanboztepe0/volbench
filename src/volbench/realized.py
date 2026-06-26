"""Non-parametric realized measures from intraday returns.

This module implements the standard estimators of (co)variation used in the
high-frequency volatility literature. Every estimator takes a 1-D array of
*intraday log-returns for a single day* and returns a scalar.

All estimators are consistent (under their respective assumptions) for either
the integrated variance ``IV = \\int_0^1 v_s ds`` or the total quadratic
variation ``QV = IV + JV`` where ``JV`` is the jump variation. The docstring of
each function states which quantity it targets and under which assumptions.

References
----------
- Andersen, Bollerslev, Diebold, Labys (2001), realized variance.
- Barndorff-Nielsen, Shephard (2004, 2006), bipower variation and jump tests.
- Barndorff-Nielsen, Hansen, Lunde, Shephard (2008), realized kernels.
- Barndorff-Nielsen, Kinnebrock, Shephard (2010), realized semivariance.
- Andersen, Dobrev, Schaumburg (2012), median realized variance.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gamma

# --- Statistical constants for jump-robust estimators -----------------------
# E|Z| for Z ~ N(0, 1). Bipower variation rescales pairwise abs-products by 1/mu1^2.
MU1: float = np.sqrt(2.0 / np.pi)
BV_SCALE: float = 1.0 / (MU1 ** 2)  # == pi / 2

# Median realized variance scaling (Andersen, Dobrev, Schaumburg 2012, eq. 2.4).
MEDRV_SCALE: float = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)

# Tripower quarticity uses E|Z|^{4/3}.
MU_43: float = (2.0 ** (2.0 / 3.0)) * _gamma(7.0 / 6.0) / _gamma(0.5)
TP_SCALE: float = 1.0 / (MU_43 ** 3)

# BNS jump-test asymptotic constant theta = pi^2/4 + pi - 5 (Huang-Tauchen 2005).
BNS_THETA: float = (np.pi ** 2) / 4.0 + np.pi - 5.0

# Default realized-kernel bandwidth constant in H = ceil(C_RK * n^{3/5}).
C_RK_DEFAULT: float = 1.0

# Minimum observations required for the higher-order (lagged) estimators.
_MIN_OBS_PAIRWISE: int = 2
_MIN_OBS_TRIPLE: int = 3


def _as_clean_array(intraday_returns: np.ndarray) -> np.ndarray:
    """Validate and coerce an intraday-return input to a 1-D float array.

    Parameters
    ----------
    intraday_returns
        Intraday log-returns for a single day.

    Returns
    -------
    np.ndarray
        Finite returns as a contiguous 1-D float64 array (non-finite dropped).

    Raises
    ------
    ValueError
        If the input is empty or not 1-D after squeezing.
    """
    arr = np.asarray(intraday_returns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("intraday_returns is empty; need at least one return.")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("intraday_returns contains no finite values.")
    return arr


def realized_variance(intraday_returns: np.ndarray) -> float:
    """Realized variance ``RV = sum r_i^2``.

    Consistent for the total quadratic variation ``IV + JV`` as the sampling
    frequency increases, in the *absence* of microstructure noise.
    """
    r = _as_clean_array(intraday_returns)
    return float(np.sum(r ** 2))


def realized_semivariance(intraday_returns: np.ndarray) -> tuple[float, float]:
    """Realized semivariances ``(RSV_minus, RSV_plus)``.

    ``RSV_plus = sum r_i^2 1{r_i > 0}`` and ``RSV_minus = sum r_i^2 1{r_i < 0}``.
    By construction ``RSV_plus + RSV_minus == RV`` (zero returns contribute 0).
    The signed components isolate downside vs. upside variation (leverage).
    """
    r = _as_clean_array(intraday_returns)
    sq = r ** 2
    rsv_plus = float(np.sum(sq[r > 0.0]))
    rsv_minus = float(np.sum(sq[r < 0.0]))
    return rsv_minus, rsv_plus


def bipower_variation(intraday_returns: np.ndarray, correct_small_sample: bool = True) -> float:
    """Bipower variation ``BV``.

    ``BV = (pi/2) * sum_{i=2}^M |r_{i-1}| |r_i|`` (optionally times ``M/(M-1)``).
    Consistent for the *integrated variance only* (``IV``); it is robust to
    finite-activity jumps because a jump contaminates only one product term.
    Comparing ``RV`` and ``BV`` is the basis of jump detection.
    """
    r = _as_clean_array(intraday_returns)
    if r.size < _MIN_OBS_PAIRWISE:
        return np.nan
    abs_prod = np.abs(r[1:]) * np.abs(r[:-1])
    bv = BV_SCALE * float(np.sum(abs_prod))
    if correct_small_sample:
        m = r.size
        bv *= m / (m - 1.0)
    return bv


def median_rv(intraday_returns: np.ndarray, correct_small_sample: bool = True) -> float:
    """Median realized variance ``medRV`` (Andersen, Dobrev, Schaumburg 2012).

    Uses rolling medians of three consecutive absolute returns; consistent for
    ``IV`` and more robust to jumps and zero-returns than bipower variation.
    """
    r = _as_clean_array(intraday_returns)
    if r.size < _MIN_OBS_TRIPLE:
        return np.nan
    a = np.abs(r)
    triplets = np.vstack([a[:-2], a[1:-1], a[2:]])
    med = np.median(triplets, axis=0)
    medrv = MEDRV_SCALE * float(np.sum(med ** 2))
    if correct_small_sample:
        m = r.size
        medrv *= m / (m - 2.0)
    return medrv


def realized_quarticity(intraday_returns: np.ndarray) -> float:
    """Realized quarticity ``RQ = (M/3) * sum r_i^4``.

    Consistent for the integrated quarticity ``IQ = \\int_0^1 v_s^2 ds``. It
    enters the standard errors of ``RV`` and the HARQ forecasting model.
    """
    r = _as_clean_array(intraday_returns)
    m = r.size
    return float((m / 3.0) * np.sum(r ** 4))


def tripower_quarticity(intraday_returns: np.ndarray) -> float:
    """Tripower quarticity ``TQ`` (jump-robust estimator of integrated quarticity).

    ``TQ = M * mu_{4/3}^{-3} * sum_{i=3}^M |r_{i-2}|^{4/3}|r_{i-1}|^{4/3}|r_i|^{4/3}``.
    Used inside the BNS jump test to studentise ``RV - BV``.
    """
    r = _as_clean_array(intraday_returns)
    if r.size < _MIN_OBS_TRIPLE:
        return np.nan
    p = np.abs(r) ** (4.0 / 3.0)
    triple = p[2:] * p[1:-1] * p[:-2]
    m = r.size
    return float(m * TP_SCALE * np.sum(triple))


def parzen_kernel(x: np.ndarray | float) -> np.ndarray | float:
    """Parzen kernel weights ``k(x)`` on ``[0, 1]`` (0 outside).

    ``k(x) = 1 - 6x^2 + 6x^3`` for ``0 <= x <= 1/2``, ``2(1-x)^3`` for
    ``1/2 < x <= 1``, and ``0`` otherwise.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    lo = (x >= 0.0) & (x <= 0.5)
    hi = (x > 0.5) & (x <= 1.0)
    out[lo] = 1.0 - 6.0 * x[lo] ** 2 + 6.0 * x[lo] ** 3
    out[hi] = 2.0 * (1.0 - x[hi]) ** 3
    return out if out.ndim else float(out)


def realized_kernel_parzen(intraday_returns: np.ndarray, bandwidth: int | None = None,
                           c_bandwidth: float = C_RK_DEFAULT) -> float:
    """Flat-top Parzen realized kernel (Barndorff-Nielsen et al. 2008).

    ``RK = gamma_0 + 2 * sum_{h=1}^H k((h-1)/H) * gamma_h`` where ``gamma_h`` is
    the h-th return autocovariance and ``k`` is the Parzen kernel. Unlike ``RV``
    it stays consistent for ``IV`` under iid microstructure noise (the noise
    induces negative ``gamma_1`` which the kernel undoes).

    Parameters
    ----------
    bandwidth
        Number of autocovariance lags ``H``. If ``None`` it is set to
        ``ceil(c_bandwidth * n^{3/5})``.
    """
    r = _as_clean_array(intraday_returns)
    n = r.size
    if bandwidth is None:
        bandwidth = max(1, int(np.ceil(c_bandwidth * n ** (3.0 / 5.0))))
    bandwidth = min(bandwidth, n - 1)
    gamma0 = float(np.sum(r ** 2))
    if bandwidth < 1:
        return gamma0
    rk = gamma0
    for h in range(1, bandwidth + 1):
        gamma_h = float(np.sum(r[h:] * r[:n - h]))
        weight = float(parzen_kernel((h - 1) / bandwidth))
        rk += 2.0 * weight * gamma_h
    # Floor at zero: a finite-sample kernel estimate can go slightly negative under
    # strong negative first-order autocovariance with a tiny bandwidth, and a
    # variance cannot be negative.
    return max(rk, 0.0)


def subsampled_rv(intraday_returns: np.ndarray, n_grids: int = 5) -> float:
    """Subsampled realized variance averaged over ``n_grids`` offset grids.

    Averaging RV computed on sparse, offset subgrids reduces the variance of the
    estimator while keeping (most of) the noise-robustness benefit of sparse
    sampling. Returns the average RV across grids, rescaled to a per-day total.
    """
    r = _as_clean_array(intraday_returns)
    n = r.size
    if n_grids < 1:
        raise ValueError("n_grids must be >= 1.")
    n_grids = min(n_grids, n)
    estimates = []
    for offset in range(n_grids):
        # Aggregate consecutive returns into blocks of size n_grids starting at offset.
        idx = np.arange(offset, n)
        blocks = idx[: (idx.size // n_grids) * n_grids].reshape(-1, n_grids)
        if blocks.size == 0:
            continue
        coarse_returns = r[blocks].sum(axis=1)
        estimates.append(np.sum(coarse_returns ** 2))
    if not estimates:
        return realized_variance(r)
    # Each offset grid aggregates *consecutive* returns into blocks that tile the
    # whole series, so its sum of coarse squared returns already estimates the full
    # daily QV; the average over offsets is the subsampled estimator. (No n_grids
    # rescaling: that would be correct only for sparse skip-K subsampling, where
    # each grid spans ~1/n_grids of the observations — not for dense blocking.)
    return float(np.mean(estimates))


def bns_jump_test(intraday_returns: np.ndarray) -> dict[str, float]:
    """Barndorff-Nielsen & Shephard jump test (ratio statistic, Huang-Tauchen).

    Returns a dict with ``rv``, ``bv``, ``jump_variation = max(RV - BV, 0)``,
    the relative jump ``rj = (RV - BV) / RV``, the standardised statistic ``z``
    (asymptotically ``N(0, 1)`` under the null of no jumps), and the one-sided
    p-value ``p_value``. Large positive ``z`` (small p-value) signals a jump.
    """
    from scipy.stats import norm

    r = _as_clean_array(intraday_returns)
    m = r.size
    rv = realized_variance(r)
    bv = bipower_variation(r)
    if not np.isfinite(bv) or rv <= 0.0:
        return {"rv": rv, "bv": bv, "jump_variation": np.nan,
                "rj": np.nan, "z": np.nan, "p_value": np.nan}
    tq = tripower_quarticity(r)
    rj = (rv - bv) / rv
    denom = BNS_THETA * (1.0 / m) * max(1.0, tq / (bv ** 2))
    z = rj / np.sqrt(denom) if denom > 0.0 else np.nan
    p_value = float(1.0 - norm.cdf(z)) if np.isfinite(z) else np.nan
    return {
        "rv": rv,
        "bv": bv,
        "jump_variation": float(max(rv - bv, 0.0)),
        "rj": float(rj),
        "z": float(z) if np.isfinite(z) else np.nan,
        "p_value": p_value,
    }


def all_measures(intraday_returns: np.ndarray) -> dict[str, float]:
    """Compute the full panel of realized measures for one day.

    Convenience wrapper returning ``rv``, ``bv``, ``medrv``, ``rk``, ``rsv_minus``,
    ``rsv_plus``, ``rq``, and ``jump_variation`` in a single dict.
    """
    rsv_minus, rsv_plus = realized_semivariance(intraday_returns)
    rv = realized_variance(intraday_returns)
    bv = bipower_variation(intraday_returns)
    return {
        "rv": rv,
        "bv": bv,
        "medrv": median_rv(intraday_returns),
        "rk": realized_kernel_parzen(intraday_returns),
        "rsv_minus": rsv_minus,
        "rsv_plus": rsv_plus,
        "rq": realized_quarticity(intraday_returns),
        "jump_variation": float(max(rv - bv, 0.0)),
    }
