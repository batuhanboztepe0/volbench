"""Statistical tests for comparing competing forecasts.

Two tools are provided:

* :func:`diebold_mariano` - the Diebold-Mariano (1995) test of equal
  predictive accuracy for a *pair* of forecasts, with a Newey-West HAC
  variance and the Harvey-Leybourne-Newbold (1997) small-sample correction.

* :func:`model_confidence_set` - the Model Confidence Set of Hansen, Lunde and
  Nason (2011, *Econometrica*), which identifies, at a given confidence level,
  the subset of models that contains the best model(s) with prescribed
  probability. It uses the range statistic with a moving-block bootstrap.

Both operate on per-observation loss series produced by :mod:`volbench.losses`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_BOOTSTRAP_REPS: int = 2000
_DEFAULT_BLOCK_LENGTH: int = 10
_MIN_BOOTSTRAP_VARIANCE: float = 1e-300  # guards divide-by-zero in standardisation


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------
def _newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance of a (demeaned) series.

    Parameters
    ----------
    d : np.ndarray
        Loss-differential series.
    lag : int
        Truncation lag (number of autocovariances, Bartlett-weighted).

    Returns
    -------
    float
        Estimated long-run variance (sum of weighted autocovariances).
    """
    n = d.size
    d0 = d - d.mean()
    gamma0 = float(d0 @ d0) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float(d0[k:] @ d0[:-k]) / n
        lrv += 2.0 * w * gamma_k
    return lrv


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    lag: int | None = None,
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests the null of equal expected loss between forecast A and forecast B.
    The differential is ``d_t = loss_a_t - loss_b_t``; a *negative* mean
    differential favours A (A has lower loss). The statistic uses a Newey-West
    HAC variance with Bartlett weights and applies the Harvey-Leybourne-Newbold
    (1997) finite-sample correction, comparing the corrected statistic to a
    Student-t distribution with ``n - 1`` degrees of freedom.

    Parameters
    ----------
    loss_a, loss_b : np.ndarray
        Per-observation loss series of equal length.
    horizon : int, default 1
        Forecast horizon ``h``. When ``lag`` is not given, the truncation lag
        defaults to ``h - 1`` (the textbook choice for h-step forecasts).
    lag : int, optional
        Override for the HAC truncation lag.

    Returns
    -------
    dict[str, float]
        Keys: ``mean_diff`` (mean of A minus B), ``dm_stat`` (HLN-corrected),
        ``p_value`` (two-sided, t with n-1 df), ``favored`` (-1 if A, +1 if B,
        0 if indistinguishable at the data level), ``n``.

    Raises
    ------
    ValueError
        If the inputs differ in length or are too short.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"loss series must match in length, got {a.shape} and {b.shape}")
    n = a.size
    if n < 8:
        raise ValueError(f"need at least 8 observations for DM, got {n}")

    d = a - b
    mean_d = float(d.mean())
    trunc = (horizon - 1) if lag is None else lag
    trunc = max(0, int(trunc))
    lrv = _newey_west_lrv(d, trunc)
    if lrv <= 0:
        return {
            "mean_diff": mean_d,
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "favored": 0.0,
            "n": float(n),
        }

    dm = mean_d / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold (1997) small-sample correction. The factor uses the
    # forecast horizon h (not the HAC truncation lag h-1 stored in `trunc`); using
    # `trunc` here is anti-conservative — negligible at large n but can flip
    # inference at small n.
    hln_factor = np.sqrt((n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n)
    dm_corrected = dm * hln_factor

    # Two-sided p-value from Student-t(n-1) via its survival function.
    p_value = 2.0 * _student_t_sf(abs(dm_corrected), df=n - 1)
    favored = -1.0 if mean_d < 0 else (1.0 if mean_d > 0 else 0.0)

    return {
        "mean_diff": mean_d,
        "dm_stat": float(dm_corrected),
        "p_value": float(p_value),
        "favored": favored,
        "n": float(n),
    }


def _student_t_sf(x: float, df: int) -> float:
    """Upper-tail probability of a Student-t distribution, ``P(T > x)``.

    Uses the regularised incomplete beta function so that no SciPy dependency
    is required inside this module.

    Parameters
    ----------
    x : float
        Quantile (assumed non-negative for the upper tail).
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        Survival-function value.
    """
    if df <= 0:
        return float("nan")
    # P(T > x) = 0.5 * I_{df/(df + x^2)}(df/2, 1/2) for x >= 0.
    xb = df / (df + x * x)
    return 0.5 * _reg_incomplete_beta(xb, df / 2.0, 0.5)


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)`` via continued fraction.

    A compact Lentz-algorithm implementation (Numerical Recipes), adequate for
    p-value computation.

    Parameters
    ----------
    x : float
        Argument in [0, 1].
    a, b : float
        Positive shape parameters.

    Returns
    -------
    float
        ``I_x(a, b)``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    from math import exp, lgamma, log

    ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(log(x) * a + log(1.0 - x) * b - ln_beta) / a

    # Continued fraction (Lentz).
    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    result = front * h
    # Use the symmetry relation when x is past the convergence-friendly region.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _reg_incomplete_beta(1.0 - x, b, a)
    return min(max(result, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------
@dataclass
class MCSResult:
    """Outcome of a Model Confidence Set procedure.

    Attributes
    ----------
    included : list[str]
        Models in the confidence set at the chosen ``alpha``.
    p_values : dict[str, float]
        MCS p-value for each model (monotone in elimination order); a model is
        in the set iff its p-value exceeds ``alpha``.
    elimination_order : list[str]
        Models in the order they were eliminated (worst first).
    mean_losses : dict[str, float]
        Average loss per model over the evaluation sample.
    alpha : float
        Significance level used to form ``included``.
    """

    included: list[str]
    p_values: dict[str, float]
    elimination_order: list[str]
    mean_losses: dict[str, float]
    alpha: float = 0.10


def _moving_block_indices(
    n: int, block_length: int, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a *circular* moving-block-bootstrap index matrix of shape ``(reps, n)``.

    Parameters
    ----------
    n : int
        Sample length.
    block_length : int
        Length of each contiguous block.
    reps : int
        Number of bootstrap replications.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Integer index matrix; row ``b`` is one resampled time ordering.
    """
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(reps, n_blocks))
    offsets = np.arange(block_length)
    # Build (reps, n_blocks, block_length) then flatten and trim to n, modulo n
    # so blocks wrap around (circular block bootstrap).
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(reps, n_blocks * block_length)[:, :n]
    return idx


def model_confidence_set(
    losses: dict[str, np.ndarray],
    alpha: float = 0.10,
    block_length: int = _DEFAULT_BLOCK_LENGTH,
    reps: int = _DEFAULT_BOOTSTRAP_REPS,
    seed: int | None = 0,
) -> MCSResult:
    """Model Confidence Set via the range statistic (Hansen-Lunde-Nason 2011).

    Iteratively tests the null that all surviving models have equal expected
    loss. The range statistic is
    ``T_R = max_{i,j} |dbar_ij| / sqrt(var(dbar_ij))`` where
    ``dbar_ij`` is the mean loss differential between models i and j and the
    variance is estimated by a moving-block bootstrap. If the null is rejected
    at level ``alpha``, the model with the largest standardised excess loss is
    removed and the procedure repeats. Each model's MCS p-value is the running
    maximum of the test p-values at and before its elimination, which yields a
    monotone p-value sequence.

    Parameters
    ----------
    losses : dict[str, np.ndarray]
        Mapping ``model_name -> per-observation loss array``. All arrays must
        share the same length and time alignment.
    alpha : float, default 0.10
        Confidence level; the returned set has coverage ``1 - alpha``.
    block_length : int, default 10
        Moving-block-bootstrap block length (handles serial dependence).
    reps : int, default 2000
        Number of bootstrap replications.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    MCSResult
        The confidence set, p-values, elimination order and mean losses.

    Raises
    ------
    ValueError
        If fewer than two models are supplied or arrays are misaligned.
    """
    names = list(losses.keys())
    if len(names) < 2:
        raise ValueError("MCS requires at least two models")
    mat = np.column_stack([np.asarray(losses[k], dtype=float).ravel() for k in names])
    n, m = mat.shape
    lengths = {k: np.asarray(v).size for k, v in losses.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"loss arrays must share length, got {lengths}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("loss arrays contain non-finite values")

    rng = np.random.default_rng(seed)
    boot_idx = _moving_block_indices(n, block_length, reps, rng)

    mean_losses_full = {names[j]: float(mat[:, j].mean()) for j in range(m)}

    alive = list(range(m))  # column indices still in the set
    elimination_order: list[str] = []
    raw_pvalues: list[float] = []  # p-value recorded when each model is eliminated

    while len(alive) > 1:
        sub = mat[:, alive]  # (n, k)
        k = len(alive)
        col_means = sub.mean(axis=0)  # (k,)

        # Bootstrap column means: (reps, k). Average over resampled rows.
        boot_means = sub[boot_idx, :].mean(axis=1)  # (reps, k)
        boot_centered = boot_means - col_means[None, :]  # center at sample mean

        # Pairwise differences. dbar_ij = col_means[i] - col_means[j].
        dbar = col_means[:, None] - col_means[None, :]  # (k, k)
        # Bootstrap variance of each pairwise mean differential.
        # diff_boot[b, i, j] = boot_centered[b, i] - boot_centered[b, j]
        diff_boot = boot_centered[:, :, None] - boot_centered[:, None, :]  # (reps,k,k)
        var_ij = diff_boot.var(axis=0)  # (k, k)
        std_ij = np.sqrt(np.maximum(var_ij, _MIN_BOOTSTRAP_VARIANCE))

        # Observed range statistic over i<j (matrix is antisymmetric in dbar).
        with np.errstate(invalid="ignore", divide="ignore"):
            t_ij = np.abs(dbar) / std_ij
        np.fill_diagonal(t_ij, 0.0)
        t_r_obs = float(t_ij.max())

        # Bootstrap distribution of the range statistic under the null.
        t_ij_boot = np.abs(diff_boot) / std_ij[None, :, :]
        # zero the diagonal across all reps
        diag = np.arange(k)
        t_ij_boot[:, diag, diag] = 0.0
        t_r_boot = t_ij_boot.reshape(reps, k * k).max(axis=1)  # (reps,)

        p_value = float((t_r_boot >= t_r_obs).mean())

        if p_value > alpha:
            # Null not rejected: every surviving model stays in the set.
            break

        # Reject: eliminate the worst model = largest standardised mean
        # differential against the set average (HLN elimination rule).
        set_mean = col_means.mean()
        var_to_setmean = diff_boot.mean(axis=2).var(axis=0)  # var of (col - setmean)
        std_to_setmean = np.sqrt(np.maximum(var_to_setmean, _MIN_BOOTSTRAP_VARIANCE))
        elim_stat = (col_means - set_mean) / std_to_setmean
        worst_local = int(np.argmax(elim_stat))
        worst_global = alive[worst_local]

        elimination_order.append(names[worst_global])
        raw_pvalues.append(p_value)
        alive.pop(worst_local)

    # Any model never eliminated belongs to the set with MCS p-value 1.
    survivors = [names[j] for j in alive]
    for s in survivors:
        elimination_order.append(s)
        raw_pvalues.append(1.0)

    # MCS p-values are the cumulative maximum along the elimination order.
    mcs_p: dict[str, float] = {}
    running = 0.0
    for name, pv in zip(elimination_order, raw_pvalues):
        running = max(running, pv)
        mcs_p[name] = float(running)

    included = [name for name, pv in mcs_p.items() if pv > alpha]
    # Guarantee the set is non-empty (the last survivor is always retained).
    if not included and survivors:
        included = list(survivors)

    return MCSResult(
        included=included,
        p_values=mcs_p,
        elimination_order=[n for n in elimination_order if n not in survivors] + survivors,
        mean_losses=mean_losses_full,
        alpha=alpha,
    )
