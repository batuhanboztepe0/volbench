"""Economic-value evaluation layer (Tier 1B) for realized-volatility forecasts.

Complements the statistical losses in :mod:`volbench.losses` with three
utility-based criteria that ask whether better forecasts translate into better
decisions:

1. **Vol-targeting** — a long-only strategy that scales its equity position so
   that the forecast daily volatility equals a target, then measures the
   Sharpe ratio, drawdown, and turnover of the resulting return stream.

2. **Option pricing** — prices an ATM European call with each model's forecast
   vol and measures how far those prices deviate from the fair price implied by
   the realized vol.

3. **Value-at-Risk backtesting** — compares the empirical violation rate of a
   normal VaR against the nominal level via the Kupiec unconditional-coverage
   test and the Christoffersen conditional-coverage test.

All public functions take raw numpy arrays and return plain dicts so they are
trivially unit-testable without any backtest infrastructure.

References
----------
- Kupiec (1995), "Techniques for verifying the accuracy of risk measurement
  models", *Journal of Derivatives*.
- Christoffersen (1998), "Evaluating interval forecasts", *International
  Economic Review*.
- Black & Scholes (1973), "The pricing of options and corporate liabilities",
  *Journal of Political Economy*.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annualize_vol(daily_variance: np.ndarray, trading_days: int = 252) -> np.ndarray:
    """Convert daily variance to annualised volatility."""
    return np.sqrt(np.maximum(daily_variance, 0.0) * trading_days)


def _equity_curve(strategy_returns: np.ndarray) -> np.ndarray:
    """Compounded equity curve starting at 1.0."""
    return np.cumprod(1.0 + strategy_returns)


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown on a compounded equity curve."""
    if equity.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    return float(drawdowns.min())


# ---------------------------------------------------------------------------
# 1. Vol-targeting strategy
# ---------------------------------------------------------------------------

def volatility_targeting(
    future_returns: np.ndarray,
    forecast_variance: np.ndarray,
    target_ann_vol: float = 0.10,
    max_leverage: float = 4.0,
    trading_days: int = 252,
) -> dict[str, float]:
    """Vol-targeting strategy: scale positions to hit a constant volatility target.

    At each origin ``t`` the position weight is::

        w_t = target_daily_vol / sqrt(forecast_variance[t])

    clipped to ``[0, max_leverage]``. The strategy return at step ``t`` is
    ``w_t * future_returns[t]``.

    Parameters
    ----------
    future_returns : np.ndarray
        Realized next-period returns aligned to the forecast origins. Shape
        ``(T,)``, decimal log returns.
    forecast_variance : np.ndarray
        Variance forecasts aligned to the same origins. Shape ``(T,)``, daily
        variance in decimal returns².
    target_ann_vol : float, default 0.10
        Target annualised volatility (10 % = 0.10).
    max_leverage : float, default 4.0
        Upper bound on the position weight; lower bound is 0.0 (no shorting).
    trading_days : int, default 252
        Convention for annualisation.

    Returns
    -------
    dict[str, float]
        Keys: ``ann_return``, ``ann_vol``, ``sharpe``, ``max_drawdown``,
        ``turnover``, ``avg_leverage``, ``bh_sharpe``, ``bh_ann_vol``.
    """
    ret = np.asarray(future_returns, dtype=float).ravel()
    fvar = np.asarray(forecast_variance, dtype=float).ravel()
    if ret.shape != fvar.shape:
        raise ValueError(
            f"future_returns and forecast_variance must match in length, "
            f"got {ret.shape} and {fvar.shape}"
        )
    if ret.size == 0:
        raise ValueError("inputs are empty")

    target_daily_vol = target_ann_vol / np.sqrt(trading_days)
    forecast_daily_vol = np.sqrt(np.maximum(fvar, 1e-300))

    weights = np.clip(target_daily_vol / forecast_daily_vol, 0.0, max_leverage)
    strat_ret = weights * ret

    # Annualised return and vol from the strategy return series.
    ann_return = float(np.mean(strat_ret) * trading_days)
    ann_vol = float(np.std(strat_ret, ddof=1) * np.sqrt(trading_days))
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    equity = _equity_curve(strat_ret)
    mdd = _max_drawdown(equity)
    turnover = float(np.mean(np.abs(np.diff(weights)))) if weights.size > 1 else 0.0
    avg_leverage = float(np.mean(weights))

    # Buy-and-hold benchmark (w = 1 throughout).
    bh_ann_vol = float(np.std(ret, ddof=1) * np.sqrt(trading_days))
    bh_ann_ret = float(np.mean(ret) * trading_days)
    bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else float("nan")

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "turnover": turnover,
        "avg_leverage": avg_leverage,
        "bh_sharpe": bh_sharpe,
        "bh_ann_vol": bh_ann_vol,
    }


# ---------------------------------------------------------------------------
# 2. Black-Scholes pricing
# ---------------------------------------------------------------------------

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool = True,
) -> float:
    """Standard Black-Scholes European option price.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Annualised volatility.
    call : bool, default True
        If True return the call price; False returns the put price.

    Returns
    -------
    float
        Option price. Returns 0.0 for non-positive ``T`` or ``sigma``.
    """
    if T <= 0.0 or sigma <= 0.0:
        # Intrinsic value only.
        intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
        return float(intrinsic)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


# ---------------------------------------------------------------------------
# 3. Option-pricing loss
# ---------------------------------------------------------------------------

def option_pricing_loss(
    forecast_variance: np.ndarray,
    realized_variance: np.ndarray,
    horizon_days: int,
    moneyness: float = 1.0,
    r: float = 0.0,
) -> dict[str, float]:
    """Option-pricing loss: price an ATM call with forecast vol vs realized vol.

    For each origin, price a European call with maturity ``T = horizon_days /
    252`` using the forecast annualised vol ``sqrt(forecast_variance * 252)``
    and again using the realized annualised vol ``sqrt(realized_variance * 252)``
    (the "true" fair price). The loss per origin is the squared price difference.

    Parameters
    ----------
    forecast_variance : np.ndarray
        Variance forecasts, daily decimal scale. Shape ``(T,)``.
    realized_variance : np.ndarray
        Realized variance, daily decimal scale. Shape ``(T,)``.
    horizon_days : int
        Option maturity in trading days.
    moneyness : float, default 1.0
        ``K / S``; 1.0 gives ATM options. Spot is fixed at 100 so
        ``S = 100`` and ``K = moneyness * 100``.
    r : float, default 0.0
        Risk-free rate (annualised, continuously compounded).

    Returns
    -------
    dict[str, float]
        Keys: ``mean_squared_price_error``, ``rmse_price``,
        ``mean_abs_price_error``.
    """
    fvar = np.asarray(forecast_variance, dtype=float).ravel()
    rvar = np.asarray(realized_variance, dtype=float).ravel()
    if fvar.shape != rvar.shape:
        raise ValueError(
            f"forecast_variance and realized_variance must match in length, "
            f"got {fvar.shape} and {rvar.shape}"
        )
    if fvar.size == 0:
        raise ValueError("inputs are empty")

    S = 100.0
    K = moneyness * S
    T = horizon_days / 252.0

    forecast_ann_vol = np.sqrt(np.maximum(fvar, 0.0) * 252.0)
    realized_ann_vol = np.sqrt(np.maximum(rvar, 0.0) * 252.0)

    price_errors = np.empty(fvar.size)
    for i in range(fvar.size):
        p_forecast = black_scholes_price(S, K, T, r, float(forecast_ann_vol[i]), call=True)
        p_realized = black_scholes_price(S, K, T, r, float(realized_ann_vol[i]), call=True)
        price_errors[i] = p_forecast - p_realized

    sq_errors = price_errors ** 2
    abs_errors = np.abs(price_errors)

    return {
        "mean_squared_price_error": float(np.mean(sq_errors)),
        "rmse_price": float(np.sqrt(np.mean(sq_errors))),
        "mean_abs_price_error": float(np.mean(abs_errors)),
    }


# ---------------------------------------------------------------------------
# 4. VaR backtesting
# ---------------------------------------------------------------------------

def _kupiec_lr(n: int, n_viol: int, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional-coverage likelihood-ratio test.

    Parameters
    ----------
    n : int
        Total number of observations.
    n_viol : int
        Number of VaR violations.
    alpha : float
        Nominal VaR level (e.g. 0.05).

    Returns
    -------
    tuple[float, float]
        ``(LR_uc statistic, p-value)`` under chi-squared(1). Returns
        ``(nan, nan)`` for degenerate cases.
    """
    if n_viol == 0 or n_viol == n:
        return float("nan"), float("nan")
    p_hat = n_viol / n
    # Log-likelihood ratio: unrestricted vs restricted (p = alpha).
    lr = -2.0 * (
        n_viol * np.log(alpha / p_hat)
        + (n - n_viol) * np.log((1.0 - alpha) / (1.0 - p_hat))
    )
    p_val = float(chi2.sf(lr, df=1))
    return float(lr), p_val


def _christoffersen_lr(
    n00: int, n01: int, n10: int, n11: int, alpha: float
) -> tuple[float, float]:
    """Christoffersen (1998) conditional-coverage (independence + coverage) test.

    Uses LR_cc = LR_uc + LR_ind, both asymptotically chi-squared.

    Parameters
    ----------
    n00, n01, n10, n11 : int
        Transition counts: n_ij = # times state i followed by state j,
        where state 1 = violation, state 0 = no violation.
    alpha : float
        Nominal VaR level.

    Returns
    -------
    tuple[float, float]
        ``(LR_cc statistic, p-value)`` under chi-squared(2).
    """
    n = n00 + n01 + n10 + n11
    n_viol = n01 + n11

    # Unconditional coverage component.
    lr_uc, _ = _kupiec_lr(n, n_viol, alpha)
    if not np.isfinite(lr_uc):
        return float("nan"), float("nan")

    # Independence component (Christoffersen 1998, eq 11).
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = n_viol / n  # unconditional

    if pi01 == 0.0 or pi11 == 0.0 or pi == 0.0 or pi == 1.0:
        return float("nan"), float("nan")
    if pi01 == 1.0 or pi11 == 1.0:
        return float("nan"), float("nan")

    ll_restricted = (
        (n00 + n10) * np.log(1.0 - pi)
        + (n01 + n11) * np.log(pi)
    )
    ll_unrestricted = (
        n00 * np.log(1.0 - pi01)
        + n01 * np.log(pi01)
        + n10 * np.log(1.0 - pi11)
        + n11 * np.log(pi11)
    )
    lr_ind = -2.0 * (ll_restricted - ll_unrestricted)

    lr_cc = float(lr_uc + lr_ind)
    p_val = float(chi2.sf(lr_cc, df=2))
    return lr_cc, p_val


def engle_manganelli_dq(
    violations: np.ndarray,
    forecast_var: np.ndarray,
    alpha: float,
    lags: int = 4,
) -> dict[str, float]:
    """Engle-Manganelli (2004) Dynamic Quantile (DQ) test.

    Regresses the centred hit series ``H_t = I_t - alpha`` on a constant,
    ``lags`` lagged hits, and the contemporaneous ``forecast_var`` (as a proxy
    for the VaR level). Under correct specification the regressors should be
    orthogonal to ``H_t``.

    The test statistic is::

        DQ = (beta' X'X beta) / (alpha * (1 - alpha))  ~  chi2(k)

    where ``k = 1 + lags + 1`` is the number of regressors.

    Parameters
    ----------
    violations : np.ndarray
        Boolean or integer array of VaR violations (1 = violation). Shape
        ``(T,)``.
    forecast_var : np.ndarray
        Variance forecasts aligned to violations. Shape ``(T,)``.
    alpha : float
        Nominal VaR level used to build the hit series.
    lags : int, default 4
        Number of lagged hits included as regressors.

    Returns
    -------
    dict[str, float]
        Keys: ``dq_stat``, ``dq_pvalue``.
    """
    viol = np.asarray(violations, dtype=float).ravel()
    fvar = np.asarray(forecast_var, dtype=float).ravel()
    if viol.shape != fvar.shape:
        raise ValueError(
            f"violations and forecast_var must match in length, "
            f"got {viol.shape} and {fvar.shape}"
        )

    T = viol.size
    n = T - lags  # effective sample after consuming lags
    if n <= lags + 2:
        return {"dq_stat": float("nan"), "dq_pvalue": float("nan")}

    hit = viol - alpha  # centred hit series H_t

    # Build regressor matrix X: [const, H_{t-1}, ..., H_{t-lags}, fvar_t]
    # Shape: (n, k) where k = 1 + lags + 1
    k = 1 + lags + 1
    X = np.empty((n, k))
    X[:, 0] = 1.0
    for lag in range(1, lags + 1):
        X[:, lag] = hit[lags - lag : T - lag]
    X[:, -1] = fvar[lags:]

    y = hit[lags:]  # H_{t} for t = lags, ..., T-1

    # OLS estimate: beta = (X'X)^{-1} X'y, but we only need beta for the stat.
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return {"dq_stat": float("nan"), "dq_pvalue": float("nan")}

    dq_stat = float((beta @ XtX @ beta) / (alpha * (1.0 - alpha)))
    dq_pvalue = float(chi2.sf(dq_stat, df=k))
    return {"dq_stat": dq_stat, "dq_pvalue": dq_pvalue}


def var_backtest(
    future_returns: np.ndarray,
    forecast_variance: np.ndarray,
    alpha: float = 0.05,
    dist: str = "normal",
) -> dict[str, float]:
    """VaR backtest with Kupiec, Christoffersen, and Dynamic Quantile tests.

    The one-step VaR at level ``alpha`` is computed as:

    * ``"normal"`` — ``VaR_t = norm.ppf(1-alpha) * sqrt(forecast_variance[t])``
    * ``"t"`` — fit a Student-t to the standardised residuals
      ``z_t = return_t / sqrt(forecast_variance[t])`` to estimate degrees of
      freedom, then ``VaR_t = t.ppf(1-alpha, dof, scale=sqrt((dof-2)/dof)) *
      sqrt(forecast_variance[t])`` (scale adjusted so the t distribution has
      unit variance).
    * ``"fhs"`` — filtered historical simulation: use the empirical
      ``alpha``-quantile of standardised residuals
      ``z_t = return_t / sqrt(forecast_variance[t])`` and multiply back by
      ``sqrt(forecast_variance[t])``.

    A violation occurs when ``future_returns[t] < -VaR_t``.

    Parameters
    ----------
    future_returns : np.ndarray
        Realized returns at each origin (decimal). Shape ``(T,)``.
    forecast_variance : np.ndarray
        Variance forecasts, daily decimal scale. Shape ``(T,)``.
    alpha : float, default 0.05
        VaR confidence level (5 % tail).
    dist : str, default ``"normal"``
        Distribution assumption: ``"normal"``, ``"t"``, or ``"fhs"``.

    Returns
    -------
    dict[str, float]
        Keys: ``violation_rate``, ``expected_rate``, ``n_violations``, ``n``,
        ``kupiec_stat``, ``kupiec_p``, ``christoffersen_stat``,
        ``christoffersen_p``, ``dq_stat``, ``dq_pvalue``.
    """
    _VALID_DISTS = {"normal", "t", "fhs"}
    ret = np.asarray(future_returns, dtype=float).ravel()
    fvar = np.asarray(forecast_variance, dtype=float).ravel()
    if ret.shape != fvar.shape:
        raise ValueError(
            f"future_returns and forecast_variance must match in length, "
            f"got {ret.shape} and {fvar.shape}"
        )
    if ret.size == 0:
        raise ValueError("inputs are empty")
    if dist not in _VALID_DISTS:
        raise ValueError(f"dist={dist!r} is not supported; choose from {_VALID_DISTS}")

    vol_t = np.sqrt(np.maximum(fvar, 0.0))
    z = np.where(vol_t > 0.0, ret / vol_t, 0.0)  # standardised residuals

    if dist == "normal":
        z_alpha = float(norm.ppf(1.0 - alpha))
        var_t = z_alpha * vol_t
    elif dist == "t":
        # Fit Student-t dof to the standardised residuals via MLE.
        # scipy's t.fit returns (df, loc, scale); we fix loc=0.
        fit_df, fit_loc, fit_scale = t_dist.fit(z, floc=0.0)
        dof = max(float(fit_df), 2.1)  # guard: t variance needs dof > 2
        # Scale so that the t distribution used for VaR has unit variance.
        unit_var_scale = float(np.sqrt((dof - 2.0) / dof))
        z_alpha_t = float(t_dist.ppf(1.0 - alpha, dof, scale=unit_var_scale))
        var_t = z_alpha_t * vol_t
    else:  # fhs
        q_alpha = float(np.quantile(z, alpha))  # negative number for left tail
        var_t = -q_alpha * vol_t

    violations = ret < -var_t  # boolean array

    n = int(ret.size)
    n_viol = int(violations.sum())
    violation_rate = n_viol / n

    # Kupiec test.
    kupiec_stat, kupiec_p = _kupiec_lr(n, n_viol, alpha)

    # Christoffersen test: build transition counts from the violation sequence.
    v = violations.astype(int)
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())

    christoffersen_stat, christoffersen_p = _christoffersen_lr(n00, n01, n10, n11, alpha)

    # Dynamic Quantile test.
    dq = engle_manganelli_dq(violations.astype(int), fvar, alpha)

    return {
        "violation_rate": float(violation_rate),
        "expected_rate": float(alpha),
        "n_violations": float(n_viol),
        "n": float(n),
        "kupiec_stat": float(kupiec_stat),
        "kupiec_p": float(kupiec_p),
        "christoffersen_stat": float(christoffersen_stat),
        "christoffersen_p": float(christoffersen_p),
        "dq_stat": dq["dq_stat"],
        "dq_pvalue": dq["dq_pvalue"],
    }
