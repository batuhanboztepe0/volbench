"""Economic-value evaluation layer (Tier 1B) for realized-volatility forecasts.

Complements the statistical losses in :mod:`volbench.losses` with three
utility-based criteria that ask whether better forecasts translate into better
decisions:

1. **Vol-targeting**: a long-only strategy that scales its equity position so
   that the forecast daily volatility equals a target, then measures the
   Sharpe ratio, drawdown, and turnover of the resulting return stream.

2. **Option pricing**: prices an ATM European call with each model's forecast
   vol and measures how far those prices deviate from the fair price implied by
   the realized vol.

3. **Value-at-Risk backtesting**: compares the empirical violation rate of a
   normal VaR against the nominal level via the Kupiec unconditional-coverage
   test and the Christoffersen conditional-coverage test.

4. **Expected Shortfall (ES/CVaR)**: Basel-FRTB risk measure: the conditional
   mean return given a VaR breach.  ES forecasts are ranked via the
   Fissler-Ziegel (FZ0) loss and tested with the Acerbi-Székely (2014) Z1/Z2
   statistics.  ES is a *risk-layer* quantity: never compare FZ0 values to
   QLIKE from the RV-forecast track (Invariant 4).

Sign conventions
----------------
All ES values in this module are **negative** (left-tail losses on decimal
returns).  A breach occurs when ``return_t < -VaR_t`` (VaR positive), and the
ES is the conditional mean of those negative returns.

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
- Acerbi & Székely (2014), "Backtesting Expected Shortfall", *Risk*, 27, 76–81.
- Fissler & Ziegel (2016), "Higher order elicitability and Osband's principle",
  *Annals of Statistics*.
- Taylor (2019), "Forecasting value at risk and expected shortfall using a
  semiparametric approach", *Management Science*.
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
    warmup: int = 500,
    *,
    return_es: bool = False,
) -> dict[str, float]:
    """VaR backtest with Kupiec, Christoffersen, and Dynamic Quantile tests.

    The one-step VaR at level ``alpha`` is computed as:

    * ``"normal"``: ``VaR_t = norm.ppf(1-alpha) * sqrt(forecast_variance[t])``
    * ``"t"``: fit a Student-t to the standardised residuals
      ``z_t = return_t / sqrt(forecast_variance[t])`` to estimate degrees of
      freedom, then ``VaR_t = t.ppf(1-alpha, dof, scale=sqrt((dof-2)/dof)) *
      sqrt(forecast_variance[t])`` (scale adjusted so the t distribution has
      unit variance).
    * ``"fhs"``: filtered historical simulation: use the empirical
      ``alpha``-quantile of standardised residuals
      ``z_t = return_t / sqrt(forecast_variance[t])`` and multiply back by
      ``sqrt(forecast_variance[t])``.

    The estimated distributions (``"t"``, ``"fhs"``) are calibrated **only on a
    leading warm-up block** of standardised residuals and coverage is evaluated on
    the held-out remainder; otherwise the tail shape would be fit on the very
    residuals it is scored against (for ``"fhs"`` the violation rate would equal
    ``alpha`` by construction). ``"normal"`` needs no calibration but is scored on
    the same held-out window so all distributions share one out-of-sample sample.

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
    warmup : int, default 500
        Number of leading observations used to calibrate the estimated
        distributions (``"t"``, ``"fhs"``); these are excluded from scoring. Capped
        at ``n // 3`` so a majority of the sample is always evaluated.
    return_es : bool, default False
        If True, also compute ES forecasts and append Acerbi-Székely backtest
        and FZ loss to the output dict.  Default ``False`` preserves the
        original return signature exactly.

    Returns
    -------
    dict[str, float]
        Keys: ``violation_rate``, ``expected_rate``, ``n_violations``, ``n``,
        ``kupiec_stat``, ``kupiec_p``, ``christoffersen_stat``,
        ``christoffersen_p``, ``dq_stat``, ``dq_pvalue``.
        When ``return_es=True`` the dict additionally contains
        ``es_mean``, ``as_Z1``, ``as_Z2``, ``as_p``, ``fz_mean``.
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

    # Calibrate any *estimated* tail shape on a leading warm-up block only, then
    # score coverage on the held-out remainder (see the docstring): this removes
    # the in-sample look-ahead (and, for FHS, the violation-rate == alpha
    # tautology) that arises from fitting the quantile on the scored residuals.
    w = min(int(warmup), ret.size // 3)
    z_train = z[:w]
    if dist in ("t", "fhs") and z_train.size < 30:
        raise ValueError(
            f"dist={dist!r} needs at least 30 warm-up observations to calibrate; "
            f"got {z_train.size} from n={ret.size}"
        )

    if dist == "normal":
        mult = float(norm.ppf(1.0 - alpha))
    elif dist == "t":
        # Fit Student-t dof on the warm-up residuals only (out-of-sample).
        # scipy's t.fit returns (df, loc, scale); we fix loc=0.
        fit_df, _, _ = t_dist.fit(z_train, floc=0.0)
        dof = max(float(fit_df), 2.1)  # guard: t variance needs dof > 2
        # Scale so that the t distribution used for VaR has unit variance.
        unit_var_scale = float(np.sqrt((dof - 2.0) / dof))
        mult = float(t_dist.ppf(1.0 - alpha, dof, scale=unit_var_scale))
    else:  # fhs
        mult = -float(np.quantile(z_train, alpha))  # empirical quantile, past only

    var_t = mult * vol_t

    # Evaluate only on the held-out window [w:].
    ret_s = ret[w:]
    var_s = var_t[w:]
    violations = ret_s < -var_s  # boolean array

    n = int(ret_s.size)
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

    # Dynamic Quantile test (on the held-out window). Use the VaR level itself as
    # the DQ regressor so this path matches backtest_var_forecasts (the CAViaR path)
    # and the two engines are compared on the same specification.
    dq = engle_manganelli_dq(v, var_s, alpha)

    result: dict[str, float] = {
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

    if return_es:
        # Use the SAME warm-up block ``w`` so the ES VaR thresholds coincide with
        # the thresholds scored above, then evaluate ES on the SAME held-out
        # window ``[w:]`` (WFB-2: no VaR-vs-ES threshold mismatch for t / FHS).
        es_dict = expected_shortfall_forecast(
            future_returns=ret,
            forecast_variance=fvar,
            alpha=alpha,
            dist=dist,
            warmup=w,
        )
        es_s = es_dict["es_forecast"][w:]
        var_es_s = es_dict["var_forecast"][w:]
        as_res = acerbi_szekely_backtest(ret_s, es_s, var_es_s, alpha)
        fz = fz_loss(ret_s, var_es_s, es_s, alpha)
        result["es_mean"] = float(np.mean(es_s))
        result["as_Z1"] = as_res["Z1"]
        result["as_Z2"] = as_res["Z2"]
        result["as_p"] = as_res["p"]
        result["fz_mean"] = float(np.mean(fz))

    return result


def backtest_var_forecasts(
    future_returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Backtest VaR forecasts supplied **directly** (e.g. from CAViaR).

    Unlike :func:`var_backtest`, which derives the VaR from a variance forecast and
    a distributional assumption, this scores models that produce the ``alpha``-tail
    VaR itself. Origins with a non-finite ``var_forecasts`` value (e.g. the leading
    walk-forward warm-up) are dropped; the remainder is scored with the same
    Kupiec, Christoffersen, and Engle-Manganelli Dynamic-Quantile tests as
    :func:`var_backtest`, so the two are directly comparable.

    A violation occurs when ``future_returns[t] < -var_forecasts[t]``.

    Parameters
    ----------
    future_returns : np.ndarray
        Realized returns (decimal). Shape ``(T,)``.
    var_forecasts : np.ndarray
        One-step VaR forecasts (positive loss thresholds). Shape ``(T,)``; may
        contain leading/missing ``nan`` which are excluded from scoring.
    alpha : float, default 0.05
        VaR confidence level.

    Returns
    -------
    dict[str, float]
        Same keys as :func:`var_backtest` (minus the ES extras): ``violation_rate``,
        ``expected_rate``, ``n_violations``, ``n``, ``kupiec_stat``, ``kupiec_p``,
        ``christoffersen_stat``, ``christoffersen_p``, ``dq_stat``, ``dq_pvalue``.
    """
    ret = np.asarray(future_returns, dtype=float).ravel()
    var = np.asarray(var_forecasts, dtype=float).ravel()
    if ret.shape != var.shape:
        raise ValueError(
            f"future_returns and var_forecasts must match in length, "
            f"got {ret.shape} and {var.shape}"
        )
    mask = np.isfinite(var) & np.isfinite(ret)
    ret_s = ret[mask]
    var_s = var[mask]
    if ret_s.size < 30:
        raise ValueError(f"need at least 30 scored origins, got {ret_s.size}")

    violations = ret_s < -var_s
    n = int(ret_s.size)
    n_viol = int(violations.sum())

    kupiec_stat, kupiec_p = _kupiec_lr(n, n_viol, alpha)

    v = violations.astype(int)
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())
    christoffersen_stat, christoffersen_p = _christoffersen_lr(n00, n01, n10, n11, alpha)

    # Use the VaR level itself as the DQ specification regressor.
    dq = engle_manganelli_dq(v, var_s, alpha)

    return {
        "violation_rate": float(n_viol / n),
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


# ---------------------------------------------------------------------------
# 5. Expected Shortfall forecasts
# ---------------------------------------------------------------------------

def expected_shortfall_forecast(
    future_returns: np.ndarray,
    forecast_variance: np.ndarray,
    alpha: float = 0.05,
    dist: str = "normal",
    warmup: int = 0,
) -> dict[str, np.ndarray]:
    """One-step ES forecasts under normal, Student-t, or FHS assumptions.

    ES is the conditional mean return given a VaR breach; it is a **negative**
    number (left-tail loss in decimal return units).

    Computation mirrors ``var_backtest``: standardised residuals
    ``z_t = return_t / sqrt(forecast_variance[t])`` are computed over the full
    input window.  For ``dist="fhs"`` the first ``warmup`` observations are
    used to calibrate the empirical tail (if ``warmup=0``, half the series
    is used as warmup, mirroring the walk-forward convention).

    Parameters
    ----------
    future_returns : np.ndarray
        Realized returns, decimal. Shape ``(T,)``.
    forecast_variance : np.ndarray
        Variance forecasts (daily, decimal²). Shape ``(T,)``.
        Must be strictly positive; values ≤ 0 are clipped to ``1e-300``.
    alpha : float, default 0.05
        VaR / ES probability level (left-tail, 5 %).
    dist : str, default ``"normal"``
        One of ``"normal"``, ``"t"``, ``"fhs"``.
    warmup : int, default 0
        Number of observations used to calibrate standardised residuals for
        FHS.  If 0, defaults to ``max(1, T // 2)``.  Has no effect for
        ``dist`` in ``{"normal", "t"}``.

    Returns
    -------
    dict[str, np.ndarray]
        ``"es_forecast"`` : shape ``(T,)``, negative decimal returns (units:
            decimal daily return; ES at level ``alpha`` so approx -1.6 * sigma
            for normal at 5 %).
        ``"var_forecast"`` : shape ``(T,)``, the VaR thresholds used (positive,
            decimal daily return) so that a breach is ``return_t < -var_t``.
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
        raise ValueError(f"dist={dist!r} not in {_VALID_DISTS}")

    vol_t = np.sqrt(np.maximum(fvar, 1e-300))
    z = np.where(vol_t > 0.0, ret / vol_t, 0.0)  # standardised residuals

    # Calibrate any estimated tail (the t dof, the FHS quantile) on the leading
    # warm-up block only, so that when var_backtest calls this with its own
    # ``warmup`` the ES thresholds coincide with the VaR thresholds it scored
    # (consistency, WFB-2). ``warmup=0`` defaults to half the window.
    w_es = warmup if warmup > 0 else max(1, ret.size // 2)
    z_calib = z[:w_es]

    if dist == "normal":
        # ES_normal = -sigma * phi(z_alpha) / alpha   (negative, z_alpha < 0)
        # Derivation: E[Z | Z < z_alpha] = -phi(z_alpha)/alpha for Z ~ N(0,1).
        z_alpha = float(norm.ppf(alpha))          # e.g. -1.645 at 5%
        es_z = -norm.pdf(z_alpha) / alpha         # negative scalar ~-2.06 at 5%
        # VaR threshold (positive) = -z_alpha * sigma
        var_t = -z_alpha * vol_t                  # positive
        es_t = es_z * vol_t                       # negative

    elif dist == "t":
        # Fit Student-t to the warm-up standardised residuals (same window as
        # var_backtest, so the t dof (hence the VaR/ES thresholds) coincide).
        fit_df, _fit_loc, _fit_scale = t_dist.fit(z_calib, floc=0.0)
        dof = max(float(fit_df), 2.1)  # variance requires dof > 2

        # Unit-variance Student-t: scale s.t. Var = 1 -> scale = sqrt((nu-2)/nu).
        # The alpha-quantile of the unit-variance t is:
        #   q = t.ppf(alpha, nu) * sqrt((nu-2)/nu)
        unit_var_scale = float(np.sqrt((dof - 2.0) / dof))
        q_alpha = float(t_dist.ppf(alpha, dof)) * unit_var_scale  # negative

        # Truncated-t mean (analytical, NOT a single arithmetic op):
        # For X ~ t(nu) scaled to unit variance,
        #   E[X | X < q_alpha] = -t_pdf(q_alpha/scale, nu) * (nu + (q_alpha/scale)^2)
        #                        / ((nu - 1) * alpha) * scale
        # where scale = sqrt((nu-2)/nu).
        # Unscaled quantile (back in t(nu) units):
        q_unscaled = q_alpha / unit_var_scale     # t.ppf(alpha, nu)
        # PDF of t(nu) evaluated at q_unscaled (strictly positive):
        pdf_q = float(t_dist.pdf(q_unscaled, dof))
        # Truncated mean of unit-variance t distribution:
        es_z = -(pdf_q * (dof + q_unscaled ** 2) / (dof - 1.0)) / alpha * unit_var_scale

        var_t = -q_alpha * vol_t                  # positive
        es_t = es_z * vol_t                       # negative

    else:  # fhs
        # z_calib (the warm-up block) was computed once above (WFB-2 consistency).
        tail = z_calib[z_calib <= np.quantile(z_calib, alpha)]
        if tail.size == 0:
            # Fallback: use all calibration residuals at or below the quantile.
            tail = z_calib
        es_z = float(np.mean(tail))              # negative
        q_alpha = float(np.quantile(z_calib, alpha))  # negative
        var_t = -q_alpha * vol_t                  # positive
        es_t = es_z * vol_t                       # negative

    return {
        "es_forecast": es_t,      # np.ndarray, negative, units: decimal return
        "var_forecast": var_t,    # np.ndarray, positive, units: decimal return
    }


# ---------------------------------------------------------------------------
# 6. Acerbi-Székely backtest
# ---------------------------------------------------------------------------

def acerbi_szekely_backtest(
    future_returns: np.ndarray,
    es_forecast: np.ndarray,
    var_forecast: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Acerbi-Székely (2014) expected-shortfall backtest (Z1 and Z2 tests).

    Both statistics are approximately zero under H0 (well-specified ES) and
    **negative** when the model underestimates tail risk.  p-values are
    one-sided (lower tail) via Monte Carlo under H0.

    Sign convention: this module uses **negative** ES (left-tail conditional
    mean), so the formulas are re-expressed relative to the original paper
    (which uses positive ES = loss magnitude) as follows:

    .. math::

        Z_1 = 1 - \\frac{1}{n_{\\text{breach}}}
              \\sum_{t:\\,r_t < -\\text{VaR}_t} \\frac{r_t}{\\text{ES}_t}

        Z_2 = \\frac{\\sum_{t} r_t I_t}{n \\alpha (-\\overline{\\text{ES}})} + 1

    where :math:`I_t = \\mathbf{1}\\{r_t < -\\text{VaR}_t\\}`,
    :math:`\\text{ES}_t < 0`, and :math:`\\overline{\\text{ES}} = \\text{mean}(\\text{ES}_t)`.
    Under H0: both :math:`Z_1 \\approx 0` and :math:`Z_2 \\approx 0`.
    ES underestimation (|ES| too small) gives :math:`Z_1 < 0` and :math:`Z_2 < 0`.

    Acerbi-Székely's third test (Z3) is intentionally **not** reported: it is
    rank/ESF-based and needs the full one-step predictive CDF, whereas this
    function (and the DM/MCS pipeline it feeds) carries only the point
    ``(VaR, ES)`` forecasts. Z1 (conditional) and Z2 (unconditional) are the
    tail-severity statistics computable from a ``(VaR, ES)`` pair alone.

    Parameters
    ----------
    future_returns : np.ndarray
        Realized returns, decimal. Shape ``(T,)``.
    es_forecast : np.ndarray
        ES forecasts, **negative** decimal returns. Shape ``(T,)``.
    var_forecast : np.ndarray
        VaR forecasts, **positive** decimal returns (breach = return < -VaR).
        Shape ``(T,)``.
    alpha : float, default 0.05
        Tail probability used to build breach indicators.
    n_boot : int, default 2000
        Monte-Carlo replications for p-value estimation.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    dict[str, float]
        ``Z1`` : test statistic (≈ 0 under H0; < 0 when ES underestimated).
        ``Z2`` : test statistic (≈ 0 under H0; < 0 when ES underestimated).
        ``p``  : min(p_Z1, p_Z2): joint p-value proxy (one-sided lower tail).
    """
    ret = np.asarray(future_returns, dtype=float).ravel()
    es = np.asarray(es_forecast, dtype=float).ravel()
    var = np.asarray(var_forecast, dtype=float).ravel()
    n = ret.size
    if not (es.shape == var.shape == ret.shape):
        raise ValueError("future_returns, es_forecast, var_forecast must have the same shape")
    if n == 0:
        raise ValueError("inputs are empty")

    # Clip ES away from zero; ES must be negative (left-tail convention).
    es_safe = np.where(es < -1e-300, es, -1e-300)

    breach = ret < -var      # boolean indicator I_t
    n_breach = int(breach.sum())

    # Z1 = 1 - mean(r_t / ES_t | breach)
    # With ES_t < 0 and r_t < 0 on breach days:
    #   Under H0: r_t ≈ ES_t, ratio ≈ 1 → Z1 ≈ 0.
    #   ES underestimated (|ES| too small): |r_t| > |ES_t| → r_t/ES_t > 1 → Z1 < 0.
    # This matches the sign convention of Acerbi-Szekely (2014):
    # their Z1 = mean(r/ES_paper | breach) + 1 with ES_paper > 0; translates
    # to 1 - mean(r/ES_neg | breach) with our negative-ES convention.
    if n_breach == 0:
        Z1 = float("nan")
    else:
        Z1 = 1.0 - float(np.mean(ret[breach] / es_safe[breach]))

    # Z2 = sum(r_t * I_t) / (n * alpha * (-mean(ES))) + 1
    # With ES_t < 0: -mean(ES) > 0, and sum(r_t*I_t) < 0 under H0.
    # Under H0: sum(r_t*I_t) ≈ n*alpha*mean(ES) → ratio → -1 → Z2 ≈ 0.
    # ES underestimated: |sum(r*I)| > n*alpha*|ES| → ratio < -1 → Z2 < 0.
    mean_es = float(np.mean(es_safe))  # negative
    denom_z2 = n * alpha * (-mean_es)   # positive
    if abs(denom_z2) < 1e-300:
        Z2 = float("nan")
    else:
        Z2 = float(np.sum(ret * breach.astype(float))) / denom_z2 + 1.0

    # Monte-Carlo p-values under H0.
    # H0: the forecast (VaR_t, ES_t) is the TRUE (VaR, ES) of r_t.  For a normal
    # assumption, the H0 return distribution has
    #   sigma_null_t = |ES_t| * alpha / phi(Phi^{-1}(alpha))
    # so that E[r | r < -VaR_t] = ES_t exactly.  (``var`` here is the VaR threshold,
    # already a positive return level, not a variance; breach when r < -var.)
    rng = np.random.default_rng(seed)
    sigma_null = np.abs(es_safe) * alpha / float(norm.pdf(norm.ppf(alpha)))

    boot_Z1 = np.empty(n_boot)
    boot_Z2 = np.empty(n_boot)
    for b in range(n_boot):
        r_sim = rng.normal(0.0, 1.0, n) * sigma_null
        breach_sim = r_sim < -var  # var is the VaR threshold (positive)
        n_breach_sim = int(breach_sim.sum())
        if n_breach_sim == 0:
            boot_Z1[b] = 0.0
        else:
            boot_Z1[b] = 1.0 - float(np.mean(r_sim[breach_sim] / es_safe[breach_sim]))
        denom_sim = n * alpha * (-mean_es)
        if abs(denom_sim) < 1e-300:
            boot_Z2[b] = 0.0
        else:
            boot_Z2[b] = float(np.sum(r_sim * breach_sim.astype(float))) / denom_sim + 1.0

    # One-sided p-value: probability that the null statistic is <= observed.
    p_Z1 = float(np.mean(boot_Z1 <= Z1)) if np.isfinite(Z1) else float("nan")
    p_Z2 = float(np.mean(boot_Z2 <= Z2)) if np.isfinite(Z2) else float("nan")
    # Joint p-value: use the minimum (most significant).
    if np.isfinite(p_Z1) and np.isfinite(p_Z2):
        p = min(p_Z1, p_Z2)
    elif np.isfinite(p_Z1):
        p = p_Z1
    elif np.isfinite(p_Z2):
        p = p_Z2
    else:
        p = float("nan")

    return {"Z1": Z1, "Z2": Z2, "p": p}


# ---------------------------------------------------------------------------
# 7. Fissler-Ziegel (FZ0) loss
# ---------------------------------------------------------------------------

def fz_loss(
    future_returns: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Fissler-Ziegel FZ0 per-origin loss for joint (VaR, ES) forecasts.

    FZ0 is a strictly consistent scoring rule for the joint (VaR, ES) pair
    (Fissler & Ziegel 2016; Taylor 2019).  It enables DM/MCS comparison of
    tail models, exactly analogous to QLIKE for variance models, but in the
    risk layer.

    The formula (Patton, Ziegel & Chen 2019 eq. 2 / Taylor 2019; written for
    left-tail losses with VaR > 0, ES < 0, substituting the quantile ``q=-VaR``)::

        FZ0_t = (alpha - I_t) / alpha * VaR_t
                - 1 / alpha
                + I_t * r_t / (alpha * ES_t)
                + log(-ES_t)

    where ``I_t = 1{r_t < -VaR_t}`` and ``ES_t < 0`` (so on a breach the term
    ``I_t * r_t / (alpha * ES_t) > 0``).  The ``-1/alpha`` constant is identical
    for every model on the same series and does not affect comparisons.  FZ0 is
    minimised (in expectation) by the true (VaR, ES) pair.

    **Invariant note:** FZ0 is a risk-layer loss; never place FZ0 values in
    the same DM/MCS table as Track-1 QLIKE (Invariant 4).

    Parameters
    ----------
    future_returns : np.ndarray
        Realized returns, decimal. Shape ``(T,)``.
    var_forecast : np.ndarray
        VaR forecasts, **positive** decimal returns. Shape ``(T,)``.
    es_forecast : np.ndarray
        ES forecasts, **negative** decimal returns. Shape ``(T,)``.
    alpha : float, default 0.05
        Tail probability (5 % left tail).

    Returns
    -------
    np.ndarray
        Per-origin FZ0 loss values. Shape ``(T,)``. Units: dimensionless
        (log-scale loss; comparable across models on the same return series).
    """
    ret = np.asarray(future_returns, dtype=float).ravel()
    var = np.asarray(var_forecast, dtype=float).ravel()
    es = np.asarray(es_forecast, dtype=float).ravel()
    if not (var.shape == es.shape == ret.shape):
        raise ValueError("future_returns, var_forecast, es_forecast must have the same shape")
    if ret.size == 0:
        raise ValueError("inputs are empty")

    # Guard: ES must be strictly negative for log(-ES) to be defined.
    # Values near zero indicate a degenerate forecast; clip to -1e-300.
    es_safe = np.where(es < -1e-300, es, -1e-300)

    I_t = (ret < -var).astype(float)  # breach indicator

    # FZ0 formula (Patton, Ziegel & Chen 2019 eq. 2; Fissler & Ziegel 2016 with G(e)=-1/e):
    # With q = -VaR < 0 and e = ES < 0:
    #   FZ0(r; q, e) = (I-alpha)/alpha * q - 1/alpha + I*r/(alpha*e) + log(-e)
    # Substituting q = -var (var > 0):
    #   FZ0 = (I-alpha)/alpha * (-var) - 1/alpha + I*r/(alpha*e) + log(-e)
    #       = (alpha-I)/alpha * var - 1/alpha + I*r/(alpha*e) + log(-e)
    # The -1/alpha constant does not affect model comparisons (same for all models
    # on the same return series), but is included for correctness.
    loss = (
        (alpha - I_t) / alpha * var
        - 1.0 / alpha
        + I_t * ret / (alpha * es_safe)
        + np.log(-es_safe)
    )
    return loss
