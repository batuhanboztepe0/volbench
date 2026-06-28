"""Vol-targeting strategy with transaction costs and a jump/regime de-risking overlay.

Three public functions:

* :func:`vol_target_backtest`: leakage-free vol-targeting with transaction costs.
* :func:`regime_overlay`: scale weights down in turbulent or jump-heavy regimes
  using only backward-looking quantiles.
* :func:`compare_books`: convenience wrapper comparing buy-and-hold, vol-target,
  and vol-target + regime overlay for a single return/forecast series.
"""

from __future__ import annotations

import numpy as np

from .deflated_sharpe import per_period_sharpe, probabilistic_sharpe_ratio

# ---------------------------------------------------------------------------
# Internal helpers (same pattern as economic.py)
# ---------------------------------------------------------------------------

def _equity_curve(strategy_returns: np.ndarray) -> np.ndarray:
    """Compounded equity curve; the first value is ``1 + r[0]``, not exactly 1."""
    return np.cumprod(1.0 + strategy_returns)


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown on a compounded equity curve (<=0)."""
    if equity.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def _book_stats(
    strat_ret: np.ndarray,
    gross_weights: np.ndarray,
    net_weights: np.ndarray,
    future_returns: np.ndarray,
    cost_per_turnover: float,
    trading_days: int,
) -> dict[str, float]:
    """Compute the full set of per-book statistics.

    Parameters
    ----------
    strat_ret : np.ndarray
        Strategy returns *after* costs.
    gross_weights : np.ndarray
        Position weights before costs (used for gross sharpe).
    net_weights : np.ndarray
        Same weights (used for turnover / avg leverage reporting).
    future_returns : np.ndarray
        Buy-and-hold returns for reference.
    cost_per_turnover : float
        Cost per unit of weight change.
    trading_days : int
        Annualisation factor.

    Returns
    -------
    dict[str, float]
    """
    # Gross return series (no cost deduction, same weights)
    if gross_weights.size > 1:
        gross_cost = cost_per_turnover * np.abs(np.diff(gross_weights, prepend=0.0))
    else:
        gross_cost = np.zeros_like(gross_weights)
    gross_ret = gross_weights * future_returns - gross_cost + gross_cost  # identical to gross_weights * ret
    gross_ret = gross_weights * future_returns  # no-cost version

    ann_gross = float(np.mean(gross_ret) * trading_days)
    std_gross = float(np.std(gross_ret, ddof=1) * np.sqrt(trading_days))
    gross_sharpe = ann_gross / std_gross if std_gross > 0 else float("nan")

    ann_net = float(np.mean(strat_ret) * trading_days)
    std_net = float(np.std(strat_ret, ddof=1) * np.sqrt(trading_days))
    net_sharpe = ann_net / std_net if std_net > 0 else float("nan")

    equity = _equity_curve(strat_ret)
    mdd = _max_drawdown(equity)

    turnover = (
        float(np.mean(np.abs(np.diff(net_weights, prepend=0.0))))
        if net_weights.size >= 1
        else 0.0
    )
    avg_leverage = float(np.mean(net_weights))

    # Buy-and-hold reference
    bh_ann_vol = float(np.std(future_returns, ddof=1) * np.sqrt(trading_days))
    bh_ann_ret = float(np.mean(future_returns) * trading_days)
    bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else float("nan")
    bh_equity = _equity_curve(future_returns)
    bh_mdd = _max_drawdown(bh_equity)

    return {
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "ann_return": ann_net,
        "ann_vol": std_net,
        "max_drawdown": mdd,
        "turnover": turnover,
        "avg_leverage": avg_leverage,
        "bh_sharpe": bh_sharpe,
        "bh_ann_vol": bh_ann_vol,
        "bh_max_drawdown": bh_mdd,
        # Honest Sharpe: per-period Sharpe and the probability the *true* Sharpe is
        # > 0 once skew, fat tails and sample length are accounted for (daily,
        # non-overlapping returns, so n_eff = n).
        "sharpe_pp": per_period_sharpe(strat_ret),
        "psr": probabilistic_sharpe_ratio(strat_ret),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vol_target_backtest(
    future_returns: np.ndarray,
    forecast_variance: np.ndarray,
    target_ann_vol: float = 0.10,
    max_leverage: float = 3.0,
    cost_per_turnover: float = 0.0005,
    trading_days: int = 252,
) -> dict[str, float]:
    """Vol-targeting backtest with transaction costs.

    At each origin the weight is::

        w_t = clip(target_daily_vol / sqrt(forecast_variance[t]), 0, max_leverage)

    The net return is ``w_t * future_returns[t] - cost_per_turnover * |w_t - w_{t-1}|``
    with ``w_{-1} = 0``.

    Parameters
    ----------
    future_returns : np.ndarray
        Realized next-period returns aligned to forecast origins. Shape ``(T,)``.
    forecast_variance : np.ndarray
        Variance forecasts aligned to the same origins. Shape ``(T,)``, daily
        decimal variance.
    target_ann_vol : float, default 0.10
        Annualised volatility target (10 % = 0.10).
    max_leverage : float, default 3.0
        Upper bound on the position weight; lower bound is 0.0.
    cost_per_turnover : float, default 0.0005
        Round-trip cost per unit of absolute weight change.
    trading_days : int, default 252
        Annualisation convention.

    Returns
    -------
    dict[str, float]
        Keys: ``gross_sharpe``, ``net_sharpe``, ``ann_return``, ``ann_vol``,
        ``max_drawdown``, ``turnover``, ``avg_leverage``,
        ``bh_sharpe``, ``bh_ann_vol``, ``bh_max_drawdown``.
    """
    ret = np.asarray(future_returns, dtype=float).ravel()
    fvar = np.asarray(forecast_variance, dtype=float).ravel()
    if ret.shape != fvar.shape:
        raise ValueError(
            f"future_returns and forecast_variance must match in length, "
            f"got {ret.shape} vs {fvar.shape}"
        )
    if ret.size == 0:
        raise ValueError("inputs are empty")

    target_daily_vol = target_ann_vol / np.sqrt(trading_days)
    weights = np.clip(
        target_daily_vol / np.sqrt(np.maximum(fvar, 1e-300)),
        0.0,
        max_leverage,
    )

    # Costs based on weight changes; w_{-1} = 0 so first trade is a full buy-in.
    prev_weights = np.concatenate([[0.0], weights[:-1]])
    costs = cost_per_turnover * np.abs(weights - prev_weights)
    strat_ret = weights * ret - costs

    return _book_stats(strat_ret, weights, weights, ret, cost_per_turnover, trading_days)


def regime_overlay(
    weights: np.ndarray,
    rv_at_origin: np.ndarray,
    jump_at_origin: np.ndarray,
    rv_window: int = 63,
    vol_quantile: float = 0.8,
    jump_quantile: float = 0.9,
) -> np.ndarray:
    """De-risk weights in turbulent or jump-heavy regimes (no look-ahead).

    A weight at origin ``t`` is scaled by 0.5 when either:

    * ``rv_at_origin[t]`` exceeds the trailing ``rv_window``-day rolling
      empirical quantile ``vol_quantile`` of ``rv_at_origin[:t]`` (turbulent
      regime), **or**
    * ``jump_at_origin[t]`` exceeds the expanding quantile ``jump_quantile``
      of ``jump_at_origin[:t]`` (jump day).

    Both quantiles use only *past* data to ensure leakage-freedom.

    Parameters
    ----------
    weights : np.ndarray
        Position weights to adjust. Shape ``(T,)``.
    rv_at_origin : np.ndarray
        Realized variance at each origin. Shape ``(T,)``.
    jump_at_origin : np.ndarray
        Jump variation at each origin. Shape ``(T,)``.
    rv_window : int, default 63
        Look-back window (in number of origins) for the rolling RV quantile.
    vol_quantile : float, default 0.8
        Quantile threshold for declaring a turbulent regime.
    jump_quantile : float, default 0.9
        Expanding quantile threshold for declaring a jump day.

    Returns
    -------
    np.ndarray
        Adjusted weights of the same shape.
    """
    w = np.asarray(weights, dtype=float).ravel().copy()
    rv = np.asarray(rv_at_origin, dtype=float).ravel()
    jmp = np.asarray(jump_at_origin, dtype=float).ravel()
    n = w.size

    for t in range(n):
        # ---- Turbulent-regime check (trailing rv_window rows before t) ----
        start = max(0, t - rv_window)
        past_rv = rv[start:t]  # excludes t itself
        turbulent = False
        if past_rv.size > 0:
            thresh_rv = np.quantile(past_rv, vol_quantile)
            turbulent = bool(rv[t] > thresh_rv)

        # ---- Jump-day check (expanding window before t) ----
        past_jmp = jmp[:t]  # excludes t itself
        jump_day = False
        if past_jmp.size > 0:
            thresh_jmp = np.quantile(past_jmp, jump_quantile)
            jump_day = bool(jmp[t] > thresh_jmp)

        if turbulent or jump_day:
            w[t] *= 0.5

    return w


def compare_books(
    future_returns: np.ndarray,
    forecast_variance: np.ndarray,
    rv_at_origin: np.ndarray,
    jump_at_origin: np.ndarray,
    target_ann_vol: float = 0.10,
    max_leverage: float = 3.0,
    cost_per_turnover: float = 0.0005,
    trading_days: int = 252,
    rv_window: int = 63,
    vol_quantile: float = 0.8,
    jump_quantile: float = 0.9,
) -> dict[str, dict[str, float]]:
    """Compare buy-and-hold, vol-target, and vol-target + regime overlay.

    Runs :func:`vol_target_backtest` and :func:`regime_overlay` on the same
    aligned inputs and returns a three-way comparison dict.

    Parameters
    ----------
    future_returns : np.ndarray
        Realized next-period returns aligned to forecast origins.
    forecast_variance : np.ndarray
        Variance forecasts aligned to ``future_returns``.
    rv_at_origin : np.ndarray
        Realized variance at each origin (for the regime filter).
    jump_at_origin : np.ndarray
        Jump variation at each origin (for the jump filter).
    target_ann_vol : float, default 0.10
        Vol-targeting annual volatility target.
    max_leverage : float, default 3.0
        Leverage cap.
    cost_per_turnover : float, default 0.0005
        Round-trip cost per unit weight change.
    trading_days : int, default 252
        Annualisation convention.
    rv_window : int, default 63
        Rolling window for RV quantile in :func:`regime_overlay`.
    vol_quantile : float, default 0.8
        RV percentile threshold.
    jump_quantile : float, default 0.9
        Jump percentile threshold.

    Returns
    -------
    dict[str, dict[str, float]]
        Keys ``"buy_hold"``, ``"vol_target"``, ``"vol_target_plus_overlay"``.
        Each value is the stats dict produced by :func:`vol_target_backtest`
        (or equivalent stats for buy-and-hold).
    """
    ret = np.asarray(future_returns, dtype=float).ravel()
    fvar = np.asarray(forecast_variance, dtype=float).ravel()

    # ---- Vol-target book ----
    vt = vol_target_backtest(
        ret, fvar,
        target_ann_vol=target_ann_vol,
        max_leverage=max_leverage,
        cost_per_turnover=cost_per_turnover,
        trading_days=trading_days,
    )

    # ---- Base weights (no costs yet, needed for overlay) ----
    target_daily_vol = target_ann_vol / np.sqrt(trading_days)
    base_weights = np.clip(
        target_daily_vol / np.sqrt(np.maximum(fvar, 1e-300)),
        0.0,
        max_leverage,
    )

    # ---- Overlay book: apply regime filter, then recompute costs/returns ----
    overlay_weights = regime_overlay(
        base_weights,
        np.asarray(rv_at_origin, dtype=float).ravel(),
        np.asarray(jump_at_origin, dtype=float).ravel(),
        rv_window=rv_window,
        vol_quantile=vol_quantile,
        jump_quantile=jump_quantile,
    )
    prev_ow = np.concatenate([[0.0], overlay_weights[:-1]])
    overlay_costs = cost_per_turnover * np.abs(overlay_weights - prev_ow)
    overlay_ret = overlay_weights * ret - overlay_costs
    overlay_stats = _book_stats(
        overlay_ret, overlay_weights, overlay_weights, ret, cost_per_turnover, trading_days
    )

    # ---- Buy-and-hold book ----
    bh_ann_vol_val = float(np.std(ret, ddof=1) * np.sqrt(trading_days))
    bh_ann_ret_val = float(np.mean(ret) * trading_days)
    bh_sharpe_val = bh_ann_ret_val / bh_ann_vol_val if bh_ann_vol_val > 0 else float("nan")
    bh_equity = _equity_curve(ret)
    bh_mdd = _max_drawdown(bh_equity)

    bh_stats: dict[str, float] = {
        "gross_sharpe": bh_sharpe_val,
        "net_sharpe": bh_sharpe_val,
        "ann_return": bh_ann_ret_val,
        "ann_vol": bh_ann_vol_val,
        "max_drawdown": bh_mdd,
        "turnover": 0.0,
        "avg_leverage": 1.0,
        "bh_sharpe": bh_sharpe_val,
        "bh_ann_vol": bh_ann_vol_val,
        "bh_max_drawdown": bh_mdd,
        "sharpe_pp": per_period_sharpe(ret),
        "psr": probabilistic_sharpe_ratio(ret),
    }

    return {
        "buy_hold": bh_stats,
        "vol_target": vt,
        "vol_target_plus_overlay": overlay_stats,
    }
