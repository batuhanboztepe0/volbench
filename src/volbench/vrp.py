"""Variance Risk Premium (VRP) analysis and strategy.

The equity variance risk premium is implied variance minus expected (forecast)
realized variance. It is positive on average — sellers of variance earn it
roughly 90 % of trading days — meaning the options market consistently
over-prices realized variance. A good RV forecaster (log-HAR) allows timing:
scale up the short-variance position when implied looks rich relative to the
model forecast, and scale down (or go long) when it looks cheap.

Two public-facing functions are provided:

* :func:`variance_risk_premium` — the ex-ante VRP signal (implied - forecast).
* :func:`vrp_strategy` — a short-variance timing backtest comparing an
  always-short book, a forecast-timed book, and a long/short book.
"""

from __future__ import annotations

import numpy as np

from .deflated_sharpe import (
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)
from .evaluation import diebold_mariano


def variance_risk_premium(
    implied_var: np.ndarray,
    forecast_var: np.ndarray,
) -> np.ndarray:
    """Ex-ante variance risk premium signal.

    Parameters
    ----------
    implied_var : np.ndarray
        Implied daily variance at each origin (e.g. ``(VIX/100)**2/252``).
    forecast_var : np.ndarray
        Model forecast of average realized daily variance over the horizon.

    Returns
    -------
    np.ndarray
        ``implied_var - forecast_var`` element-wise.
    """
    iv = np.asarray(implied_var, dtype=float).ravel()
    fv = np.asarray(forecast_var, dtype=float).ravel()
    if iv.shape != fv.shape:
        raise ValueError(
            f"implied_var and forecast_var must match in length, "
            f"got {iv.shape} and {fv.shape}"
        )
    return iv - fv


def _max_drawdown(cum_pnl: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a cumulative P&L series.

    Parameters
    ----------
    cum_pnl : np.ndarray
        Cumulative profit-and-loss series.

    Returns
    -------
    float
        Maximum drawdown (a non-negative number).
    """
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    return float(drawdowns.max())


def _nonoverlapping(pnl: np.ndarray, horizon: int) -> np.ndarray:
    """Non-overlapping h-day payoffs (every ``horizon``-th observation).

    The daily VRP P&L overlaps because consecutive positions settle over windows
    that share ``horizon - 1`` days; subsampling every ``horizon``-th payoff yields
    an (approximately) independent sample for honest significance testing.
    """
    step = max(int(horizon), 1)
    return np.asarray(pnl, dtype=float).ravel()[::step]


def _book_stats(
    pnl: np.ndarray,
    horizon: int,
    sr_star: float = 0.0,
) -> dict[str, float]:
    """Compute per-book summary statistics.

    Parameters
    ----------
    pnl : np.ndarray
        Per-period P&L series.
    horizon : int
        Forecast horizon in days, used to annualise the Sharpe ratio and to set the
        effective sample size for the deflated metrics (the payoffs overlap over
        ``horizon`` days, so the independent sample is ``n / horizon``).
    sr_star : float, default 0.0
        Per-period selection benchmark for the deflated Sharpe (the expected best
        Sharpe across the books tried). 0.0 gives the undeflated PSR.

    Returns
    -------
    dict[str, float]
        Keys: ``ann_sharpe``, ``hit_rate``, ``mean_pnl``, ``total_pnl``,
        ``max_drawdown``, ``psr`` (P[true Sharpe > 0]), ``dsr`` (selection-deflated),
        ``sharpe_pp`` (per-period Sharpe).
    """
    p = np.asarray(pnl, dtype=float)
    mu = float(p.mean())
    sigma = float(p.std(ddof=1)) if p.size > 1 else float("nan")
    ann_sharpe = (mu / sigma * np.sqrt(252.0 / horizon)) if sigma > 0 else float("nan")
    hit_rate = float((p > 0.0).mean())
    cum = np.cumsum(p)
    # The daily P&L series overlaps over `horizon` days (consecutive payoffs settle
    # over near-identical windows), which both inflates the per-period Sharpe and
    # over-counts the sample. For the deflated metrics we therefore use the
    # *non-overlapping* h-day payoffs p[::horizon] — a smaller, honest sample whose
    # Sharpe is not overlap-inflated. (This is why the headline annualised Sharpe
    # above is inflated relative to these probabilities.)
    sub = _nonoverlapping(p, horizon)
    return {
        "ann_sharpe": ann_sharpe,
        "hit_rate": hit_rate,
        "mean_pnl": mu,
        "total_pnl": float(p.sum()),
        "max_drawdown": _max_drawdown(cum),
        "sharpe_pp": per_period_sharpe(sub),
        "psr": probabilistic_sharpe_ratio(sub, benchmark_sr=0.0),
        "dsr": probabilistic_sharpe_ratio(sub, benchmark_sr=sr_star),
    }


def vrp_strategy(
    implied_var: np.ndarray,
    forecast_var: np.ndarray,
    realized_future_var: np.ndarray,
    horizon: int = 1,
    scale: float = 1.0,
    costs: float = 0.0,
) -> dict[str, object]:
    """Short-variance timing strategy backtest.

    Evaluates three trading books:

    * **always_short** — unit short-variance position every period.
    * **timed** — position scaled by the relative richness of implied variance
      vs the model forecast: ``clip(scale*(IV-FC)/IV, -1, 2)``.
    * **longshort** — same timed position but without the ``-1`` floor, capped
      at ``(-2, 2)`` (allows a more aggressive long when implied looks cheap).

    NO look-ahead: positions use only ``implied_var[t]`` and ``forecast_var[t]``
    (both known at time *t*); payoffs use ``realized_future_var[t]``, which is
    the realized variance over the *future* window.

    Parameters
    ----------
    implied_var : np.ndarray
        Implied daily variance at each origin.
    forecast_var : np.ndarray
        Model forecast of average realized daily variance over the horizon.
    realized_future_var : np.ndarray
        Realized average daily variance over the horizon (the ex-post outcome).
    horizon : int, default 1
        Forecast horizon in days (used for Sharpe annualisation and DM test).
    scale : float, default 1.0
        Multiplicative scaling for the timing signal before clipping.
    costs : float, default 0.0
        Round-trip proportional transaction cost applied to the absolute change
        in position: ``pnl -= costs * |pos[t] - pos[t-1]|``.

    Returns
    -------
    dict[str, object]
        Keys: ``always_short``, ``timed``, ``longshort`` (each a dict of book
        statistics — see :func:`_book_stats`), and ``dm_timed_vs_always_short``
        (a :func:`~volbench.evaluation.diebold_mariano` result comparing the
        timed book against the always-short book using negative P&L as the
        "loss").
    """
    iv = np.asarray(implied_var, dtype=float).ravel()
    fv = np.asarray(forecast_var, dtype=float).ravel()
    rv = np.asarray(realized_future_var, dtype=float).ravel()

    n = iv.size
    if fv.shape != iv.shape or rv.shape != iv.shape:
        raise ValueError("all input arrays must have the same length")

    # Payoff to a unit short-variance position: collect premium, pay realised.
    raw_payoff = iv - rv  # shape (n,)

    # --- Always-short book (position = 1 every period) -----------------------
    pos_always = np.ones(n)

    # --- Timed book: position proportional to richness, clipped to [-1, 2] ---
    iv_safe = np.where(iv > 0.0, iv, np.finfo(float).tiny)
    signal = scale * (iv - fv) / iv_safe
    pos_timed = np.clip(signal, -1.0, 2.0)

    # --- Long/short book: wider clip to [-2, 2] ------------------------------
    pos_longshort = np.clip(signal, -2.0, 2.0)

    def _pnl(pos: np.ndarray) -> np.ndarray:
        """Compute P&L with transaction costs applied to position changes."""
        p = pos * raw_payoff
        if costs > 0.0:
            pos_prev = np.concatenate([[0.0], pos[:-1]])
            p = p - costs * np.abs(pos - pos_prev)
        return p

    pnl_always = _pnl(pos_always)
    pnl_timed = _pnl(pos_timed)
    pnl_longshort = _pnl(pos_longshort)

    # Deflated-Sharpe selection benchmark: three book configurations were tried, so
    # the reported Sharpe must clear the expected best-of-3 under the null. Use the
    # same non-overlapping h-day payoffs as the PSR/DSR so the units match.
    sr_star = expected_max_sharpe([
        per_period_sharpe(_nonoverlapping(pnl_always, horizon)),
        per_period_sharpe(_nonoverlapping(pnl_timed, horizon)),
        per_period_sharpe(_nonoverlapping(pnl_longshort, horizon)),
    ])

    # DM test: use negative P&L as the "loss" (lower is better).
    # Tests whether timed beats always-short.
    dm = diebold_mariano(-pnl_timed, -pnl_always, horizon=horizon)

    return {
        "always_short": _book_stats(pnl_always, horizon, sr_star),
        "timed": _book_stats(pnl_timed, horizon, sr_star),
        "longshort": _book_stats(pnl_longshort, horizon, sr_star),
        "dm_timed_vs_always_short": dm,
    }
