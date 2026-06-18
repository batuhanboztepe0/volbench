"""Variance Risk Premium study on S&P 500 (Track D2).

Uses the log-HAR forecast (horizon=22) together with VIX-implied variance to
measure the ex-ante VRP and evaluate whether timing the short-variance trade
beats an always-short position.

Results are written to results/vrp.json and results/figures/vrp.png.

Usage
-----
    PYTHONPATH=src python3 scripts/run_vrp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import TRADING_DAYS, load_oxford_rv, load_vix  # noqa: E402
from volbench.models import HAR, LogHAR  # noqa: E402
from volbench.vrp import vrp_strategy  # noqa: E402

TICKER: str = ".SPX"
HORIZON: int = 22
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _jsonify(obj: object) -> object:
    """Recursively convert numpy scalars to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def run_vrp() -> dict:
    """Execute the VRP analysis and write results.

    Returns
    -------
    dict
        Summary of the VRP analysis, JSON-serialisable.
    """
    # ------------------------------------------------------------------
    # 1. Load data and run the log-HAR backtest
    # ------------------------------------------------------------------
    ds = load_oxford_rv(tickers=[TICKER])
    rv = ds.series(TICKER)
    fr = ds.frame(TICKER)

    res = run_backtest(
        rv,
        horizon=HORIZON,
        models=[HAR(), LogHAR()],
        mcs_reps=100,
        seed=0,
        benchmark="HAR",
    )

    forecast = res.forecasts["LogHAR"]       # shape (n_origins,)
    realized_future = res.realized            # shape (n_origins,)
    origins = res.origins                     # positional integer indices

    # ------------------------------------------------------------------
    # 2. Map origins to calendar dates and align VIX
    # ------------------------------------------------------------------
    rv_dates = fr.index                       # DatetimeIndex length = len(rv)
    origin_dates = rv_dates[origins]          # dates at each test origin

    vix_series = load_vix()                   # daily VIX in annualised %

    # Reindex VIX to the origin dates; forward-fill within 5 days, then drop.
    vix_aligned = vix_series.reindex(
        origin_dates, method="nearest", tolerance=pd.Timedelta("5D")
    )

    keep_mask = vix_aligned.notna().values
    vix_vals = vix_aligned.values[keep_mask]

    forecast_k = forecast[keep_mask]
    realized_k = realized_future[keep_mask]
    dates_k = origin_dates[keep_mask]

    n_kept = keep_mask.sum()
    print(f"  Origins: {origins.size}  |  With VIX: {n_kept}")

    # Implied daily variance from VIX: (VIX/100)^2 / TRADING_DAYS
    implied_var = (vix_vals / 100.0) ** 2 / TRADING_DAYS

    # ------------------------------------------------------------------
    # 3. Compute ex-ante VRP and summary statistics
    # ------------------------------------------------------------------
    vrp_realized = implied_var - realized_k   # ex-post VRP

    mean_implied_vol = float(np.sqrt(np.mean(implied_var) * TRADING_DAYS) * 100)
    mean_realized_vol = float(np.sqrt(np.mean(realized_k) * TRADING_DAYS) * 100)
    mean_vrp = float(np.mean(vrp_realized))
    vrp_positive_fraction = float(np.mean(vrp_realized > 0))

    print(f"\n  Mean implied vol  : {mean_implied_vol:.2f} %")
    print(f"  Mean realized vol : {mean_realized_vol:.2f} %")
    print(f"  Mean VRP (daily)  : {mean_vrp:.2e}")
    print(f"  VRP > 0 fraction  : {vrp_positive_fraction:.3f}")

    # ------------------------------------------------------------------
    # 4. Run the three trading books
    # ------------------------------------------------------------------
    strat = vrp_strategy(implied_var, forecast_k, realized_k, horizon=HORIZON)

    print(f"\n  {'Book':<15} {'AnnSh':>7} {'Sh/swap':>8} {'HitRate':>9} {'MaxDD':>10} {'PSR':>8} {'DSR':>8}")
    print(f"  {'-' * 66}")
    for book in ("always_short", "timed", "longshort"):
        b = strat[book]
        print(f"  {book:<15} {b['ann_sharpe']:>7.3f} {b['sharpe_pp']:>8.3f} {b['hit_rate']:>9.3f} "
              f"{b['max_drawdown']:>10.4e} {b['psr']:>8.4f} {b['dsr']:>8.4f}")
    print("  (AnnSh is gross of costs on overlapping payoffs and inflated; Sh/swap, PSR and DSR "
          "use the\n   non-overlapping 22-day payoffs and a best-of-3-books selection benchmark — "
          "the honest figures.)")

    # NOTE: on this dataset the "timed" and "longshort" books are identical. The
    # raw signal (iv - forecast)/iv never falls below -1 (the LogHAR forecast never
    # exceeds ~2x implied variance), so the wider long/short clip [-2, 2] is never
    # exercised and collapses onto the timed clip [-1, 2]. They are reported
    # separately only to document this degeneracy, not as two distinct strategies.
    if np.allclose(strat["timed"]["ann_sharpe"], strat["longshort"]["ann_sharpe"]):
        print("  (timed == longshort here: signal never breaches -1, so the "
              "long/short bound is inactive)")

    dm = strat["dm_timed_vs_always_short"]
    print("\n  DM timed vs always-short:")
    print(f"    mean_diff={dm['mean_diff']:.4e}  stat={dm['dm_stat']:.3f}  "
          f"p={dm['p_value']:.4f}  favored={'timed' if dm['favored'] < 0 else 'always_short'}")

    # ------------------------------------------------------------------
    # 5. Save figure: cumulative P&L of the three books
    # ------------------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Re-compute cumulative P&L for each book so we can plot by date.
    iv = implied_var
    rv_f = realized_k
    fv = forecast_k

    raw_payoff = iv - rv_f
    iv_safe = np.where(iv > 0.0, iv, np.finfo(float).tiny)
    signal = np.clip((iv - fv) / iv_safe, -1.0, 2.0)

    cum_always = np.cumsum(raw_payoff)
    cum_timed = np.cumsum(signal * raw_payoff)
    cum_longshort = np.cumsum(np.clip((iv - fv) / iv_safe, -2.0, 2.0) * raw_payoff)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates_k, cum_always, label="always-short", linewidth=1.0)
    ax.plot(dates_k, cum_timed, label="timed (LogHAR)", linewidth=1.0)
    ax.plot(dates_k, cum_longshort, label="long/short (LogHAR)", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Variance Risk Premium strategy — cumulative P&L (horizon=22d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (daily variance units)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES_DIR / "vrp.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\n  Saved figure to {fig_path}")

    # ------------------------------------------------------------------
    # 6. Write JSON results
    # ------------------------------------------------------------------
    out = {
        "ticker": TICKER,
        "horizon": HORIZON,
        "n_origins": int(n_kept),
        "mean_implied_vol_pct": mean_implied_vol,
        "mean_realized_vol_pct": mean_realized_vol,
        "mean_vrp_daily": mean_vrp,
        "vrp_positive_fraction": vrp_positive_fraction,
        "books": {
            book: strat[book]
            for book in ("always_short", "timed", "longshort")
        },
        "dm_timed_vs_always_short": strat["dm_timed_vs_always_short"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "vrp.json"
    with open(out_path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=2)
    print(f"  Saved results to {out_path}")
    return out


if __name__ == "__main__":
    run_vrp()
