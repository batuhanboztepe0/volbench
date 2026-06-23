"""Vol-targeting strategy benchmark across eight indices.

For each index:
- Runs a horizon-1 LogHAR backtest to get forecast variance and origins.
- Computes the aligned next-day close-to-close log returns.
- Calls compare_books to evaluate buy-and-hold vs vol-target vs overlay.

Additionally compares LogHAR / RW / EWMA as forecast drivers for the vol-target
book to test whether the better forecaster yields a better risk-adjusted Sharpe.

Writes results/strategy.json and results/figures/strategy.png.

Usage
-----
    PYTHONPATH=src python3 scripts/run_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import load_oxford_rv  # noqa: E402
from volbench.models import EWMA, LogHAR, RandomWalk  # noqa: E402
from volbench.strategy import compare_books, vol_target_backtest  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_inputs(ds, tk: str):
    """Run a single horizon-1 backtest for LogHAR / RW / EWMA and return aligned data.

    Returns
    -------
    tuple or None
        ``(future_returns, forecasts_dict, rv_origins, jump_origins, origins)``
        where ``forecasts_dict`` maps model name -> np.ndarray.
        Returns ``None`` if the series is too short.
    """
    fr = ds.frame(tk)
    rv = ds.series(tk, "rv5")
    close = fr["close_price"].to_numpy(dtype=float)
    jump = fr["jump"].to_numpy(dtype=float)

    res = run_backtest(
        rv, horizon=1,
        models=[LogHAR(), RandomWalk(), EWMA()],
        mcs_reps=1,
        benchmark="LogHAR",
    )
    origins = res.origins  # positional int indices into rv

    # Next-day log return: log(close[t+1] / close[t])
    # Drop origins where t+1 is out of range.
    n = close.size
    valid_mask = (origins + 1) < n
    origins = origins[valid_mask]

    if origins.size == 0:
        return None

    future_returns = np.log(close[origins + 1] / close[origins])
    forecasts = {
        name: res.forecasts[name][valid_mask] for name in ["LogHAR", "RW", "EWMA"]
    }
    rv_origins = rv[origins]
    jump_origins = jump[origins]

    return future_returns, forecasts, rv_origins, jump_origins, origins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_strategy() -> dict:
    """Execute the strategy comparison across all indices and write outputs."""
    ds = load_oxford_rv()
    tickers = ds.tickers

    all_books: dict[str, list[dict]] = {
        "buy_hold": [],
        "vol_target": [],
        "vol_target_plus_overlay": [],
    }
    model_comp: dict[str, list[float]] = {
        "LogHAR": [],
        "RW": [],
        "EWMA": [],
    }

    per_ticker: dict[str, dict] = {}
    spx_equity: dict[str, np.ndarray] | None = None
    spx_dates = None

    for tk in tickers:
        result = _all_inputs(ds, tk)
        if result is None:
            print(f"  {tk}: skipped (too short)")
            continue
        future_returns, forecasts, rv_origins, jump_origins, origins = result
        forecast_var = forecasts["LogHAR"]

        books = compare_books(
            future_returns, forecast_var, rv_origins, jump_origins,
        )
        per_ticker[tk] = books

        for book_name in all_books:
            all_books[book_name].append(books[book_name])

        # Forecaster comparison (net Sharpe per driver model) — same origin set
        for mname in ["LogHAR", "RW", "EWMA"]:
            vt = vol_target_backtest(future_returns, forecasts[mname])
            model_comp[mname].append(vt["net_sharpe"])

        # Save .SPX equity curves for the figure
        if tk == ".SPX":
            # Map positional origins to calendar dates for the x-axis.
            spx_dates = ds.frame(tk).index[origins]

            bh_eq = np.cumprod(1.0 + future_returns)
            tdv = 0.10 / np.sqrt(252)
            w_vt = np.clip(tdv / np.sqrt(np.maximum(forecast_var, 1e-300)), 0.0, 3.0)
            prev = np.concatenate([[0.0], w_vt[:-1]])
            costs_vt = 0.0005 * np.abs(w_vt - prev)
            eq_vt = np.cumprod(1.0 + w_vt * future_returns - costs_vt)

            from volbench.strategy import regime_overlay
            w_ov = regime_overlay(w_vt, rv_origins, jump_origins)
            prev_ov = np.concatenate([[0.0], w_ov[:-1]])
            costs_ov = 0.0005 * np.abs(w_ov - prev_ov)
            eq_ov = np.cumprod(1.0 + w_ov * future_returns - costs_ov)

            spx_equity = {
                "buy_hold": bh_eq,
                "vol_target": eq_vt,
                "vol_target_plus_overlay": eq_ov,
            }

        print(
            f"  {tk:<12}  "
            f"BH={books['buy_hold']['net_sharpe']:+.3f}  "
            f"VT={books['vol_target']['net_sharpe']:+.3f}  "
            f"OL={books['vol_target_plus_overlay']['net_sharpe']:+.3f}  "
            f"MDD_VT={books['vol_target']['max_drawdown']:.3f}  "
            f"MDD_OL={books['vol_target_plus_overlay']['max_drawdown']:.3f}"
        )

    # ---------- Aggregate summary ----------
    def _mean_stat(book_list: list[dict], key: str) -> float:
        vals = [b[key] for b in book_list if np.isfinite(b[key])]
        return float(np.mean(vals)) if vals else float("nan")

    agg: dict[str, dict[str, float]] = {}
    for book_name, book_list in all_books.items():
        agg[book_name] = {
            "mean_net_sharpe": _mean_stat(book_list, "net_sharpe"),
            "mean_max_drawdown": _mean_stat(book_list, "max_drawdown"),
            "mean_ann_vol": _mean_stat(book_list, "ann_vol"),
            "mean_ann_return": _mean_stat(book_list, "ann_return"),
        }

    model_sharpes = {m: float(np.mean(v)) if v else float("nan")
                    for m, v in model_comp.items()}

    print("\n--- Aggregate across indices ---")
    for book_name, stats in agg.items():
        print(f"  {book_name:<28}  "
              f"Sharpe={stats['mean_net_sharpe']:+.3f}  "
              f"MDD={stats['mean_max_drawdown']:.3f}")

    print("\n--- Forecaster comparison (mean net Sharpe for vol-target) ---")
    for m, s in model_sharpes.items():
        print(f"  {m:<10}  {s:+.3f}")

    # ---------- Figure (.SPX equity curves) ----------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if spx_equity is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = {
            "buy_hold": "Buy & Hold",
            "vol_target": "Vol-Target",
            "vol_target_plus_overlay": "Vol-Target + Overlay",
        }
        x_axis = spx_dates if spx_dates is not None else range(len(next(iter(spx_equity.values()))))
        for key, label in labels.items():
            ax.plot(x_axis, spx_equity[key], label=label)
        ax.set_title(".SPX: Cumulative Equity (net of costs)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (1 = start)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "strategy.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved figure to {FIGURES_DIR / 'strategy.png'}")

    # ---------- JSON output ----------
    output: dict = {
        "aggregate": agg,
        "model_forecast_comparison": model_sharpes,
        "per_ticker": {
            tk: {
                book_name: {k: float(v) for k, v in stats.items()}
                for book_name, stats in books.items()
            }
            for tk, books in per_ticker.items()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "strategy.json", "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Saved results to {RESULTS_DIR / 'strategy.json'}")

    return output


if __name__ == "__main__":
    run_strategy()
