"""Track 1B driver: economic-value evaluation on eight indices.

Runs a small backtest (horizon=1, mcs_reps=200) per index, then evaluates the
five key models on three economic criteria:

- **Vol-targeting Sharpe** via :func:`volbench.economic.volatility_targeting`.
- **VaR coverage** via :func:`volbench.economic.var_backtest`.
- **Option-pricing error** via :func:`volbench.economic.option_pricing_loss`.

Results are printed as a per-model table for each index and written to
``results/economic.json``.

Headline question
-----------------
Does the statistically best model (LogHAR by QLIKE ranking) also deliver the
best economic value: highest vol-targeted Sharpe and VaR coverage closest to
the 5% nominal level?

Usage
-----
    PYTHONPATH=src python3 scripts/run_economic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import load_oxford_rv  # noqa: E402
from volbench.economic import option_pricing_loss, var_backtest, volatility_targeting  # noqa: E402

HORIZON: int = 1
MCS_REPS: int = 200
SEED: int = 0
RESULTS_DIR = ROOT / "results"

# Models to report (must be present in default_models()).
REPORT_MODELS: tuple[str, ...] = ("LogHAR", "HAR", "RW", "EWMA", "GBRT")


def _next_day_returns(close: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned next-day log returns for each origin.

    Parameters
    ----------
    close : np.ndarray
        Full close-price series, length N.
    origins : np.ndarray
        Positional indices into the rv / close series (from BacktestResult).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(ret_next, mask)`` where ``ret_next[k] = log(close[origins[k]+1] /
        close[origins[k]])`` and ``mask`` is a boolean array that is False for
        any origin where ``origin + 1`` would be out of range.
    """
    n = close.size
    mask = origins + 1 < n
    valid_origins = origins[mask]
    ret_next = np.log(close[valid_origins + 1] / close[valid_origins])
    return ret_next, mask


def run_economic() -> dict:
    """Evaluate economic value for all indices and key models.

    Returns
    -------
    dict
        Nested results keyed by ticker then model name.
    """
    ds = load_oxford_rv()
    tickers = ds.tickers
    all_results: dict[str, dict[str, dict]] = {}

    for tk in tickers:
        rv = ds.series(tk)
        frame = ds.frame(tk)
        close = frame["close_price"].to_numpy(dtype=float)

        res = run_backtest(rv, horizon=HORIZON, mcs_reps=MCS_REPS, seed=SEED)
        origins = res.origins  # positional indices into rv/close

        ret_next, mask = _next_day_returns(close, origins)

        # Statistical winner for reference.
        best_stat = min(res.mean_losses["QLIKE"], key=res.mean_losses["QLIKE"].get)

        ticker_results: dict[str, dict] = {}
        print(f"\n{'=' * 72}")
        print(f"  {tk}   (n_origins={origins.size}, n_valid={mask.sum()}, "
              f"stat_best={best_stat})")
        print(f"  {'Model':<12} {'Sharpe':>8} {'MaxDD':>8} {'ViolRate':>10} "
              f"{'KupiecP':>9} {'OptRMSE':>10}")
        print(f"  {'-' * 61}")

        for name in REPORT_MODELS:
            if name not in res.forecasts:
                continue  # model absent from this run's suite

            fc_all = res.forecasts[name]  # aligned to res.origins
            fc = fc_all[mask]
            rv_realized = res.realized[mask]

            vt = volatility_targeting(ret_next, fc)
            vr = var_backtest(ret_next, fc)
            opl = option_pricing_loss(fc, rv_realized, horizon_days=HORIZON)

            ticker_results[name] = {
                "vol_targeting": vt,
                "var_backtest": vr,
                "option_loss": opl,
                "qlike": res.mean_losses["QLIKE"].get(name, float("nan")),
            }

            print(
                f"  {name:<12} {vt['sharpe']:>8.3f} {vt['max_drawdown']:>8.3f} "
                f"{vr['violation_rate']:>10.4f} {vr['kupiec_p']:>9.4f} "
                f"{opl['rmse_price']:>10.5f}"
            )

        all_results[tk] = ticker_results

    # Summarise across tickers: for each model compute average Sharpe and avg
    # |violation_rate - 0.05| (proximity to nominal VaR coverage).
    print(f"\n{'=' * 72}")
    print("  CROSS-INDEX SUMMARY")
    print(f"  {'Model':<12} {'AvgSharpe':>10} {'AvgViolDev':>12} {'AvgOptRMSE':>12}")
    print(f"  {'-' * 50}")
    summary_rows: dict[str, dict[str, float]] = {}
    for name in REPORT_MODELS:
        sharpes, viol_devs, opt_rmses = [], [], []
        for tk in tickers:
            if name in all_results.get(tk, {}):
                r = all_results[tk][name]
                sharpes.append(r["vol_targeting"]["sharpe"])
                viol_devs.append(abs(r["var_backtest"]["violation_rate"] - 0.05))
                opt_rmses.append(r["option_loss"]["rmse_price"])
        if not sharpes:
            continue
        avg_sharpe = float(np.nanmean(sharpes))
        avg_viol_dev = float(np.nanmean(viol_devs))
        avg_opt_rmse = float(np.nanmean(opt_rmses))
        summary_rows[name] = {
            "avg_sharpe": avg_sharpe,
            "avg_viol_dev": avg_viol_dev,
            "avg_opt_rmse": avg_opt_rmse,
        }
        print(f"  {name:<12} {avg_sharpe:>10.3f} {avg_viol_dev:>12.4f} {avg_opt_rmse:>12.5f}")

    # Headline finding.
    if summary_rows:
        best_econ = max(summary_rows, key=lambda n: summary_rows[n]["avg_sharpe"])
        best_var = min(summary_rows, key=lambda n: summary_rows[n]["avg_viol_dev"])
        print(f"\n  Headline: economic Sharpe winner = {best_econ}, "
              f"VaR coverage winner = {best_var}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "economic.json"
    # Convert numpy scalars to plain Python floats for JSON serialisation.
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None  # JSON has no NaN; use null
        return obj

    with open(out_path, "w") as fh:
        json.dump(_jsonify({"tickers": tickers, "by_ticker": all_results, "summary": summary_rows}),
                  fh, indent=2)
    print(f"\n  Saved results to {out_path}")
    return all_results


if __name__ == "__main__":
    run_economic()
