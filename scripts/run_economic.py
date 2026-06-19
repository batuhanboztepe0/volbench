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

# VaR tail-distribution assumptions to backtest on real data (the t/fhs shapes are
# calibrated out-of-sample on a leading warm-up block — see economic.var_backtest).
VAR_DISTS: tuple[str, ...] = ("normal", "t", "fhs")


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
            # return_es=True also attaches the Expected-Shortfall backtest
            # (Acerbi-Szekely Z1/Z2, FZ0 loss) on the same held-out window (C1d).
            var_by_dist = {d: var_backtest(ret_next, fc, dist=d, return_es=True) for d in VAR_DISTS}
            vr = var_by_dist["normal"]  # kept as the back-compat default
            opl = option_pricing_loss(fc, rv_realized, horizon_days=HORIZON)

            ticker_results[name] = {
                "vol_targeting": vt,
                "var_backtest": vr,
                "var_by_dist": var_by_dist,
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

    # VaR coverage by tail distribution, averaged across indices (for the headline
    # forecaster). This is what grounds the "which distribution fixes 5% coverage"
    # claim: normal vs Student-t vs FHS, each evaluated out-of-sample.
    HEADLINE_VAR_MODEL = "LogHAR"
    coverage_by_dist: dict[str, dict[str, float]] = {}
    print(f"\n  VaR coverage by tail distribution ({HEADLINE_VAR_MODEL}, nominal 5%)")
    print(f"  {'Dist':<8} {'AvgViolRate':>12} {'AvgViolDev':>12} {'DQ_reject_frac':>15}")
    print(f"  {'-' * 49}")
    for d in VAR_DISTS:
        rates, devs, dq_reject = [], [], []
        for tk in tickers:
            r = all_results.get(tk, {}).get(HEADLINE_VAR_MODEL)
            if r is None or "var_by_dist" not in r:
                continue
            vd = r["var_by_dist"][d]
            rates.append(vd["violation_rate"])
            devs.append(abs(vd["violation_rate"] - 0.05))
            dqp = vd.get("dq_pvalue")
            if dqp is not None and not np.isnan(dqp):
                dq_reject.append(1.0 if dqp < 0.05 else 0.0)
        if not rates:
            continue
        coverage_by_dist[d] = {
            "avg_violation_rate": float(np.nanmean(rates)),
            "avg_viol_dev": float(np.nanmean(devs)),
            "dq_reject_frac": float(np.nanmean(dq_reject)) if dq_reject else float("nan"),
            "n_indices": float(len(rates)),
        }
        c = coverage_by_dist[d]
        print(f"  {d:<8} {c['avg_violation_rate']:>12.4f} {c['avg_viol_dev']:>12.4f} "
              f"{c['dq_reject_frac']:>15.3f}")

    # Expected Shortfall by tail distribution (C1d), averaged across indices for
    # the headline forecaster. Goes beyond *how often* VaR breaches (coverage,
    # above) to *how bad*: Acerbi-Szekely Z (≈0 well-specified, <0 understated
    # tail) and the FZ0 loss (lower is better; a strictly consistent (VaR,ES)
    # score — siloed risk layer, never mixed with Track-1 QLIKE).
    es_by_dist: dict[str, dict[str, float]] = {}
    print(f"\n  Expected Shortfall by tail distribution ({HEADLINE_VAR_MODEL}, nominal 5%)")
    print(f"  {'Dist':<8} {'AS_Z1':>9} {'AS_Z2':>9} {'AS_rej_frac':>12} {'FZ_mean':>10}")
    print(f"  {'-' * 50}")
    for d in VAR_DISTS:
        z1s, z2s, rej, fzs = [], [], [], []
        for tk in tickers:
            r = all_results.get(tk, {}).get(HEADLINE_VAR_MODEL)
            if r is None or "var_by_dist" not in r:
                continue
            vd = r["var_by_dist"][d]
            if "as_Z1" not in vd:
                continue
            if not np.isnan(vd["as_Z1"]):
                z1s.append(vd["as_Z1"])
            if not np.isnan(vd["as_Z2"]):
                z2s.append(vd["as_Z2"])
            asp = vd.get("as_p")
            if asp is not None and not np.isnan(asp):
                rej.append(1.0 if asp < 0.05 else 0.0)
            fzs.append(vd["fz_mean"])
        if not fzs:
            continue
        es_by_dist[d] = {
            "avg_as_Z1": float(np.nanmean(z1s)) if z1s else float("nan"),
            "avg_as_Z2": float(np.nanmean(z2s)) if z2s else float("nan"),
            "as_reject_frac": float(np.nanmean(rej)) if rej else float("nan"),
            "avg_fz_loss": float(np.nanmean(fzs)),
            "n_indices": float(len(fzs)),
        }
        e = es_by_dist[d]
        print(f"  {d:<8} {e['avg_as_Z1']:>9.3f} {e['avg_as_Z2']:>9.3f} "
              f"{e['as_reject_frac']:>12.3f} {e['avg_fz_loss']:>10.4f}")
    if es_by_dist:
        best_fz = min(es_by_dist, key=lambda dd: es_by_dist[dd]["avg_fz_loss"])
        print(f"\n  ES headline: lowest FZ0 loss = {best_fz} "
              f"(the best-scoring (VaR, ES) tail model by the consistent FZ rule)")

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
        json.dump(_jsonify({
            "tickers": tickers,
            "by_ticker": all_results,
            "summary": summary_rows,
            "var_coverage_by_dist": coverage_by_dist,
            "es_by_dist": es_by_dist,
        }), fh, indent=2)
    print(f"\n  Saved results to {out_path}")
    return all_results


if __name__ == "__main__":
    run_economic()
