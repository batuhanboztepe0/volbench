"""CAViaR VaR evaluation on the eight Oxford-Man equity indices.

Completes the VaR layer (ROADMAP method gap): the variance-based VaR families
(normal / Student-t / FHS in ``economic.var_backtest``) under-cover and fail the
Engle-Manganelli Dynamic-Quantile test on real data. CAViaR (Engle-Manganelli
2004) models the 5% return quantile *directly* by regression-quantile
minimisation. This script runs three CAViaR specifications and, on the *same*
held-out window per index, the normal/t/FHS baselines and the return-based
GARCH / GJR-GARCH / EWMA conditional-variance engines, so the Dynamic-Quantile
pass-rates are all directly comparable.

Specifications:
  SAV       symmetric absolute value
  AS        asymmetric slope (leverage)
  REALIZED  CAViaR augmented with the LogHAR variance forecast (sqrt) — ties the
            project's realized-volatility benchmark into the VaR layer

Usage:  PYTHONPATH=src python3 scripts/run_caviar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.backtest import run_backtest  # noqa: E402
from volbench.caviar import caviar_var_forecasts  # noqa: E402
from volbench.conditional_var import ewma_variance_forecast, garch_variance_forecast  # noqa: E402
from volbench.data import load_oxford_rv  # noqa: E402
from volbench.economic import backtest_var_forecasts, var_backtest  # noqa: E402
from volbench.models import HAR, LogHAR  # noqa: E402

HORIZON = 1
SEED = 0
ALPHA = 0.05
MIN_TRAIN = 500       # CAViaR walk-forward warm-up on the return series
REFIT_EVERY = 125
N_STARTS = 5
CAVIAR_SPECS = ("SAV", "AS", "REALIZED")
VAR_DISTS = ("normal", "t", "fhs")
RESULTS_DIR = ROOT / "results"


def _next_day_returns(close: np.ndarray, origins: np.ndarray):
    n = close.size
    mask = origins + 1 < n
    valid = origins[mask]
    return np.log(close[valid + 1] / close[valid]), mask


def run_caviar() -> dict:
    ds = load_oxford_rv()
    tickers = ds.tickers
    by_ticker: dict[str, dict] = {}

    print(f"{'Ticker':<10} {'Model':<10} {'Viol':>7} {'KupiecP':>8} {'DQ_p':>7} {'DQ pass':>8}")
    print("-" * 56)

    for tk in tickers:
        rv = ds.series(tk)
        close = ds.frame(tk)["close_price"].to_numpy(dtype=float)
        # Only LogHAR (+ HAR benchmark) needed: the forecast feeds Realized-CAViaR.
        res = run_backtest(rv, horizon=HORIZON, models=[HAR(), LogHAR()],
                           benchmark="HAR", mcs_reps=200, seed=SEED)
        ret_next, mask = _next_day_returns(close, res.origins)
        fc_loghar = res.forecasts["LogHAR"][mask]  # variance forecast, aligned

        models: dict[str, dict] = {}

        for spec in CAVIAR_SPECS:
            exog = fc_loghar if spec == "REALIZED" else None
            var = caviar_var_forecasts(
                ret_next, alpha=ALPHA, spec=spec, exog=exog,
                min_train=MIN_TRAIN, refit_every=REFIT_EVERY, n_starts=N_STARTS, seed=SEED,
            )
            bt = backtest_var_forecasts(ret_next, var, alpha=ALPHA)
            bt["dq_reject"] = bool(bt["dq_pvalue"] < 0.05)
            models[f"CAViaR-{spec}"] = bt

        # same-window variance-based baselines (warmup == MIN_TRAIN for alignment)
        for d in VAR_DISTS:
            bt = var_backtest(ret_next, fc_loghar, alpha=ALPHA, dist=d, warmup=MIN_TRAIN)
            bt["dq_reject"] = bool(bt["dq_pvalue"] < 0.05)
            models[d] = bt

        # same-window return-based conditional-variance VaR engines: re-run the
        # GARCH family on this exact ret_next / min_train so the Dynamic-Quantile
        # pass-rate is directly comparable to CAViaR (arch wants percent units;
        # VaR coverage + DQ are scale-invariant).
        ret_pct = ret_next * 100.0
        for label, o in (("GARCH", 0), ("GJR-GARCH", 1)):
            gv, gorig = garch_variance_forecast(
                ret_pct, min_train=MIN_TRAIN, refit_every=REFIT_EVERY, o=o)
            bt = var_backtest(ret_pct[gorig + 1], gv, alpha=ALPHA, dist="normal", warmup=0)
            bt["dq_reject"] = bool(bt["dq_pvalue"] < 0.05)
            models[label] = bt
        ev, eorig = ewma_variance_forecast(ret_pct, min_train=MIN_TRAIN)
        bt = var_backtest(ret_pct[eorig + 1], ev, alpha=ALPHA, dist="normal", warmup=0)
        bt["dq_reject"] = bool(bt["dq_pvalue"] < 0.05)
        models["EWMA-RM"] = bt

        by_ticker[tk] = models
        for name, bt in models.items():
            print(f"{tk:<10} {name:<10} {bt['violation_rate']:>7.4f} "
                  f"{bt['kupiec_p']:>8.3f} {bt['dq_pvalue']:>7.3f} "
                  f"{'PASS' if not bt['dq_reject'] else 'fail':>8}")
        print("-" * 56)

    # ---- cross-index summary ----
    all_models = list(next(iter(by_ticker.values())).keys())
    summary: dict[str, dict] = {}
    for name in all_models:
        rows = [by_ticker[tk][name] for tk in tickers if name in by_ticker[tk]]
        summary[name] = {
            "avg_violation_rate": float(np.mean([r["violation_rate"] for r in rows])),
            "avg_viol_dev": float(np.mean([abs(r["violation_rate"] - ALPHA) for r in rows])),
            "avg_dq_pvalue": float(np.mean([r["dq_pvalue"] for r in rows])),
            "dq_reject_frac": float(np.mean([r["dq_reject"] for r in rows])),
            "dq_pass_count": int(sum(not r["dq_reject"] for r in rows)),
            "n_indices": len(rows),
        }

    print("\nCROSS-INDEX SUMMARY")
    print(f"{'Model':<14} {'AvgViol':>8} {'ViolDev':>8} {'AvgDQp':>8} {'DQ rej frac':>12} {'DQ pass':>8}")
    print("-" * 62)
    for name, s in summary.items():
        print(f"{name:<14} {s['avg_violation_rate']:>8.4f} {s['avg_viol_dev']:>8.4f} "
              f"{s['avg_dq_pvalue']:>8.3f} {s['dq_reject_frac']:>12.3f} "
              f"{s['dq_pass_count']:>5}/{s['n_indices']}")

    best = min(summary, key=lambda n: (summary[n]["dq_reject_frac"], summary[n]["avg_viol_dev"]))
    print(f"\nHeadline: best DQ pass-rate = {best} "
          f"(rejects {summary[best]['dq_reject_frac']:.0%} of indices, "
          f"vs normal {summary['normal']['dq_reject_frac']:.0%} / "
          f"t {summary['t']['dq_reject_frac']:.0%} / fhs {summary['fhs']['dq_reject_frac']:.0%})")

    return {
        "alpha": ALPHA, "horizon": HORIZON, "min_train": MIN_TRAIN,
        "refit_every": REFIT_EVERY, "n_starts": N_STARTS, "seed": SEED,
        "tickers": tickers, "by_ticker": by_ticker, "summary": summary,
        "best_dq": best,
    }


if __name__ == "__main__":
    out = run_caviar()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "caviar.json").write_text(json.dumps(out, indent=2))
    print(f"\n[written] {RESULTS_DIR / 'caviar.json'}")
