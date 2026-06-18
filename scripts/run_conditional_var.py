"""Conditional-VaR experiment: can a reactive variance engine pass the DQ test?

The economic layer's VaR is built on the log-HAR realized-variance forecast, which
is accurate on average but smooth, so on real returns it under-covers the 5% tail
and its violations cluster (the Engle-Manganelli dynamic-quantile test rejects).
This script asks whether a *reactive* conditional-variance engine — RiskMetrics
EWMA or a (GJR-)GARCH(1,1) — fixes that, evaluated out-of-sample on the eight
indices' daily returns, with a normal and an out-of-sample-calibrated FHS tail.

This is a RISK comparison only (VaR coverage + DQ); it never ranks engines by
QLIKE against the realized-variance benchmark (ROADMAP invariant 4). Writes
``results/conditional_var.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.conditional_var import (  # noqa: E402
    ewma_variance_forecast,
    garch_variance_forecast,
)
from volbench.data import DEFAULT_TICKERS, load_sp500_returns  # noqa: E402
from volbench.economic import var_backtest  # noqa: E402

RESULTS_DIR = ROOT / "results"
MIN_TRAIN = 1000
REFIT_EVERY = 63
ALPHA = 0.05
DISTS = ("normal", "fhs")

ENGINES = {
    "EWMA": lambda y: ewma_variance_forecast(y, MIN_TRAIN),
    "GARCH": lambda y: garch_variance_forecast(y, MIN_TRAIN, REFIT_EVERY, o=0),
    "GJR-GARCH": lambda y: garch_variance_forecast(y, MIN_TRAIN, REFIT_EVERY, o=1),
}


def run_conditional_var() -> dict:
    tickers = list(DEFAULT_TICKERS)
    by_ticker: dict[str, dict] = {}

    for tk in tickers:
        y = load_sp500_returns(ticker=tk).to_numpy() * 100.0  # percent returns
        by_ticker[tk] = {}
        for eng, fn in ENGINES.items():
            fc, org = fn(y)
            fut = y[org + 1]
            by_ticker[tk][eng] = {
                d: var_backtest(fut, fc, alpha=ALPHA, dist=d) for d in DISTS
            }
        print(f"  {tk}: done ({y.size} returns)")

    # Cross-index summary per engine x dist: mean violation rate and DQ-reject frac.
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for eng in ENGINES:
        summary[eng] = {}
        for d in DISTS:
            rates, devs, dq_rej = [], [], []
            for tk in tickers:
                r = by_ticker[tk][eng][d]
                rates.append(r["violation_rate"])
                devs.append(abs(r["violation_rate"] - ALPHA))
                dqp = r.get("dq_pvalue")
                if dqp is not None and not np.isnan(dqp):
                    dq_rej.append(1.0 if dqp < 0.05 else 0.0)
            summary[eng][d] = {
                "avg_violation_rate": float(np.nanmean(rates)),
                "avg_viol_dev": float(np.nanmean(devs)),
                "dq_reject_frac": float(np.nanmean(dq_rej)) if dq_rej else float("nan"),
                "n_indices": float(len(rates)),
            }

    # Baseline: the log-HAR realized-variance VaR from the economic layer.
    baseline = {}
    econ_path = RESULTS_DIR / "economic.json"
    if econ_path.exists():
        econ = json.load(open(econ_path))
        baseline = econ.get("var_coverage_by_dist", {})

    # Report.
    print(f"\n  {'Engine':<12} {'Tail':<7} {'AvgViol':>8} {'AvgDev':>8} {'DQ_rej':>8}")
    print(f"  {'-' * 45}")
    for eng in ENGINES:
        for d in DISTS:
            s = summary[eng][d]
            print(f"  {eng:<12} {d:<7} {s['avg_violation_rate']:>8.4f} "
                  f"{s['avg_viol_dev']:>8.4f} {s['dq_reject_frac']:>8.3f}")
    if baseline:
        print(f"  {'-' * 45}")
        for d in DISTS:
            b = baseline.get(d)
            if b:
                print(f"  {'LogHAR-RV':<12} {d:<7} {b['avg_violation_rate']:>8.4f} "
                      f"{b['avg_viol_dev']:>8.4f} {b['dq_reject_frac']:>8.3f}")

    out = {
        "tickers": tickers,
        "min_train": MIN_TRAIN,
        "refit_every": REFIT_EVERY,
        "alpha": ALPHA,
        "by_ticker": by_ticker,
        "summary": summary,
        "loghar_rv_baseline": baseline,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    with open(RESULTS_DIR / "conditional_var.json", "w") as fh:
        json.dump(_jsonify(out), fh, indent=2)
    print(f"\n  Saved results to {RESULTS_DIR / 'conditional_var.json'}")
    return out


if __name__ == "__main__":
    run_conditional_var()
