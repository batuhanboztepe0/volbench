"""Track 2 driver: return-based GARCH baselines on S&P 500 daily returns.

This track is **deliberately kept separate** from the realized-variance
benchmark (``ROADMAP.md`` invariant 4). It uses daily close-to-close returns
(derived from the bundled ``.SPX`` close prices) and scores one-step-ahead
*conditional-variance* forecasts against the **squared daily return** — a far
noisier proxy than 5-minute realized variance. Because of that proxy, QLIKE
levels here are an order of magnitude higher than Track 1's and the two are
never directly comparable.

Models: GARCH(1,1), GJR-GARCH(1,1,1), RiskMetrics EWMA (lambda = 0.94), and a
constant-variance benchmark. Walk-forward with periodic refits; between refits
parameters are held fixed and the conditional-variance recursion is filtered
forward (no look-ahead — every forecast uses only past returns and parameters
estimated on data available at the origin).

Writes ``results/garch.json``.

Usage
-----
    python scripts/run_garch.py [--min-train N] [--refit-every K]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.data import load_sp500_returns  # noqa: E402
from volbench.evaluation import diebold_mariano, model_confidence_set  # noqa: E402
from volbench.losses import mean_loss, mse_variance, qlike  # noqa: E402

RESULTS_DIR = ROOT / "results"
RISKMETRICS_LAMBDA: float = 0.94
DEFAULT_MIN_TRAIN: int = 1000
DEFAULT_REFIT_EVERY: int = 22


def _garch_forecast(
    y: np.ndarray, min_train: int, refit_every: int, o: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward 1-step conditional-variance forecasts from an arch model.

    Parameters are re-estimated on the expanding window every ``refit_every``
    origins; between refits they are held fixed and the variance recursion is
    filtered forward over the full return series, which yields the 1-step
    forecast for each origin from a single pass (cheap and look-ahead-free).

    Parameters
    ----------
    y : np.ndarray
        Returns in **percent** units (arch is numerically happier there).
    min_train : int
        First origin.
    refit_every : int
        Origins between parameter re-estimations.
    o : int, default 0
        Asymmetry order; ``o=1`` gives GJR-GARCH.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(forecast_var, origins)`` with forecasts in percent² for origins ``t``
        (forecasting the return at ``t + 1``).
    """
    from arch import arch_model

    n = y.size
    last_origin = n - 1  # need y[t+1]; valid origins t <= n-2
    origins = np.arange(min_train, last_origin, dtype=int)
    forecasts = np.full(origins.size, np.nan)
    am_full = arch_model(y, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="normal")

    cond_var = None
    for k, t in enumerate(origins):
        if k % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = arch_model(y[: t + 1], mean="Constant", vol="GARCH",
                                    p=1, o=o, q=1, dist="normal").fit(disp="off").params
                # Filter the conditional variance forward over the full series
                # with these fixed params; reuse it for every origin in the
                # block. cond_var[t+1] is the 1-step forecast made at t and uses
                # only returns up to t, so there is no look-ahead.
                cond_var = am_full.fix(params).conditional_volatility ** 2
        forecasts[k] = float(cond_var[t + 1])
    return forecasts, origins


def _riskmetrics_forecast(y: np.ndarray, min_train: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """RiskMetrics EWMA variance recursion (no estimation)."""
    n = y.size
    var = np.empty(n)
    var[0] = float(y[0] ** 2)
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * y[t - 1] ** 2
    # var[t] is the 1-step forecast for return t given info to t-1.
    origins = np.arange(min_train, n - 1, dtype=int)
    return var[origins + 1], origins


def _constant_forecast(y: np.ndarray, min_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window constant (sample) variance forecast."""
    n = y.size
    origins = np.arange(min_train, n - 1, dtype=int)
    csum2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    counts = origins + 1
    var = (csum2[origins + 1]) / counts  # mean of squared returns up to t
    return var, origins


def run_garch(min_train: int = DEFAULT_MIN_TRAIN, refit_every: int = DEFAULT_REFIT_EVERY) -> dict:
    """Run the Track-2 GARCH comparison and return a JSON-able summary."""
    ret = load_sp500_returns()
    y = (ret.to_numpy() * 100.0)  # percent returns

    specs = {
        "GARCH": lambda: _garch_forecast(y, min_train, refit_every, o=0),
        "GJR-GARCH": lambda: _garch_forecast(y, min_train, refit_every, o=1),
        "RiskMetrics": lambda: _riskmetrics_forecast(y, min_train, RISKMETRICS_LAMBDA),
        "Constant": lambda: _constant_forecast(y, min_train),
    }

    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fn in specs.items():
        print(f"  fitting {name} ...", flush=True)
        raw[name] = fn()

    common = raw["GARCH"][1]
    for _, org in raw.values():
        common = np.intersect1d(common, org)
    # Proxy: next-day squared percent return.
    proxy = (y[common + 1]) ** 2

    forecasts = {}
    for name, (fc, org) in raw.items():
        lookup = dict(zip(org.tolist(), fc.tolist()))
        forecasts[name] = np.array([lookup[o] for o in common.tolist()])

    qlike_loss = {n: qlike(proxy, forecasts[n]) for n in specs}
    mse_loss = {n: mse_variance(proxy, forecasts[n]) for n in specs}
    mean_q = {n: mean_loss(qlike_loss[n]) for n in specs}
    mean_m = {n: mean_loss(mse_loss[n]) for n in specs}

    mcs = model_confidence_set(qlike_loss, alpha=0.10, reps=2000, seed=0)
    dm = {
        n: diebold_mariano(qlike_loss[n], qlike_loss["GARCH"], horizon=1)
        for n in specs if n != "GARCH"
    }

    ranked = sorted(mean_q, key=mean_q.get)
    print(f"\nTrack 2 — GARCH on S&P 500 returns ({common.size} origins, proxy = r^2)")
    print(f"{'model':<14}{'QLIKE':>10}{'MSE-var':>12}{'in MCS':>9}")
    for n in ranked:
        print(f"{n:<14}{mean_q[n]:>10.3f}{mean_m[n]:>12.3f}"
              f"{'  yes' if n in mcs.included else '   no':>9}")

    return {
        "n_origins": int(common.size),
        "proxy": "squared daily return (percent^2)",
        "min_train": min_train,
        "refit_every": refit_every,
        "qlike": mean_q,
        "mse_var": mean_m,
        "mcs_included": sorted(mcs.included),
        "mcs_pvalues": mcs.p_values,
        "dm_vs_garch": dm,
        "note": "Track 2 is NOT comparable to Track 1 (different series/period/proxy).",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN, dest="min_train")
    parser.add_argument("--refit-every", type=int, default=DEFAULT_REFIT_EVERY, dest="refit_every")
    args = parser.parse_args()

    res = run_garch(min_train=args.min_train, refit_every=args.refit_every)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "garch.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
