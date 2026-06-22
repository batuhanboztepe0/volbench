"""Track 1 driver: realized-variance forecasting benchmark on eight indices.

Runs the full model suite over every bundled index and every forecast horizon,
writes per-horizon leaderboards (CSV), an aggregate summary (JSON), and prints
the headline findings. All numbers are reproducible from the bundled data with
a fixed seed.

Usage
-----
    python scripts/run_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import load_oxford_rv  # noqa: E402
from volbench.losses import mincer_zarnowitz  # noqa: E402
from volbench.meta import run_meta  # noqa: E402
from volbench.models import default_models  # noqa: E402

HORIZONS: tuple[int, ...] = (1, 5, 22)
MCS_REPS: int = 2000
SEED: int = 0
SIGNIFICANCE: float = 0.05
TABLES_DIR = ROOT / "results" / "tables"
RESULTS_DIR = ROOT / "results"


def _model_names() -> list[str]:
    """Return the model names in suite order."""
    return [m.name for m in default_models()]


def run_all() -> dict:
    """Execute the benchmark across all indices and horizons.

    Returns
    -------
    dict
        Nested summary keyed by horizon, suitable for JSON serialisation and
        for driving the report.
    """
    ds = load_oxford_rv()
    tickers = ds.tickers
    names = _model_names()
    summary: dict = {"tickers": tickers, "horizons": list(HORIZONS),
                     "meta": run_meta(SEED, MCS_REPS), "by_horizon": {}}

    for h in HORIZONS:
        print(f"\n{'=' * 64}\nHORIZON h = {h}\n{'=' * 64}")
        qlike_table = pd.DataFrame(index=names, dtype=float)
        mse_table = pd.DataFrame(index=names, dtype=float)
        mcs_counts = {n: 0 for n in names}
        dm_beats_har = {n: 0 for n in names}
        dm_loses_har = {n: 0 for n in names}
        mz_best: dict[str, dict] = {}

        for tk in tickers:
            rv = ds.series(tk)
            res = run_backtest(rv, horizon=h, mcs_reps=MCS_REPS, seed=SEED)
            qlike_table[tk] = pd.Series(res.mean_losses["QLIKE"])
            mse_table[tk] = pd.Series(res.mean_losses["MSE-var"])

            for n in res.mcs["QLIKE"].included:
                mcs_counts[n] += 1

            for n, dm in res.dm_vs_har["QLIKE"].items():
                if not np.isfinite(dm["p_value"]):
                    continue
                if dm["p_value"] < SIGNIFICANCE and dm["favored"] < 0:
                    dm_beats_har[n] += 1
                elif dm["p_value"] < SIGNIFICANCE and dm["favored"] > 0:
                    dm_loses_har[n] += 1

            # Mincer-Zarnowitz for the per-index QLIKE winner.
            best_name = min(res.mean_losses["QLIKE"], key=res.mean_losses["QLIKE"].get)
            mz = mincer_zarnowitz(res.realized, res.forecasts[best_name])
            mz_best[tk] = {"model": best_name, **mz}

            print(f"  {tk:<10} best={best_name:<8} "
                  f"QLIKE={res.mean_losses['QLIKE'][best_name]:.4f} "
                  f"MCS={sorted(res.mcs['QLIKE'].included)}")

        # Average rank (1 = best) across indices, by QLIKE.
        ranks = qlike_table.rank(axis=0, method="average")
        avg_rank = ranks.mean(axis=1).sort_values()

        qlike_table["avg_QLIKE"] = qlike_table.mean(axis=1)
        qlike_table["avg_rank"] = avg_rank
        qlike_table["MCS_count"] = pd.Series(mcs_counts)
        qlike_table = qlike_table.sort_values("avg_rank")

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        qlike_table.to_csv(TABLES_DIR / f"leaderboard_qlike_h{h}.csv")
        mse_table.to_csv(TABLES_DIR / f"leaderboard_mse_h{h}.csv")

        summary["by_horizon"][str(h)] = {
            "avg_rank": {n: float(avg_rank[n]) for n in names},
            "avg_qlike": {n: float(qlike_table.loc[n, "avg_QLIKE"]) for n in names},
            "mcs_count": mcs_counts,
            "dm_beats_har": dm_beats_har,
            "dm_loses_har": dm_loses_har,
            "mz_best": mz_best,
            "n_indices": len(tickers),
        }

        print(f"\n  Average rank (h={h}):")
        for n in avg_rank.index:
            print(f"    {n:<10} rank={avg_rank[n]:.2f}  "
                  f"MCS={mcs_counts[n]}/{len(tickers)}  "
                  f"beats_HAR={dm_beats_har[n]}  loses_HAR={dm_loses_har[n]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved summary to {RESULTS_DIR / 'summary.json'}")
    return summary


if __name__ == "__main__":
    run_all()
