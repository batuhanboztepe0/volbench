"""Tier 1C: which HAR variant wins, using real jump / semivariance measures.

The headline benchmark (``run_benchmark.py``) answers "HAR vs everything else".
This script answers the within-family question a quant reviewer actually asks:
given real bipower variation, jump variation and realized semivariances, does
**HAR-J**, **HAR-CJ** (continuous/jump split) or **SHAR** (semivariance HAR)
beat plain HAR — and do the log variants beat the level variants? Each model is
scored with the same walk-forward + DM + MCS machinery, benchmarked against
log-HAR.

Writes ``results/har_family.json`` and per-horizon leaderboards.

Usage
-----
    python scripts/run_har_family.py [--mcs-reps N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import load_oxford_rv  # noqa: E402
from volbench.models import har_family  # noqa: E402

HORIZONS: tuple[int, ...] = (1, 5, 22)
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"
BENCHMARK = "LogHAR"


def run_all(mcs_reps: int = 2000, seed: int = 0) -> dict:
    """Run the HAR-family comparison across indices and horizons."""
    ds = load_oxford_rv()
    tickers = ds.tickers
    names = [m.name for m in har_family(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))]
    summary: dict = {"tickers": tickers, "horizons": list(HORIZONS), "benchmark": BENCHMARK,
                     "by_horizon": {}}

    for h in HORIZONS:
        print(f"\n{'=' * 64}\nHAR FAMILY  h = {h}\n{'=' * 64}")
        qlike_table = pd.DataFrame(index=names, dtype=float)
        mcs_counts = {n: 0 for n in names}
        beats_loghar = {n: 0 for n in names}

        for tk in tickers:
            fr = ds.frame(tk)
            rv = fr["rv5"].to_numpy()
            fam = har_family(fr["cont"].to_numpy(), fr["jump"].to_numpy(),
                             fr["rsv_minus"].to_numpy(), fr["rsv_plus"].to_numpy())
            res = run_backtest(rv, horizon=h, models=fam, mcs_reps=mcs_reps,
                               seed=seed, benchmark=BENCHMARK)
            qlike_table[tk] = pd.Series(res.mean_losses["QLIKE"])
            for n in res.mcs["QLIKE"].included:
                mcs_counts[n] += 1
            for n, dm in res.dm_vs_har["QLIKE"].items():
                if np.isfinite(dm["p_value"]) and dm["p_value"] < 0.05 and dm["favored"] < 0:
                    beats_loghar[n] += 1
            best = min(res.mean_losses["QLIKE"], key=res.mean_losses["QLIKE"].get)
            print(f"  {tk:<10} best={best:<10} QLIKE={res.mean_losses['QLIKE'][best]:.4f} "
                  f"MCS={sorted(res.mcs['QLIKE'].included)}")

        ranks = qlike_table.rank(axis=0, method="average")
        avg_rank = ranks.mean(axis=1).sort_values()
        qlike_table["avg_QLIKE"] = qlike_table.mean(axis=1)
        qlike_table["avg_rank"] = avg_rank
        qlike_table["MCS_count"] = pd.Series(mcs_counts)
        qlike_table = qlike_table.sort_values("avg_rank")

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        qlike_table.to_csv(TABLES_DIR / f"har_family_qlike_h{h}.csv")

        summary["by_horizon"][str(h)] = {
            "avg_rank": {n: float(avg_rank[n]) for n in names},
            "avg_qlike": {n: float(qlike_table.loc[n, "avg_QLIKE"]) for n in names},
            "mcs_count": mcs_counts,
            "beats_loghar": beats_loghar,
            "n_indices": len(tickers),
        }
        print(f"\n  Average rank (h={h}):")
        for n in avg_rank.index:
            print(f"    {n:<10} rank={avg_rank[n]:.2f}  MCS={mcs_counts[n]}/{len(tickers)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "har_family.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'har_family.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-reps", type=int, default=2000, dest="mcs_reps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_all(mcs_reps=args.mcs_reps, seed=args.seed)


if __name__ == "__main__":
    main()
