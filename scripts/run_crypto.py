"""Track 3 driver: realized-volatility forecasting on crypto (BTC/ETH/BNB/SOL).

Crypto is the generality test and the real-intraday payoff. Unlike the equity
panel, the crypto measures are computed from real Binance 5-minute bars, so this
is the first track where:

* the full model suite is scored on a 24/7, very-high-volatility, fat-tailed
  asset class — does log-HAR still win?;
* **HARQ runs on real realized quarticity** (Bollerslev-Patton-Quaedvlieg) rather
  than on simulation;
* cross-coin **volatility spillover** is tested with the same DM/MCS machinery.

Writes ``results/crypto.json`` and per-horizon leaderboards.

Usage
-----
    python scripts/run_crypto.py [--mcs-reps N]
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
from volbench.data import CRYPTO_DAYS_PER_YEAR, load_crypto_rv  # noqa: E402
from volbench.models import HARQ, default_models  # noqa: E402
from volbench.multivariate import spillover_backtest  # noqa: E402

HORIZONS: tuple[int, ...] = (1, 5, 22)
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"


def run_all(mcs_reps: int = 2000, seed: int = 0) -> dict:
    """Run the crypto benchmark (with HARQ) and the cross-coin spillover study."""
    ds = load_crypto_rv()
    coins = ds.tickers
    base = [m.name for m in default_models()]
    names = base + ["HARQ"]
    summary: dict = {"coins": coins, "horizons": list(HORIZONS),
                     "annualisation": CRYPTO_DAYS_PER_YEAR, "by_horizon": {}, "spillover": {}}

    for h in HORIZONS:
        print(f"\n{'=' * 64}\nCRYPTO  h = {h}\n{'=' * 64}")
        qlike_table = pd.DataFrame(index=names, dtype=float)
        mcs_counts = {n: 0 for n in names}
        harq_beats_har = 0

        for c in coins:
            fr = ds.frame(c)
            rv = fr["rv5"].to_numpy()
            suite = default_models() + [HARQ(fr["rq"].to_numpy())]
            res = run_backtest(rv, horizon=h, models=suite, mcs_reps=mcs_reps,
                               seed=seed, benchmark="HAR")
            qlike_table[c] = pd.Series(res.mean_losses["QLIKE"])
            for n in res.mcs["QLIKE"].included:
                mcs_counts[n] += 1
            dm = res.dm_vs_har["QLIKE"].get("HARQ")
            if dm and np.isfinite(dm["p_value"]) and dm["p_value"] < 0.05 and dm["favored"] < 0:
                harq_beats_har += 1
            best = min(res.mean_losses["QLIKE"], key=res.mean_losses["QLIKE"].get)
            print(f"  {c:<5} best={best:<9} QLIKE={res.mean_losses['QLIKE'][best]:.4f} "
                  f"HARQ={res.mean_losses['QLIKE']['HARQ']:.4f} "
                  f"MCS={sorted(res.mcs['QLIKE'].included)}")

        ranks = qlike_table.rank(axis=0, method="average")
        avg_rank = ranks.mean(axis=1).sort_values()
        qlike_table["avg_QLIKE"] = qlike_table.mean(axis=1)
        qlike_table["avg_rank"] = avg_rank
        qlike_table["MCS_count"] = pd.Series(mcs_counts)
        qlike_table = qlike_table.sort_values("avg_rank")
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        qlike_table.to_csv(TABLES_DIR / f"crypto_qlike_h{h}.csv")

        summary["by_horizon"][str(h)] = {
            "avg_rank": {n: float(avg_rank[n]) for n in names},
            "avg_qlike": {n: float(qlike_table.loc[n, "avg_QLIKE"]) for n in names},
            "mcs_count": mcs_counts,
            "harq_beats_har": harq_beats_har,
            "n_coins": len(coins),
        }
        print(f"\n  Average rank (h={h}):")
        for n in avg_rank.index:
            print(f"    {n:<10} rank={avg_rank[n]:.2f}  MCS={mcs_counts[n]}/{len(coins)}")
        print(f"  HARQ beats HAR (real RQ) on {harq_beats_har}/{len(coins)} coins")

    # Cross-coin spillover at h = 1.
    print(f"\n{'=' * 64}\nCRYPTO SPILLOVER  h = 1\n{'=' * 64}")
    for target in coins:
        peers = [c for c in coins if c != target]
        sp = spillover_backtest(ds, target, peers, horizon=1, mcs_reps=500, seed=seed)
        summary["spillover"][target] = sp
        dm = sp.get("dm_crosshar_vs_loghar", {})
        print(f"  {target:<5} LogHAR={sp['mean_qlike'].get('LogHAR', float('nan')):.4f} "
              f"CrossHAR={sp['mean_qlike'].get('CrossHAR', float('nan')):.4f} "
              f"beats={sp.get('crosshar_beats_loghar')} (DM p={dm.get('p_value', float('nan')):.3g})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "crypto.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'crypto.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-reps", type=int, default=2000, dest="mcs_reps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_all(mcs_reps=args.mcs_reps, seed=args.seed)


if __name__ == "__main__":
    main()
