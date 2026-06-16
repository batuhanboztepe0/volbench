"""Tier 2F: regime / subsample analysis (calm vs crisis).

The real bundled data is dated (2000-2022), so we can ask the question a risk
audience cares about most: *does the model ranking survive a crisis?* For each
index we score the full model suite, map every test origin to its date and its
volatility state, then re-run the Model Confidence Set within named crisis
windows (GFC, COVID) and within calm vs turbulent halves of the sample (split by
the realized-variance level known at the origin — no look-ahead in the regime
assignment).

Writes ``results/regime.json``.

Usage
-----
    python scripts/run_regime.py [--mcs-reps N]
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
from volbench.evaluation import model_confidence_set  # noqa: E402
from volbench.losses import mean_loss  # noqa: E402

HORIZON: int = 1
RESULTS_DIR = ROOT / "results"
MIN_REGIME_OBS: int = 40  # need enough origins for a meaningful MCS

CRISIS_WINDOWS = {
    "GFC": ("2008-09-01", "2009-06-30"),
    "COVID": ("2020-02-15", "2020-06-30"),
}


def _regime_masks(dates: pd.DatetimeIndex, rv_at_origin: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks over origins for each regime."""
    masks: dict[str, np.ndarray] = {"Full": np.ones(len(dates), dtype=bool)}
    in_any_crisis = np.zeros(len(dates), dtype=bool)
    for name, (lo, hi) in CRISIS_WINDOWS.items():
        m = np.asarray((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))
        masks[name] = m
        in_any_crisis |= m
    # Volatility-state split on the level known at the origin (median cut).
    med = float(np.median(rv_at_origin))
    masks["Calm"] = (rv_at_origin <= med) & ~in_any_crisis
    masks["Turbulent"] = rv_at_origin > med
    return masks


def run_all(mcs_reps: int = 1000, seed: int = 0) -> dict:
    """Run the regime analysis across indices."""
    ds = load_oxford_rv()
    tickers = ds.tickers
    names = None
    # regime -> model -> list of per-index (mean_qlike, in_mcs)
    agg: dict[str, dict] = {}

    for tk in tickers:
        fr = ds.frame(tk)
        rv = fr["rv5"].to_numpy()
        dates_all = fr.index
        # Cheap MCS here (we only need the per-origin losses); regime MCS is run below.
        res = run_backtest(rv, horizon=HORIZON, mcs_reps=100, seed=seed)
        names = res.model_names
        origin_dates = dates_all[res.origins]
        rv_at_origin = rv[res.origins]
        masks = _regime_masks(origin_dates, rv_at_origin)

        for regime, mask in masks.items():
            if int(mask.sum()) < MIN_REGIME_OBS:
                continue
            sub_losses = {n: res.losses["QLIKE"][n][mask] for n in names}
            mcs = model_confidence_set(sub_losses, alpha=0.10, reps=mcs_reps, seed=seed)
            means = {n: mean_loss(sub_losses[n]) for n in names}
            slot = agg.setdefault(regime, {"n_obs": [], "models": {n: {"qlike": [], "mcs": 0}
                                                                   for n in names}})
            slot["n_obs"].append(int(mask.sum()))
            for n in names:
                slot["models"][n]["qlike"].append(means[n])
                if n in mcs.included:
                    slot["models"][n]["mcs"] += 1

    # Summarise: per regime, average QLIKE and MCS count across indices.
    summary: dict = {"horizon": HORIZON, "tickers": tickers, "by_regime": {}}
    order = ["Full", "Calm", "Turbulent", "GFC", "COVID"]
    for regime in order:
        if regime not in agg:
            continue
        slot = agg[regime]
        n_idx = len(slot["n_obs"])
        rows = {}
        for n in names:
            q = slot["models"][n]["qlike"]
            rows[n] = {"avg_qlike": float(np.mean(q)), "mcs_count": slot["models"][n]["mcs"],
                       "n_indices": n_idx}
        summary["by_regime"][regime] = {
            "mean_origins_per_index": float(np.mean(slot["n_obs"])),
            "n_indices": n_idx,
            "models": rows,
        }
        ranked = sorted(rows, key=lambda m: rows[m]["avg_qlike"])
        print(f"\n=== {regime}  (~{np.mean(slot['n_obs']):.0f} origins/index, "
              f"{n_idx} indices) ===")
        print(f"{'model':<10}{'QLIKE':>10}{'MCS/idx':>10}")
        for n in ranked:
            print(f"{n:<10}{rows[n]['avg_qlike']:>10.4f}{rows[n]['mcs_count']:>7}/{n_idx}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "regime.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'regime.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-reps", type=int, default=1000, dest="mcs_reps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_all(mcs_reps=args.mcs_reps, seed=args.seed)


if __name__ == "__main__":
    main()
