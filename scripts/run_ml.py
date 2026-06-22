"""Tier 2D: a rigorous, leakage-free ML-vs-HAR comparison.

Tests whether modern learners beat log-HAR out of sample, and crucially whether
they win when given **richer features** (the continuous/jump decomposition and
the realized semivariances) where they plausibly could. Every ML model tunes its
hyperparameters with a strict expanding-window inner CV (no leakage) and forecasts
in log-variance space. A forecast combination (log-HAR + the best ML) tests
whether averaging beats either alone.

Models, all scored on each index's own series (identical origins): log-HAR
(benchmark), LightGBM on plain HAR features, LightGBM / XGBoost / MLP on the
enriched feature set, and a log-HAR + LightGBM-enriched combination.

Cross-asset enrichment (peers' lagged RV) is deliberately left to the separate
spillover study (`run_multivariate.py`) so the two effects are not conflated.

Writes ``results/ml.json`` and per-horizon leaderboards.

Usage
-----
    python scripts/run_ml.py [--mcs-reps N] [--horizons 1 5 22]
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
from volbench.evaluation import diebold_mariano, model_confidence_set  # noqa: E402
from volbench.losses import mean_loss, qlike  # noqa: E402
from volbench.meta import run_meta  # noqa: E402
from volbench.ml import enriched_ml, plain_ml  # noqa: E402
from volbench.models import LogHAR  # noqa: E402

RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"
BENCHMARK = "LogHAR"
COMBO = "Combo"  # post-hoc average of LogHAR and LGBM-enriched
# Models actually fit by run_backtest (Combo is built from their forecasts).
# Tree-based gradient boosting is the appropriate ML for tabular HAR features at
# this data size; a feed-forward MLP was also tested (see volbench.ml) but is
# data-starved and unstable on ~5,000 daily observations, so it is not featured.
MODEL_NAMES = ["LogHAR", "LGBM-plain", "LGBM-enriched", "XGB-enriched"]
ALL_NAMES = MODEL_NAMES + [COMBO]


def _build_models(fr: pd.DataFrame, seed: int, refit_every: int) -> list:
    """Instantiate the fitted model suite for one index from its measure frame."""
    cont = fr["cont"].to_numpy()
    jump = fr["jump"].to_numpy()
    rm = fr["rsv_minus"].to_numpy()
    rp = fr["rsv_plus"].to_numpy()
    re = refit_every
    return [
        LogHAR(),
        plain_ml("lgbm", refit_every=re, random_state=seed),
        enriched_ml("lgbm", cont, jump, rm, rp, refit_every=re, random_state=seed),
        enriched_ml("xgb", cont, jump, rm, rp, refit_every=re, random_state=seed),
    ]


def run_all(horizons=(1, 5, 22), mcs_reps: int = 1000, seed: int = 0,
            refit_every: int = 132) -> dict:
    """Run the ML comparison across indices and horizons."""
    ds = load_oxford_rv()
    tickers = ds.tickers
    names = ALL_NAMES
    summary: dict = {"tickers": tickers, "horizons": list(horizons),
                     "benchmark": BENCHMARK, "refit_every": refit_every,
                     "meta": run_meta(seed, mcs_reps, refit_every=refit_every),
                     "by_horizon": {}}

    for h in horizons:
        print(f"\n{'=' * 64}\nML COMPARISON  h = {h}\n{'=' * 64}")
        qlike_table = pd.DataFrame(index=names, dtype=float)
        mcs_counts = {n: 0 for n in names}
        beats_loghar = {n: 0 for n in names}
        # Track the LightGBM plain -> enriched improvement per index.
        enrich_gain = []

        for tk in tickers:
            fr = ds.frame(tk)
            rv = fr["rv5"].to_numpy()
            models = _build_models(fr, seed, refit_every)
            res = run_backtest(rv, horizon=h, models=models, mcs_reps=mcs_reps,
                               seed=seed, benchmark=BENCHMARK)
            # Build the combination post-hoc from the already-computed forecasts
            # (avoids re-fitting LGBM-enriched a second time).
            combo_fc = 0.5 * (res.forecasts["LogHAR"] + res.forecasts["LGBM-enriched"])
            combo_loss = qlike(res.realized, combo_fc)
            loss_dict = dict(res.losses["QLIKE"])
            loss_dict[COMBO] = combo_loss
            ql = dict(res.mean_losses["QLIKE"])
            ql[COMBO] = mean_loss(combo_loss)
            qlike_table[tk] = pd.Series(ql)

            mcs = model_confidence_set(loss_dict, alpha=0.10,
                                       block_length=max(10, h + 2), reps=mcs_reps, seed=seed)
            for n in mcs.included:
                mcs_counts[n] += 1
            dm_all = dict(res.dm_vs_har["QLIKE"])
            dm_all[COMBO] = diebold_mariano(combo_loss, res.losses["QLIKE"]["LogHAR"], horizon=h)
            for n, dm in dm_all.items():
                if np.isfinite(dm["p_value"]) and dm["p_value"] < 0.05 and dm["favored"] < 0:
                    beats_loghar[n] += 1
            enrich_gain.append(100.0 * (ql["LGBM-plain"] - ql["LGBM-enriched"]) / ql["LGBM-plain"])
            best = min(ql, key=ql.get)
            print(f"  {tk:<10} best={best:<14} LogHAR={ql['LogHAR']:.4f} "
                  f"MCS={sorted(mcs.included)}")

        ranks = qlike_table.rank(axis=0, method="average")
        avg_rank = ranks.mean(axis=1).sort_values()
        qlike_table["avg_QLIKE"] = qlike_table.mean(axis=1)
        qlike_table["avg_rank"] = avg_rank
        qlike_table["MCS_count"] = pd.Series(mcs_counts)
        qlike_table = qlike_table.sort_values("avg_rank")
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        qlike_table.to_csv(TABLES_DIR / f"ml_qlike_h{h}.csv")

        summary["by_horizon"][str(h)] = {
            "avg_rank": {n: float(avg_rank[n]) for n in names},
            "avg_qlike": {n: float(qlike_table.loc[n, "avg_QLIKE"]) for n in names},
            "mcs_count": mcs_counts,
            "beats_loghar": beats_loghar,
            "lgbm_enrichment_gain_pct_mean": float(np.mean(enrich_gain)) if enrich_gain else None,
            "n_indices": len(tickers),
        }
        print(f"\n  Average rank (h={h}):")
        for n in avg_rank.index:
            print(f"    {n:<14} rank={avg_rank[n]:.2f}  MCS={mcs_counts[n]}/{len(tickers)}  "
                  f"beats_LogHAR={beats_loghar[n]}")
        if enrich_gain:
            print(f"  LightGBM plain->enriched mean QLIKE gain: {np.mean(enrich_gain):+.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ml.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'ml.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-reps", type=int, default=1000, dest="mcs_reps")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refit-every", type=int, default=132, dest="refit_every")
    args = parser.parse_args()
    run_all(horizons=tuple(args.horizons), mcs_reps=args.mcs_reps, seed=args.seed,
            refit_every=args.refit_every)


if __name__ == "__main__":
    main()
