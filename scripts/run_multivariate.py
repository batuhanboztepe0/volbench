"""Volatility-spillover benchmark: does adding peer indices help?

For each of the 8 Oxford-Man indices as the target (peers = the other 7),
runs :func:`~volbench.multivariate.spillover_backtest` at horizon=1 (and 5),
prints a summary table, and writes ``results/multivariate.json``.

Headline question: does adding peer indices' lagged daily log-RV improve
a target index's realized-variance forecast out-of-sample, after honest
Diebold-Mariano and Model Confidence Set testing?

Usage
-----
    PYTHONPATH=src python3 scripts/run_multivariate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.data import load_oxford_rv  # noqa: E402
from volbench.multivariate import spillover_backtest  # noqa: E402

HORIZONS: tuple[int, ...] = (1, 5)
MCS_REPS: int = 500
SEED: int = 0
RESULTS_DIR = ROOT / "results"


def _fmt(x: object, decimals: int = 6) -> str:
    """Format a float or fallback to str."""
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def run_all() -> dict:
    """Run spillover backtest for every ticker as target at each horizon."""
    ds = load_oxford_rv()
    tickers = ds.tickers
    summary: dict = {"horizons": list(HORIZONS), "by_horizon": {}}

    for h in HORIZONS:
        print(f"\n{'=' * 72}")
        print(f"HORIZON h = {h}")
        print(f"{'=' * 72}")

        # Header row.
        header = (
            f"{'Target':<12} {'LogHAR QLIKE':>14} {'CrossHAR QLIKE':>16}"
            f" {'Improv %':>10} {'DM p-val':>10} {'CrossHAR wins':>14}"
        )
        print(header)
        print("-" * 72)

        horizon_results: list[dict] = []
        n_wins = 0

        for target in tickers:
            peers = [t for t in tickers if t != target]
            res = spillover_backtest(
                ds, target, peers, horizon=h, mcs_reps=MCS_REPS, seed=SEED
            )
            horizon_results.append(res)

            loghar_q = res["mean_qlike"].get("LogHAR", float("nan"))
            crosshar_q = res["mean_qlike"].get("CrossHAR", float("nan"))
            pct = res["pct_improvement"]
            dm = res["dm_crosshar_vs_loghar"]
            pval = dm.get("p_value", float("nan")) if dm else float("nan")
            beats = res["crosshar_beats_loghar"]
            if beats:
                n_wins += 1

            print(
                f"{target:<12} {_fmt(loghar_q):>14} {_fmt(crosshar_q):>16}"
                f" {_fmt(pct, 2):>10} {_fmt(pval, 4):>10} {'YES' if beats else 'no':>14}"
            )

        print("-" * 72)
        print(
            f"CrossHAR significantly beats LogHAR (QLIKE, DM p<0.10) in"
            f" {n_wins}/{len(tickers)} indices at h={h}"
        )

        summary["by_horizon"][str(h)] = horizon_results

    return summary


def main() -> None:
    """Entry point."""
    summary = run_all()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "multivariate.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults written to {out_path}")

    # Headline finding.
    h1 = summary["by_horizon"].get("1", [])
    wins_h1 = sum(1 for r in h1 if r.get("crosshar_beats_loghar"))
    print(
        f"\nHeadline: At h=1, CrossHAR (peers' lagged RV) significantly beats"
        f" LogHAR (own-index only) for {wins_h1}/{len(h1)} of the 8 indices."
    )
    if wins_h1 == 0:
        print("  -> Peer spillover adds NO statistically significant predictive value.")
    elif wins_h1 >= len(h1) // 2:
        print("  -> Peer spillover shows broad statistically significant improvement.")
    else:
        print("  -> Mixed evidence: spillover helps for some indices only.")


if __name__ == "__main__":
    main()
