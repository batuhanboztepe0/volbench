"""Pre-registered cross-asset test on the EXPANDED crypto universe.

Runs the H1 protocol (docs/PREREGISTRATION.md) on the survivorship-corrected
crypto panel built by ``build_crypto_expanded.py`` (20 live large-caps + the dead
coins LUNA/FTT). For each coin and horizon h ∈ {1, 5, 22} it scores the full
pre-registered model set — baselines + the HAR family (incl. LogSHAR) + HARQ —
with the 90% MCS, benchmarked against **LogHAR** (the reference champion), and
records:

* the per-coin §6 verdict (HAR family *dominates* / *degrades* / *competitive*);
* **Q1** — does HARQ beat LogHAR (HARQ transfer)?  DM, favored sign, p;
* **Q2** — does LogSHAR's semivariance edge survive in crypto? DM vs LogHAR with
  the **sign** of the loss differential (negative ⇒ LogSHAR still better; positive
  ⇒ the equity edge has vanished/flipped — the H1 prediction);
* per-coin ``n_test`` (so the thin dead-coin windows are visible, never hidden).

This is the RV-forecast track only; the VaR/CAViaR layer stays siloed (invariant
4). Writes ``results/crypto_expanded.json`` and per-horizon QLIKE tables. The
original 4-coin ``results/crypto.json`` is left untouched for comparison.

Usage
-----
    python scripts/run_crypto_expanded.py                 # registered run (B=10000)
    python scripts/run_crypto_expanded.py --coins BTC LUNA --mcs-reps 500   # smoke
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
from volbench.data import (  # noqa: E402
    CRYPTO_CONFIG,
    CRYPTO_DAYS_PER_YEAR,
    load_realized_panel,
)
from volbench.models import (  # noqa: E402
    EWMA,
    GBRT,
    HARQ,
    AR1Log,
    HistoricalMean,
    MovingAverage,
    RandomWalk,
    har_family,
)

HORIZONS: tuple[int, ...] = (1, 5, 22)
BENCHMARK = "LogHAR"   # H1 reference champion; Q1/Q2 DM are vs this
Q1_MODEL = "HARQ"      # HARQ transfer
Q2_MODEL = "LogSHAR"   # semivariance-edge sign-flip
DEFAULT_DATA = ROOT / "data" / "crypto_expanded_realized.csv"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"

# The 8 measure-based HAR-family names (HAR, LogHAR, HAR-J, ..., LogSHAR).
HAR_FAMILY: set[str] = {m.name for m in har_family(*[np.ones(50)] * 4)}


def build_suite(fr: pd.DataFrame) -> list:
    """The pre-registered model set for one coin (baselines + HAR family + HARQ)."""
    cont = fr["cont"].to_numpy()
    jump = fr["jump"].to_numpy()
    rsv_minus = fr["rsv_minus"].to_numpy()
    rsv_plus = fr["rsv_plus"].to_numpy()
    rq = fr["rq"].to_numpy()
    baselines = [RandomWalk(), HistoricalMean(), MovingAverage(), EWMA(), AR1Log(), GBRT()]
    return baselines + har_family(cont, jump, rsv_minus, rsv_plus) + [HARQ(rq)]


def _verdict(mcs_included: set[str], mean_q: dict[str, float],
             dm: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Apply the §6 per-class decision rule at (QLIKE, this horizon).

    Returns (verdict, best_model). ``dm`` is DM-vs-LogHAR for every non-benchmark
    model; the displacement test uses LogHAR as the HAR reference (the champion),
    matching the Q1/Q2 framing.
    """
    fam_in = any(n in mcs_included for n in HAR_FAMILY)
    best = min(mean_q, key=lambda k: mean_q[k])
    if not fam_in:
        return "degrades", best              # HAR family excluded from the MCS
    if best in HAR_FAMILY:
        return "dominates", best             # a HAR-family model is single-best
    d = dm.get(best)                          # best is a non-HAR variance forecaster
    if d and np.isfinite(d["p_value"]) and d["p_value"] < 0.05 and d["favored"] < 0:
        return "degrades", best              # displaced as single-best, DM-significant
    return "competitive", best               # in MCS, not single-best, DM n.s.


def _dm_record(dm: dict[str, dict[str, float]], model: str) -> dict | None:
    d = dm.get(model)
    if d is None:
        return None
    return {
        "mean_diff": float(d["mean_diff"]),   # model - LogHAR (negative ⇒ model better)
        "p_value": float(d["p_value"]),
        "favored": int(d["favored"]),         # -1 model, +1 LogHAR, 0 tie
        "beats_loghar": bool(np.isfinite(d["p_value"]) and d["p_value"] < 0.05 and d["favored"] < 0),
    }


def run_all(data: Path, coins: list[str] | None, mcs_reps: int, seed: int) -> dict:
    present = list(pd.read_csv(data, usecols=["symbol"])["symbol"].unique())
    wanted = [c for c in (coins or present) if c in present]
    ds = load_realized_panel(data, CRYPTO_CONFIG, symbols=wanted)
    coins = ds.tickers
    print(f"loaded {len(coins)} coins: {coins}")

    summary: dict = {
        "data": str(data.name),
        "coins": coins,
        "horizons": list(HORIZONS),
        "benchmark": BENCHMARK,
        "annualisation": CRYPTO_DAYS_PER_YEAR,
        "mcs_reps": mcs_reps,
        "by_horizon": {},
    }

    for h in HORIZONS:
        print(f"\n{'=' * 70}\nEXPANDED CRYPTO  h = {h}\n{'=' * 70}")
        per_coin: dict[str, dict] = {}
        qlike_rows: dict[str, dict[str, float]] = {}
        for c in coins:
            fr = ds.frame(c)
            rv = fr["rv5"].to_numpy()
            suite = build_suite(fr)
            try:
                res = run_backtest(rv, horizon=h, models=suite, mcs_reps=mcs_reps,
                                   seed=seed, benchmark=BENCHMARK)
            except ValueError as exc:
                print(f"  {c:<6} SKIP ({exc})")
                per_coin[c] = {"verdict": "insufficient-data", "reason": str(exc)}
                continue
            mcs_inc = set(res.mcs["QLIKE"].included)
            mean_q = res.mean_losses["QLIKE"]
            dm = res.dm_vs_har["QLIKE"]
            v, best = _verdict(mcs_inc, mean_q, dm)
            qlike_rows[c] = mean_q
            per_coin[c] = {
                "verdict": v,
                "best": best,
                "n_test": int(len(res.origins)),
                "har_family_in_mcs": any(n in mcs_inc for n in HAR_FAMILY),
                "loghar_in_mcs": BENCHMARK in mcs_inc,
                "mcs": sorted(mcs_inc),
                "q1_harq_vs_loghar": _dm_record(dm, Q1_MODEL),
                "q2_logshar_vs_loghar": _dm_record(dm, Q2_MODEL),
            }
            q2 = per_coin[c]["q2_logshar_vs_loghar"]
            q2sign = ("LogSHAR better" if q2 and q2["mean_diff"] < 0 else "LogHAR better") if q2 else "n/a"
            print(f"  {c:<6} {v:<12} best={best:<10} n={len(res.origins):>4} "
                  f"Q1(HARQ beats)={per_coin[c]['q1_harq_vs_loghar'] and per_coin[c]['q1_harq_vs_loghar']['beats_loghar']}"
                  f"  Q2={q2sign}")

        verdicts = [d["verdict"] for d in per_coin.values()]
        counts = {v: verdicts.count(v) for v in
                  ("dominates", "degrades", "competitive", "insufficient-data")}
        scored = [c for c in coins if per_coin[c]["verdict"] != "insufficient-data"]
        harq_beats = sum(1 for c in scored if per_coin[c]["q1_harq_vs_loghar"]
                         and per_coin[c]["q1_harq_vs_loghar"]["beats_loghar"])
        logshar_beats = sum(1 for c in scored if per_coin[c]["q2_logshar_vs_loghar"]
                            and per_coin[c]["q2_logshar_vs_loghar"]["beats_loghar"])
        logshar_loghar_better = sum(
            1 for c in scored if per_coin[c]["q2_logshar_vs_loghar"]
            and per_coin[c]["q2_logshar_vs_loghar"]["mean_diff"] > 0)
        class_verdict = max(("dominates", "degrades", "competitive"),
                            key=lambda v: counts[v]) if scored else "n/a"

        if qlike_rows:
            tbl = pd.DataFrame(qlike_rows)
            tbl["avg_QLIKE"] = tbl.mean(axis=1)
            tbl = tbl.sort_values("avg_QLIKE")
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            tbl.to_csv(TABLES_DIR / f"crypto_expanded_qlike_h{h}.csv")

        summary["by_horizon"][str(h)] = {
            "per_coin": per_coin,
            "verdict_counts": counts,
            "class_verdict_majority": class_verdict,
            "q1_harq_beats_loghar": f"{harq_beats}/{len(scored)}",
            "q2_logshar_beats_loghar": f"{logshar_beats}/{len(scored)}",
            "q2_loghar_better_than_logshar": f"{logshar_loghar_better}/{len(scored)}",
        }
        print(f"\n  h={h} class verdict (majority): {class_verdict}  counts={counts}")
        print(f"  Q1 HARQ beats LogHAR: {harq_beats}/{len(scored)}  "
              f"(H1 predicts ~0 — HARQ does not transfer)")
        print(f"  Q2 LogSHAR beats LogHAR: {logshar_beats}/{len(scored)}; "
              f"LogHAR better than LogSHAR: {logshar_loghar_better}/{len(scored)} "
              f"(H1 predicts the equity LogSHAR edge vanishes/flips)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "crypto_expanded.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--coins", nargs="+", default=None)
    parser.add_argument("--mcs-reps", type=int, default=10000, dest="mcs_reps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_all(args.data, args.coins, args.mcs_reps, args.seed)


if __name__ == "__main__":
    main()
