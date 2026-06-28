"""Pre-registered HAR protocol on VOLARE's 13 FX pairs (confirmatory FX class).

Mirrors ``run_volare_futures.py`` for the FX arm (pre-registered §9, 2026-06-20). Data fetched by
``build_volare.py --fetch forex`` -> ``data/volare_forex_realized.csv``. Pairs are
grouped by the H1 clause each probes:

* **major**: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD → the
  dominance prediction (mature, deep, calendar-synchronized FX).
* **secondary_em**: USDKRW, USDPLN, ZARUSD, USDNOK, USDSEK, USDSGD → a within-class
  probe of H1 clause (iii) microstructure-noise / thinner liquidity.

For each pair and horizon h∈{1,5,22} it scores the full registered model set
(baselines + HAR family incl. LogSHAR + HARQ + ARFIMA) with the 90% MCS vs LogHAR
and records the §6 verdict, Q1 (HARQ transfer) and Q2 (LogSHAR). Writes
``results/volare_fx.json`` + per-horizon QLIKE tables.

Usage
-----
    python scripts/run_volare_fx.py                       # registered run (B=10000)
    python scripts/run_volare_fx.py --symbols EURUSD ZARUSD --mcs-reps 500   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_crypto_expanded import (  # noqa: E402  (reuse the MCS/verdict/Q1/Q2 logic)
    ALPHA_SENS,
    BENCHMARK,
    HORIZONS,
    Q1_MODEL,
    Q2_MODEL,
    _dm_pair,
    _dm_record,
    _mcs_at,
    _verdict,
    build_suite,
)

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import AssetClassConfig, load_realized_panel  # noqa: E402
from volbench.models import ARFIMALog  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "volare_forex_realized.csv"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"

# Sub-class of each VOLARE FX pair = which H1 clause it probes (pre-registered §9).
SUBCLASS: dict[str, str] = {
    "EURUSD": "major", "GBPUSD": "major", "USDJPY": "major", "USDCHF": "major",
    "USDCAD": "major", "AUDUSD": "major", "NZDUSD": "major",
    "USDKRW": "secondary_em", "USDPLN": "secondary_em", "ZARUSD": "secondary_em",
    "USDNOK": "secondary_em", "USDSEK": "secondary_em", "USDSGD": "secondary_em",
}

# FX annualized vol spans ~5% (USDSGD) to ~18% (ZARUSD); band gates unit errors.
VOLARE_FX_CONFIG = AssetClassConfig(
    name="volare-fx",
    days_per_year=252,
    ann_vol_band=(0.02, 0.60),
    required_columns=("rv5", "bv", "medrv", "rk_parzen", "rsv", "rq", "close_price"),
    default_symbols=tuple(SUBCLASS),
)


def run_all(data: Path, symbols: list[str] | None, mcs_reps: int, seed: int) -> dict:
    present = list(pd.read_csv(data, usecols=["symbol"])["symbol"].unique())
    wanted = [s for s in (symbols or present) if s in present]
    ds = load_realized_panel(data, VOLARE_FX_CONFIG, symbols=wanted)
    pairs = ds.tickers
    print(f"loaded {len(pairs)} FX pairs: {pairs}")

    summary: dict = {
        "data": str(data.name), "pairs": pairs, "subclass": SUBCLASS,
        "horizons": list(HORIZONS), "benchmark": BENCHMARK, "mcs_reps": mcs_reps,
        "by_horizon": {},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "volare_fx.json"

    for h in HORIZONS:
        print(f"\n{'=' * 72}\nVOLARE FX  h = {h}\n{'=' * 72}")
        per_pair: dict[str, dict] = {}
        qlike_rows: dict[str, dict[str, float]] = {}
        for c in pairs:
            fr = ds.frame(c)
            rv = fr["rv5"].to_numpy()
            suite = build_suite(fr) + [ARFIMALog()]  # full registered set (+ARFIMA)
            try:
                res = run_backtest(rv, horizon=h, models=suite,
                                   mcs_reps=mcs_reps, seed=seed, benchmark=BENCHMARK)
            except ValueError as exc:
                print(f"  {c:<7} SKIP ({exc})")
                per_pair[c] = {"verdict": "insufficient-data", "subclass": SUBCLASS.get(c)}
                continue
            mcs_inc = set(res.mcs["QLIKE"].included)
            mean_q = res.mean_losses["QLIKE"]
            dm = res.dm_vs_har["QLIKE"]
            v, best = _verdict(mcs_inc, mean_q, dm)
            mcs_inc_25 = set(_mcs_at(res, ALPHA_SENS, h, mcs_reps, seed))
            v25, _ = _verdict(mcs_inc_25, mean_q, dm)  # alpha=0.25 sensitivity
            qlike_rows[c] = mean_q
            per_pair[c] = {
                "subclass": SUBCLASS.get(c), "verdict": v, "best": best,
                "n_test": int(len(res.origins)), "mcs": sorted(mcs_inc),
                "q1_harq_vs_loghar": _dm_record(dm, Q1_MODEL),
                "q1_harq_vs_har": _dm_pair(res, "HARQ", "HAR", h),
                "q2_logshar_vs_loghar": _dm_record(dm, Q2_MODEL),
                "mcs_a25": sorted(mcs_inc_25),
                "verdict_a25": v25,
            }
            print(f"  {c:<7} [{SUBCLASS.get(c):<12}] {v:<12} best={best:<10} "
                  f"n={len(res.origins):>4}")

        if qlike_rows:
            tbl = pd.DataFrame(qlike_rows)
            tbl["avg_QLIKE"] = tbl.mean(axis=1)
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            tbl.sort_values("avg_QLIKE").to_csv(TABLES_DIR / f"volare_fx_qlike_h{h}.csv")

        by_sub: dict[str, dict[str, int]] = {}
        for d in per_pair.values():
            sub = d.get("subclass", "?")
            by_sub.setdefault(sub, {"dominates": 0, "degrades": 0, "competitive": 0,
                                    "insufficient-data": 0})
            by_sub[sub][d["verdict"]] = by_sub[sub].get(d["verdict"], 0) + 1
        summary["by_horizon"][str(h)] = {"per_contract": per_pair, "by_subclass": by_sub}
        print(f"\n  h={h} verdicts by sub-class:")
        for sub, counts in by_sub.items():
            hit = "  <-- HAR BREAKS" if counts.get("degrades", 0) else ""
            print(f"    {sub:<13} {counts}{hit}")

        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  checkpointed {out} (through h={h})")

    print(f"\nSaved {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--mcs-reps", type=int, default=10000, dest="mcs_reps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_all(args.data, args.symbols, args.mcs_reps, args.seed)


if __name__ == "__main__":
    main()
