"""Break-HAR test on VOLARE futures (the degradation arm of pre-registered H1).

Runs the pre-registered HAR protocol on VOLARE's 13 futures (fetched by
``build_volare.py --fetch futures`` → ``data/volare_futures_realized.csv``),
grouped by the sub-class whose H1 clause it probes:

* **rates** — FV (5y) + TY (10y) Treasury futures → H1 clause iv (auction/FOMC
  *calendar* structure). The single best break-HAR bet.
* **commodity** — CL/NG (energy), GC/SI/HG (metals), C/S/W (ags) → clause iv
  (event-driven persistence; jumps).
* **equity_index** — ES, NQ → the dominance *control* (HAR should still win).
* **fx** — EU (Euro) → predicted-dominance class.

For each contract and horizon h∈{1,5,22} it scores the full model set (baselines +
HAR family incl. LogSHAR + HARQ) with the 90% MCS vs LogHAR and records the §6
verdict (dominates / degrades / competitive), Q1 (HARQ transfer) and Q2 (LogSHAR).
A contract where the HAR family is **excluded / displaced** is the first cell of
the §6.4 cross-asset transfer matrix. Reuses the crypto runner's helpers (DRY).

Writes ``results/volare_futures.json`` + per-horizon QLIKE tables.

Usage
-----
    python scripts/run_volare_futures.py                  # registered run (B=10000)
    python scripts/run_volare_futures.py --symbols FV TY --mcs-reps 500   # smoke
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
    BENCHMARK,
    HORIZONS,
    Q1_MODEL,
    Q2_MODEL,
    _dm_record,
    _verdict,
    build_suite,
)

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import AssetClassConfig, load_realized_panel  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "volare_futures_realized.csv"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"

# Sub-class of each VOLARE futures contract = which H1 clause it probes.
SUBCLASS: dict[str, str] = {
    "FV": "rates", "TY": "rates",
    "CL": "commodity", "NG": "commodity", "GC": "commodity", "SI": "commodity",
    "HG": "commodity", "C": "commodity", "S": "commodity", "W": "commodity",
    "ES": "equity_index", "NQ": "equity_index",
    "EU": "fx",
}

# Wide vol band: Treasuries (~4-7% ann) to natural gas (~40-80% ann) share one class.
VOLARE_FUTURES_CONFIG = AssetClassConfig(
    name="volare-futures",
    days_per_year=252,
    ann_vol_band=(0.02, 1.50),
    required_columns=("rv5", "bv", "medrv", "rk_parzen", "rsv", "rq", "close_price"),
    default_symbols=tuple(SUBCLASS),
)


def run_all(data: Path, symbols: list[str] | None, mcs_reps: int, seed: int) -> dict:
    present = list(pd.read_csv(data, usecols=["symbol"])["symbol"].unique())
    wanted = [s for s in (symbols or present) if s in present]
    ds = load_realized_panel(data, VOLARE_FUTURES_CONFIG, symbols=wanted)
    contracts = ds.tickers
    print(f"loaded {len(contracts)} futures: {contracts}")

    summary: dict = {
        "data": str(data.name), "contracts": contracts, "subclass": SUBCLASS,
        "horizons": list(HORIZONS), "benchmark": BENCHMARK, "mcs_reps": mcs_reps,
        "by_horizon": {},
    }

    for h in HORIZONS:
        print(f"\n{'=' * 72}\nVOLARE FUTURES  h = {h}\n{'=' * 72}")
        per_contract: dict[str, dict] = {}
        qlike_rows: dict[str, dict[str, float]] = {}
        for c in contracts:
            fr = ds.frame(c)
            rv = fr["rv5"].to_numpy()
            try:
                res = run_backtest(rv, horizon=h, models=build_suite(fr),
                                   mcs_reps=mcs_reps, seed=seed, benchmark=BENCHMARK)
            except ValueError as exc:
                print(f"  {c:<5} SKIP ({exc})")
                per_contract[c] = {"verdict": "insufficient-data", "subclass": SUBCLASS.get(c)}
                continue
            mcs_inc = set(res.mcs["QLIKE"].included)
            mean_q = res.mean_losses["QLIKE"]
            dm = res.dm_vs_har["QLIKE"]
            v, best = _verdict(mcs_inc, mean_q, dm)
            qlike_rows[c] = mean_q
            per_contract[c] = {
                "subclass": SUBCLASS.get(c), "verdict": v, "best": best,
                "n_test": int(len(res.origins)), "mcs": sorted(mcs_inc),
                "q1_harq_vs_loghar": _dm_record(dm, Q1_MODEL),
                "q2_logshar_vs_loghar": _dm_record(dm, Q2_MODEL),
            }
            print(f"  {c:<5} [{SUBCLASS.get(c):<12}] {v:<12} best={best:<10} "
                  f"n={len(res.origins):>4}")

        if qlike_rows:
            tbl = pd.DataFrame(qlike_rows)
            tbl["avg_QLIKE"] = tbl.mean(axis=1)
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            tbl.sort_values("avg_QLIKE").to_csv(TABLES_DIR / f"volare_futures_qlike_h{h}.csv")

        # Per-sub-class verdict tally (the transfer-matrix rows).
        by_sub: dict[str, dict[str, int]] = {}
        for d in per_contract.values():
            sub = d.get("subclass", "?")
            by_sub.setdefault(sub, {"dominates": 0, "degrades": 0, "competitive": 0,
                                    "insufficient-data": 0})
            by_sub[sub][d["verdict"]] = by_sub[sub].get(d["verdict"], 0) + 1
        summary["by_horizon"][str(h)] = {"per_contract": per_contract, "by_subclass": by_sub}
        print(f"\n  h={h} verdicts by sub-class:")
        for sub, counts in by_sub.items():
            hit = "  <-- HAR BREAKS" if counts.get("degrades", 0) else ""
            print(f"    {sub:<13} {counts}{hit}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "volare_futures.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
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
