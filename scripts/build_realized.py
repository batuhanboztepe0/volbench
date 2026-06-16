"""Rebuild ``data/oxford_realized.csv`` from the original Oxford-Man library.

The bundled subset is committed for offline reproducibility, but this script
documents and reproduces exactly how it was made: download the final public
release of the Oxford-Man Institute Realized Library (retrieved from the
Internet Archive, since the Institute's site is gone), keep the requested
indices and the columns the benchmark uses, and write the compact CSV.

Usage
-----
    python scripts/build_realized.py [--tickers .SPX .FTSE ...] [--out PATH]
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.data import DEFAULT_TICKERS  # noqa: E402

# Internet Archive snapshot of the official ZIP (the live site shut down in 2022).
WAYBACK_URL = (
    "http://web.archive.org/web/20220301022212id_/"
    "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
)
KEEP_COLUMNS = [
    "rv5", "rv5_ss", "bv", "medrv", "rk_parzen", "rsv",
    "close_price", "open_to_close", "nobs",
]
DEFAULT_OUT = ROOT / "data" / "oxford_realized.csv"


def fetch_raw() -> pd.DataFrame:
    """Download and parse the full Oxford-Man realized library CSV."""
    print(f"Downloading Oxford-Man library from\n  {WAYBACK_URL}")
    with urllib.request.urlopen(WAYBACK_URL, timeout=300) as resp:  # noqa: S310
        blob = resp.read()
    print(f"  got {len(blob) / 1e6:.1f} MB; unzipping ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh)
    df = df.rename(columns={df.columns[0]: "date"})
    return df


def build(tickers: list[str], out: Path) -> None:
    """Filter the raw library to ``tickers`` and write the compact CSV."""
    df = fetch_raw()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    sub = df[df["Symbol"].isin(tickers)][["date", "Symbol", *KEEP_COLUMNS]].copy()
    sub = sub.rename(columns={"Symbol": "symbol"})
    sub = sub[sub["rv5"].notna() & (sub["rv5"] > 0)]
    sub = sub.sort_values(["symbol", "date"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, float_format="%.10g")
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    for tk in tickers:
        s = sub[sub.symbol == tk]
        if len(s):
            print(f"  {tk:<10} n={len(s):>5}  {s.date.min().date()}..{s.date.max().date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.tickers, args.out)


if __name__ == "__main__":
    main()
