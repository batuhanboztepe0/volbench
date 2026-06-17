"""Rebuild ``data/vix.csv`` from FRED (public-domain CBOE VIX series).

The variance-risk-premium study needs an implied-volatility series. The CBOE VIX
is published by the St. Louis Fed as series ``VIXCLS`` and is public domain, so
unlike the realized-measure panel it is committed directly; this script just
documents and reproduces it.

Usage
-----
    python scripts/build_vix.py [--out PATH] [--start 2000-01-01] [--end 2022-12-31]
"""

from __future__ import annotations

import argparse
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
DEFAULT_OUT = ROOT / "data" / "vix.csv"


def build(out: Path, start: str, end: str) -> None:
    """Download VIXCLS from FRED and write a clean date/vix CSV."""
    print(f"Downloading VIX (VIXCLS) from {FRED_URL}")
    with urllib.request.urlopen(FRED_URL, timeout=120) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(StringIO(raw))
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna()
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df.sort_values("date").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"Wrote {out}  rows={len(df)}  {df.date.min().date()}..{df.date.max().date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2022-12-31")
    args = parser.parse_args()
    build(args.out, args.start, args.end)


if __name__ == "__main__":
    main()
