"""Build ``data/crypto_expanded_realized.csv`` from Binance Vision 5-minute klines.

This is the *expanded-universe* crypto builder for the cross-asset study
(pre-registered §3 amendment). Unlike ``build_crypto.py``, which uses
the live REST API and therefore can only see *currently-listed* pairs; this
script reads the **Binance Vision** public archive (``data.binance.vision``),
which **retains delisted symbols**. That is what makes the survivorship
correction possible: dead coins (LUNA pre-collapse, FTT pre-collapse) are fetched
from the same bucket as the live large-caps.

The realized-measure aggregation is reused verbatim from ``build_crypto.daily_measures``
so the expanded panel is computed identically to the original Track-3 panel; only
the data *source* differs. Output is a compact daily panel, not raw bars, and is
**not committed** (exchange-data redistribution terms are unclear).

Timestamp note: Binance Vision switched the kline ``open_time`` field from
milliseconds to microseconds in 2025; this script normalises both to ms.

Usage
-----
    python scripts/build_crypto_expanded.py [--coins BTC ETH ...] [--end 2025-06-01]
    python scripts/build_crypto_expanded.py --coins LUNA FTT   # just the dead coins
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_crypto import daily_measures  # noqa: E402  (reuse the aggregation)

BINANCE_VISION = "https://data.binance.vision/data/spot/monthly/klines"
DEFAULT_OUT = ROOT / "data" / "crypto_expanded_realized.csv"
_US_THRESHOLD = 1e15  # open_time above this is microseconds (2025+), else milliseconds

# Pre-registered expanded crypto universe (§3 / §9 amendment).
# coin -> (Binance symbol, first month to try, delisting cutoff or None for live).
# Generous start months are fine: months with no Vision file are skipped (404).
REGISTRY: dict[str, tuple[str, str, str | None]] = {
    # --- original Track-3 four (re-fetched here from Vision for one consistent source) ---
    "BTC": ("BTCUSDT", "2019-01-01", None),
    "ETH": ("ETHUSDT", "2019-01-01", None),
    "BNB": ("BNBUSDT", "2019-01-01", None),
    "SOL": ("SOLUSDT", "2020-08-01", None),
    # --- expanded live large-caps: top-20 by ~2022-01-01 market cap with Binance USDT 5m ---
    "XRP": ("XRPUSDT", "2019-01-01", None),
    "ADA": ("ADAUSDT", "2019-01-01", None),
    "DOGE": ("DOGEUSDT", "2019-07-01", None),
    "AVAX": ("AVAXUSDT", "2020-09-01", None),
    "DOT": ("DOTUSDT", "2020-08-01", None),
    "TRX": ("TRXUSDT", "2019-01-01", None),
    "LINK": ("LINKUSDT", "2019-01-01", None),
    "MATIC": ("MATICUSDT", "2019-04-01", None),
    "LTC": ("LTCUSDT", "2019-01-01", None),
    "BCH": ("BCHUSDT", "2019-01-01", None),
    "ATOM": ("ATOMUSDT", "2019-04-01", None),
    "XLM": ("XLMUSDT", "2019-01-01", None),
    "ETC": ("ETCUSDT", "2019-01-01", None),
    "ALGO": ("ALGOUSDT", "2019-06-01", None),
    "VET": ("VETUSDT", "2019-01-01", None),
    "FIL": ("FILUSDT", "2020-10-01", None),
    # --- dead coins for the survivorship correction (Vision retains the delisted history) ---
    "LUNA": ("LUNAUSDT", "2020-08-01", "2022-05-13"),  # Terra collapse, May 2022
    "FTT": ("FTTUSDT", "2020-08-01", "2022-12-01"),    # FTX collapse, Nov 2022
}
EXPANDED_COINS = list(REGISTRY)


def _get_bytes(url: str, retries: int = 4) -> bytes | None:
    """GET raw bytes; return ``None`` on 404 (month not in the archive)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (418, 429):  # rate limited
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} retries: {url}")


def fetch_klines_vision(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download monthly 5-minute klines from Binance Vision over [start_ms, end_ms).

    Returns a ``[open_time, close]`` frame matching ``build_crypto.fetch_klines``
    so the existing :func:`daily_measures` aggregation applies unchanged.
    """
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    months = pd.period_range(start.tz_localize(None).normalize(),
                             end.tz_localize(None).normalize(), freq="M")
    times: list[np.ndarray] = []
    closes: list[np.ndarray] = []
    for p in months:
        ym = f"{p.year:04d}-{p.month:02d}"
        url = f"{BINANCE_VISION}/{symbol}/5m/{symbol}-5m-{ym}.zip"
        data = _get_bytes(url)
        if data is None:
            continue
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open(z.namelist()[0]) as f:
                raw = pd.read_csv(f, header=None, usecols=[0, 4])
        times.append(raw[0].to_numpy())
        closes.append(raw[4].to_numpy(dtype=float))
        time.sleep(0.05)
    if not times:
        return pd.DataFrame(columns=["open_time", "close"])
    ot = np.concatenate(times).astype("int64")
    cl = np.concatenate(closes)
    # Normalise the 2025 microsecond switch: anything above the threshold is µs.
    ot_ms = np.where(ot > _US_THRESHOLD, ot // 1000, ot)
    df = pd.DataFrame({"open_time": ot_ms, "close": cl}).drop_duplicates(subset="open_time")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] < end_ms)]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time", "close"]].sort_values("open_time").reset_index(drop=True)


def build(coins: list[str], out: Path, end: str) -> None:
    """Fetch (from Vision) and assemble the expanded crypto realized panel."""
    end_ts = pd.Timestamp(end, tz="UTC")
    frames = []
    for coin in coins:
        symbol, start, dead_end = REGISTRY[coin]
        coin_end_ts = min(end_ts, pd.Timestamp(dead_end, tz="UTC")) if dead_end else end_ts
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(coin_end_ts.timestamp() * 1000)
        tag = "dead" if dead_end else "live"
        print(f"  fetching {coin} ({symbol}, {tag}) {start}..{coin_end_ts.date()} ...", flush=True)
        bars = fetch_klines_vision(symbol, start_ms, end_ms)
        panel = daily_measures(coin, bars)
        if len(panel):
            print(f"    {len(bars)} bars -> {len(panel)} daily obs "
                  f"({panel.date.min().date()}..{panel.date.max().date()})")
        else:
            print(f"    {len(bars)} bars -> 0 daily obs")
        frames.append(panel)
    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False, float_format="%.10g")
    print(f"\nWrote {out}  rows={len(full)}  ({out.stat().st_size / 1e6:.2f} MB)")
    for coin in coins:
        s = full[full.symbol == coin]
        if len(s):
            annvol = float(np.sqrt(s.rv5.mean() * 365)) * 100
            print(f"  {coin:<5} n={len(s):>5}  mean_annvol={annvol:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", nargs="+", default=EXPANDED_COINS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--end", default="2025-06-01")
    args = parser.parse_args()
    build(args.coins, args.out, args.end)


if __name__ == "__main__":
    main()
