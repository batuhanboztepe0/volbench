"""Build ``data/crypto_realized.csv`` from Binance 5-minute klines.

Unlike the Oxford-Man equity panel, crypto exchanges expose **free intraday
bars**, so for the first time we can compute the *full* realized-measure suite
from real high-frequency data, including realized quarticity, which unlocks
HARQ on real data, and a real (not simulated) volatility signature plot.

For each coin we page through Binance 5-minute klines, group them by UTC day
(crypto trades 24/7, so a "day" is a calendar UTC day with ~288 bars and no
overnight gap), and compute the daily realized measures with
:mod:`volbench.realized`. The output is a compact daily panel, not the raw bars.

The panel is **not committed** (exchange-data redistribution terms are unclear);
this script regenerates it. Tests that need it skip when it is absent.

Usage
-----
    python scripts/build_crypto.py [--coins BTC ETH SOL BNB] [--end 2025-06-01]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.realized import (  # noqa: E402
    bipower_variation,
    median_rv,
    realized_kernel_parzen,
    realized_quarticity,
    realized_semivariance,
    realized_variance,
)

BINANCE = "https://api.binance.com/api/v3/klines"
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000
EXPECTED_BARS = 288          # 5-minute bars in a 24h day
MIN_BARS = 240               # require a reasonably complete day
DEFAULT_OUT = ROOT / "data" / "crypto_realized.csv"

# Binance trading-pair symbol and the date its 5-minute history begins.
COINS = {
    "BTC": ("BTCUSDT", "2019-01-01"),
    "ETH": ("ETHUSDT", "2019-01-01"),
    "BNB": ("BNBUSDT", "2019-01-01"),
    "SOL": ("SOLUSDT", "2020-09-01"),
}


def _get(url: str, retries: int = 5) -> list:
    """GET a Binance klines page, retrying on rate-limit / transient errors."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code in (418, 429):  # rate limited
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} retries: {url}")


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Page through 5-minute klines for one symbol over [start_ms, end_ms)."""
    rows: list = []
    cur = start_ms
    while cur < end_ms:
        url = f"{BINANCE}?symbol={symbol}&interval={INTERVAL}&startTime={cur}&endTime={end_ms}&limit=1000"
        page = _get(url)
        if not page:
            break
        rows.extend(page)
        nxt = page[-1][0] + INTERVAL_MS
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame(columns=["open_time", "close"])
    df = pd.DataFrame(rows, columns=[
        "open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tb", "tq", "ig"])
    df = df.drop_duplicates(subset="open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["c"].astype(float)
    return df[["open_time", "close"]]


def daily_measures(coin: str, bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-minute bars into a daily realized-measure panel."""
    bars = bars.copy()
    bars["date"] = bars["open_time"].dt.normalize().dt.tz_localize(None)
    records: list[dict] = []
    for day, g in bars.groupby("date"):
        price = g["close"].to_numpy()
        if price.size < MIN_BARS or np.any(price <= 0):
            continue
        r = np.diff(np.log(price))
        rsv_minus, rsv_plus = realized_semivariance(r)
        rv = realized_variance(r)
        if not np.isfinite(rv) or rv <= 0:
            continue
        records.append({
            "date": day,
            "symbol": coin,
            "rv5": rv,
            "bv": bipower_variation(r),
            "medrv": median_rv(r),
            "rk_parzen": realized_kernel_parzen(r),
            "rsv": rsv_minus,
            "rsv_plus": rsv_plus,
            "rq": realized_quarticity(r),
            "close_price": float(price[-1]),
            "open_to_close": float(np.log(price[-1] / price[0])),
            "nobs": int(r.size),
        })
    return pd.DataFrame.from_records(records)


def build(coins: list[str], out: Path, end: str) -> None:
    """Fetch and assemble the crypto realized panel."""
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    frames = []
    for coin in coins:
        symbol, start = COINS[coin]
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        print(f"  fetching {coin} ({symbol}) from {start} ...", flush=True)
        bars = fetch_klines(symbol, start_ms, end_ms)
        panel = daily_measures(coin, bars)
        print(f"    {len(bars)} bars -> {len(panel)} daily obs "
              f"({panel.date.min().date()}..{panel.date.max().date()})" if len(panel) else
              f"    {len(bars)} bars -> 0 daily obs")
        frames.append(panel)
    full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False, float_format="%.10g")
    print(f"\nWrote {out}  rows={len(full)}  ({out.stat().st_size / 1e6:.2f} MB)")
    for coin in coins:
        s = full[full.symbol == coin]
        if len(s):
            annvol = float(np.sqrt(s.rv5.mean() * 365)) * 100
            print(f"  {coin:<4} n={len(s):>5}  mean_annvol={annvol:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", nargs="+", default=list(COINS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--end", default="2025-06-01")
    args = parser.parse_args()
    build(args.coins, args.out, args.end)


if __name__ == "__main__":
    main()
