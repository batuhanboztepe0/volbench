"""Fetch realized-measure panels from the VOLARE token-authenticated REST API.

VOLARE (VOLatility Archive for Realized Estimates) publishes precomputed daily
realized-volatility measures for stocks, FX and futures, free under **CC-BY 4.0**.
If you use this data you MUST cite:

    Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., &
    Spagnolo, F. (2026). VOLatility Archive for Realized Estimates (VOLARE).
    arXiv:2602.19732. https://doi.org/10.48550/arXiv.2602.19732

Access is a token-authenticated REST API. Confirmed live from a real account:

    POST /api/auth/token                  -> {access_token}   (OAuth2 password flow)
    GET  /api/financial-data/limits       -> per-class quota (stocks 109, forex 13,
                                             futures 13, ETFs 1)
    GET  /api/financial-data?asset_type=… -> JSON records (one row per date+symbol)

Each record carries: observation_date, symbol, asset_type, open/high/low/close_price,
volume, trades, and the realized measures rv1/rv5(/_ss), bv1/bv5(/_ss),
rsp1/rsn1/rsp5/rsn5(/_ss) (rsp=upside, rsn=DOWNSIDE semivariance), medrv*, minrv*,
rk, rq1/rq5(/_ss), plus pv/gk/rr5 (range-based). Data is JSON — no parquet packages
(``/files`` and ``/…/pregenerated`` 404 on a base-role account).

Credentials come from the environment (never hard-code / commit them):
    export VOLARE_EMAIL=...     VOLARE_PASSWORD=...

Usage
-----
    python scripts/build_volare.py --probe futures      # envelope + symbols (no bulk pull)
    python scripts/build_volare.py --fetch futures      # -> data/volare_futures_realized.csv
    python scripts/build_volare.py --fetch stocks --start 2015-01-01 --end 2026-06-01

Pagination is confirmed: the envelope is ``{total, page, limit, has_more}`` and
``--fetch`` walks ``page`` until ``has_more`` is false. The 13 futures include two
US Treasury futures — ``FV`` (5-yr) and ``TY`` (10-yr) — so the bond-futures
break-HAR test runs on VOLARE alone. The optional ``--start/--end`` date-filter
param names are still inferred (the full panel paginates without them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://volare.unime.it/api"
DATA = ROOT / "data"

# Auto-load credentials from a local .env (gitignored) so they need not be
# re-exported every shell session. Optional dependency: works without it too.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Confirmed VOLARE header -> volbench panel column (5-min measures; rsn=downside).
_RENAME = {
    "observation_date": "date",
    "bv5": "bv",
    "medrv5": "medrv",
    "rk": "rk_parzen",
    "rsn5": "rsv",       # DOWNSIDE realized semivariance
    "rq5": "rq",
    "trades": "nobs",
}
_VOLBENCH_COLS = [
    "date", "symbol", "rv5", "bv", "medrv", "rk_parzen", "rsv", "rq",
    "close_price", "open_to_close", "nobs",
]
_DEFAULT_LIMIT = 20000  # API caps per-page; envelope carries page/total/has_more


def get_token(email: str, password: str) -> str:
    """Authenticate (OAuth2 password flow) and return the bearer access token."""
    r = requests.post(
        f"{BASE}/auth/token", data={"username": email, "password": password}, timeout=30
    )
    r.raise_for_status()
    payload = r.json()
    token = payload.get("access_token") or payload.get("token")
    if not token:
        raise RuntimeError(f"no access_token in auth response: {payload}")
    return str(token)


def _get(token: str, path: str) -> dict:
    r = requests.get(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    return r.json()


def probe(token: str, asset_type: str) -> None:
    """Print the response envelope + the distinct symbols for one asset class.

    Reveals the pagination/metadata fields and the actual instrument universe
    (e.g. whether ``futures`` includes bond/rate contracts).
    """
    print("\n=== GET /financial-data/limits ===")
    print(json.dumps(_get(token, "/financial-data/limits"), indent=2))
    env = _get(token, f"/financial-data?asset_type={asset_type}&limit=20000")
    data = env.get("data", env if isinstance(env, list) else [])
    meta = {k: v for k, v in env.items() if k != "data"} if isinstance(env, dict) else {}
    syms = sorted({r.get("symbol") for r in data})
    dates = sorted({r.get("observation_date") for r in data})
    print(f"\n=== GET /financial-data?asset_type={asset_type} ===")
    print(f"  envelope (non-data keys): {json.dumps(meta)[:800]}")
    print(f"  records this page: {len(data)}")
    print(f"  distinct symbols ({len(syms)}): {syms}")
    if dates:
        print(f"  date span this page: {dates[0]} .. {dates[-1]}")
    if data:
        print(f"  example record keys: {sorted(data[0])}")


def fetch_asset(
    token: str, asset_type: str, start: str | None, end: str | None, limit: int = _DEFAULT_LIMIT
) -> list[dict]:
    """Paginate /financial-data for one asset class and return all records.

    Uses the confirmed envelope: ``{total, page, limit, has_more}`` — follow
    ``page`` until ``has_more`` is false.
    """
    records: list[dict] = []
    page_num = 1
    while True:
        q = f"asset_type={asset_type}&page={page_num}&limit={limit}"
        if start:
            q += f"&start_date={start}"
        if end:
            q += f"&end_date={end}"
        env = _get(token, f"/financial-data?{q}")
        rows = env.get("data", []) if isinstance(env, dict) else env
        if not rows:
            break
        records.extend(rows)
        total = env.get("total", "?") if isinstance(env, dict) else "?"
        print(f"  page {page_num}: +{len(rows)} -> {len(records)}/{total}", flush=True)
        if not (isinstance(env, dict) and env.get("has_more")):
            break
        page_num += 1
    return records


def records_to_csv(records: list[dict], asset: str, out: Path | None = None) -> Path:
    """Map VOLARE records to a volbench-schema CSV (long format)."""
    if not records:
        raise ValueError("no records to write")
    df = pd.DataFrame.from_records(records).rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    if {"close_price", "open_price"} <= set(df.columns):
        df["open_to_close"] = np.log(df["close_price"] / df["open_price"])
    missing = [c for c in _VOLBENCH_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠️ missing volbench columns {missing}; available: {sorted(df.columns)}",
              file=sys.stderr)
    cols = [c for c in _VOLBENCH_COLS if c in df.columns]
    out = out or (DATA / f"volare_{asset}_realized.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values(["symbol", "date"]).to_csv(out, index=False, float_format="%.10g")
    print(f"  wrote {out}  rows={len(df)}  symbols={df.symbol.nunique()}  "
          f"span={df.date.min().date()}..{df.date.max().date()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", metavar="ASSET", help="show envelope + symbols for an asset class")
    ap.add_argument("--fetch", metavar="ASSET", help="download an asset class (futures/stocks/forex/ETFs)")
    ap.add_argument("--start", help="start date YYYY-MM-DD (optional)")
    ap.add_argument("--end", help="end date YYYY-MM-DD (optional)")
    args = ap.parse_args()

    email, password = os.environ.get("VOLARE_EMAIL"), os.environ.get("VOLARE_PASSWORD")
    if not email or not password:
        sys.exit("set VOLARE_EMAIL and VOLARE_PASSWORD env vars first")
    token = get_token(email, password)
    print("authenticated ✓")

    if args.probe:
        probe(token, args.probe)
    elif args.fetch:
        records = fetch_asset(token, args.fetch, args.start, args.end)
        records_to_csv(records, args.fetch)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
