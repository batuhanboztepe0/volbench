"""Fetch realized-measure panels from the VOLARE token-authenticated REST API.

VOLARE (VOLatility Archive for Realized Estimates) publishes precomputed daily
realized-volatility measures for stocks, FX and futures, free under **CC-BY 4.0**.
If you use this data you MUST cite:

    Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., &
    Spagnolo, F. (2026). VOLatility Archive for Realized Estimates (VOLARE).
    arXiv:2602.19732. https://doi.org/10.48550/arXiv.2602.19732

Access is a token-authenticated REST API (reverse-engineered from the site's
front-end bundle; the endpoints below are confirmed live: ``/api/health`` → 200,
``/api/financial-data`` → 401 without a token). Authenticate with your registered
credentials, then download per-asset-class packages.

    POST /api/auth/token                                      -> {access_token}
    GET  /api/financial-data                                  (list/query; auth)
    GET  /api/financial-data/files                            (what you can pull)
    GET  /api/financial-data/limits                           (quota)
    GET  /api/financial-data/volatility/download/pregenerated/<id>
                                                 -> {download_url, file_name}
    GET  <download_url>                          (presigned MinIO URL; no auth)

⚠️ FIRST-RUN UNCERTAINTY (confirm with ``--discover`` before trusting the rest):
the exact pregenerated package **ids**, the download **query params**, and the
delivered **column headers** must be read from a live authenticated session —
the paper gives measure acronyms (Table 12) but not the final headers verbatim.
``--discover`` prints the real ``/files`` and ``/limits`` responses so you can
fill in the ids; the converter (``--convert``) is tolerant of header variants.

Credentials come from the environment (never hard-code / commit them):
    export VOLARE_EMAIL=...     VOLARE_PASSWORD=...

Usage
-----
    python scripts/build_volare.py --discover
    python scripts/build_volare.py --download futures --pkg-id <ID>
    python scripts/build_volare.py --convert data/volare_futures.parquet --asset futures
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

# volbench panel column -> ordered candidate VOLARE source headers (first present
# wins). The exact delivered headers are uncertain until --discover; this tolerates
# the documented variants (5-min measures, rsn=downside semivar, OHLC, trade count).
_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "rv5": ("rv5", "RV5", "rv_5"),
    "bv": ("bv5", "bv", "BV5"),
    "medrv": ("medrv5", "medrv", "MedRV5"),
    "rk_parzen": ("rk_parzen", "rk", "RK"),
    "rsv": ("rsn5", "rsn", "rsv", "RSn5"),           # DOWNSIDE semivariance
    "rq": ("rq5", "rq", "RQ5"),
    "close_price": ("close_price", "C", "CC", "close"),
}
_OPEN_CANDIDATES = ("O", "OO", "open", "open_price")
_NOBS_CANDIDATES = ("nobs", "N", "NN", "n_trades")


def _pick(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_token(email: str, password: str) -> str:
    """Authenticate (OAuth2 password flow) and return the bearer access token."""
    r = requests.post(
        f"{BASE}/auth/token",
        data={"username": email, "password": password},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    token = payload.get("access_token") or payload.get("token")
    if not token:
        raise RuntimeError(f"no access_token in auth response: {payload}")
    return str(token)


def _auth_get(token: str, path: str) -> dict:
    r = requests.get(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def discover(token: str) -> None:
    """Print what the account can download — use this to find the package ids."""
    for path in ("/financial-data/files", "/financial-data/limits", "/financial-data"):
        print(f"\n=== GET {path} ===", flush=True)
        try:
            print(json.dumps(_auth_get(token, path), indent=2)[:4000])
        except requests.HTTPError as exc:
            print(f"  (HTTP {exc.response.status_code}: {exc.response.text[:200]})")


def download_pregenerated(token: str, pkg_id: str, dest: Path) -> None:
    """Download one pregenerated asset-class package to ``dest`` (parquet)."""
    info = _auth_get(token, f"/financial-data/volatility/download/pregenerated/{pkg_id}")
    url = info.get("download_url")
    if not url:
        raise RuntimeError(f"no download_url in response: {info}")
    print(f"  presigned url -> {url[:80]}...", flush=True)
    blob = requests.get(url, timeout=300)
    blob.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob.content)
    print(f"  wrote {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")


def to_volbench_csv(parquet: Path, asset: str, out: Path | None = None) -> Path:
    """Convert a VOLARE parquet into a volbench-schema CSV (long format).

    Maps VOLARE's headers to volbench's panel columns (``rv5``, ``bv``, ``medrv``,
    ``rk_parzen``, ``rsv``, ``rq``, ``close_price``, ``open_to_close``, ``nobs``),
    deriving ``open_to_close`` from open/close. Header matching is tolerant
    (see ``_COLUMN_CANDIDATES``); anything unmatched is reported, not guessed.
    """
    df = pd.read_parquet(parquet)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError(f"expected date+symbol columns; got {list(df.columns)[:20]}")
    out_df = pd.DataFrame({"date": pd.to_datetime(df["date"]), "symbol": df["symbol"]})

    missing: list[str] = []
    for target, cands in _COLUMN_CANDIDATES.items():
        src = _pick(df, cands)
        if src is None:
            missing.append(target)
        else:
            out_df[target] = pd.to_numeric(df[src], errors="coerce")

    open_col = _pick(df, _OPEN_CANDIDATES)
    if open_col is not None and "close_price" in out_df:
        out_df["open_to_close"] = np.log(out_df["close_price"] / pd.to_numeric(df[open_col]))
    nobs_col = _pick(df, _NOBS_CANDIDATES)
    out_df["nobs"] = pd.to_numeric(df[nobs_col], errors="coerce") if nobs_col else float("nan")

    if missing:
        print(f"  ⚠️ unmatched volbench columns {missing} — inspect the parquet headers: "
              f"{sorted(df.columns)[:25]}", file=sys.stderr)
    out = out or (DATA / f"volare_{asset}_realized.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values(["symbol", "date"]).to_csv(out, index=False, float_format="%.10g")
    print(f"  wrote {out}  rows={len(out_df)}  symbols={out_df.symbol.nunique()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discover", action="store_true", help="list downloadable packages")
    ap.add_argument("--download", metavar="ASSET", help="asset class to download (futures/stocks/fx)")
    ap.add_argument("--pkg-id", help="pregenerated package id (from --discover)")
    ap.add_argument("--convert", metavar="PARQUET", type=Path, help="convert a parquet to volbench CSV")
    ap.add_argument("--asset", default="futures", help="asset label for --convert/--download output")
    args = ap.parse_args()

    if args.convert:  # offline; no auth needed
        to_volbench_csv(args.convert, args.asset)
        return

    email, password = os.environ.get("VOLARE_EMAIL"), os.environ.get("VOLARE_PASSWORD")
    if not email or not password:
        sys.exit("set VOLARE_EMAIL and VOLARE_PASSWORD env vars first")
    token = get_token(email, password)
    print("authenticated ✓")

    if args.discover:
        discover(token)
    elif args.download:
        if not args.pkg_id:
            sys.exit("--download needs --pkg-id (run --discover to find it)")
        download_pregenerated(token, args.pkg_id, DATA / f"volare_{args.download}.parquet")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
