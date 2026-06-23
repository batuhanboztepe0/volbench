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
import hashlib
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.data import DEFAULT_TICKERS  # noqa: E402

# Internet Archive snapshot of the official ZIP (the live site shut down in 2022).
# Additional Internet-Archive snapshot URLs can be appended here as mirrors;
# fetch_raw() tries each in order and raises only when all fail.
WAYBACK_URLS = [
    (
        "http://web.archive.org/web/20220301022212id_/"
        "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
    ),
]

# SHA256 of the ZIP blob.
# Fill in the hex digest after a first confirmed clean download, then every
# subsequent run will hard-fail on any corrupt or substituted blob.
# Until set, the download is logged but not verified.
EXPECTED_ZIP_SHA256: str | None = None

# SHA256 of the committed data/oxford_realized.csv — verified after build().
EXPECTED_CSV_SHA256 = "48843c9c2bdd7d37f583f14ef35a9e3642b1bfe9fa2f15612eeac2e0c802e3a7"

KEEP_COLUMNS = [
    "rv5", "rv5_ss", "bv", "medrv", "rk_parzen", "rsv",
    "close_price", "open_to_close", "nobs",
]
DEFAULT_OUT = ROOT / "data" / "oxford_realized.csv"


def fetch_raw() -> pd.DataFrame:
    """Download and parse the full Oxford-Man realized library CSV."""
    blob: bytes | None = None
    last_exc: Exception | None = None
    for url in WAYBACK_URLS:
        print(f"Downloading Oxford-Man library from\n  {url}")
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310
                blob = resp.read()
            break
        except urllib.error.URLError as exc:
            print(f"  WARNING: download failed ({exc}); trying next mirror ...")
            last_exc = exc
    if blob is None:
        raise RuntimeError(
            f"All {len(WAYBACK_URLS)} mirror URL(s) failed. Last error: {last_exc}"
        )

    zip_sha = hashlib.sha256(blob).hexdigest()
    print(f"  sha256 of the blob: {zip_sha}")
    if EXPECTED_ZIP_SHA256 is not None:
        if zip_sha == EXPECTED_ZIP_SHA256:
            print("  ZIP checksum OK")
        else:
            raise RuntimeError(
                f"ZIP checksum mismatch: expected {EXPECTED_ZIP_SHA256}, got {zip_sha}. "
                "The downloaded blob is corrupt or the archive has changed. "
                "Verify the source and update EXPECTED_ZIP_SHA256 if the file is legitimately new."
            )

    print(f"  got {len(blob) / 1e6:.1f} MB; unzipping ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh)
    df = df.rename(columns={df.columns[0]: "date"})
    return df


def build(tickers: list[str], out: Path, verify: bool = True) -> None:
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

    if verify:
        csv_sha = hashlib.sha256(out.read_bytes()).hexdigest()
        if csv_sha == EXPECTED_CSV_SHA256:
            print("  checksum OK")
        else:
            raise RuntimeError(
                f"checksum mismatch: expected {EXPECTED_CSV_SHA256}, "
                f"got {csv_sha}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-verify", dest="verify", action="store_false",
        help="Skip SHA256 checksum verification of the written CSV.",
    )
    args = parser.parse_args()
    build(args.tickers, args.out, verify=args.verify)


if __name__ == "__main__":
    main()
