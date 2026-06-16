"""Loaders for the bundled realized-measure panel and return series.

The benchmark runs on the bundled subset of the Oxford-Man Institute Realized
Library (``data/oxford_realized.csv``; see ``data/README.md`` for provenance).
Unlike the stripped-down series the project started with, this file ships dated
daily **variances** for several measures (5-minute realized variance, bipower
variation, median RV, realized kernel, downside semivariance) for eight
international equity indices, which unlocks the jump / semivariance HAR family
and regime analysis on real data.

Units
-----
The library reports **variances** (decimal returns², daily). They are loaded
as-is; nothing is squared. :func:`load_oxford_rv` sanity-checks that each
index's implied annualised volatility ``sqrt(mean(rv5) * 252)`` lands in a
plausible band, enforcing the unit contract described in ``ROADMAP.md``
invariant 6 (the original CSV held volatilities and *was* squared; the real
library is already variance, so the check — not a squaring — is what carries the
invariant forward).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: this file is src/volbench/data.py -> repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _REPO_ROOT / "data" / "oxford_realized.csv"

TRADING_DAYS: int = 252

# Plausibility band for annualised volatility, used to catch a units mistake
# (e.g. loading volatility as variance, or a mis-scaled column).
_MIN_ANN_VOL: float = 0.03
_MAX_ANN_VOL: float = 0.80

# Canonical eight indices bundled with the package.
DEFAULT_TICKERS: tuple[str, ...] = (
    ".SPX", ".DJI", ".FTSE", ".GDAXI", ".FCHI", ".STOXX50E", ".N225", ".HSI",
)

# Raw realized-measure columns expected in the CSV (excluding date/symbol).
_MEASURE_COLUMNS: tuple[str, ...] = (
    "rv5", "rv5_ss", "bv", "medrv", "rk_parzen", "rsv", "close_price",
    "open_to_close", "nobs",
)


@dataclass
class RealizedDataset:
    """A panel of dated daily realized measures, one frame per index.

    Attributes
    ----------
    panel : dict[str, pandas.DataFrame]
        Mapping ``ticker -> date-indexed frame``. Each frame carries the raw
        measures plus the derived columns ``rsv_plus`` (upside semivariance,
        ``rv5 - rsv``), ``rsv_minus`` (alias of ``rsv``), ``jump`` (the
        jump-variation proxy ``max(rv5 - bv, 0)``) and ``cont`` (the continuous
        part, ``min(bv, rv5)``).
    """

    panel: dict[str, pd.DataFrame]

    @property
    def tickers(self) -> list[str]:
        """Index symbols available, in insertion order."""
        return list(self.panel)

    def frame(self, ticker: str) -> pd.DataFrame:
        """Full date-indexed measure frame for one index."""
        return self.panel[ticker]

    def series(self, ticker: str, measure: str = "rv5") -> np.ndarray:
        """A single measure as a 1-D float array (default: 5-minute RV)."""
        return self.panel[ticker][measure].to_numpy(dtype=float)

    def dates(self, ticker: str) -> pd.DatetimeIndex:
        """Trading dates for one index."""
        return self.panel[ticker].index


def _add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach jump / semivariance decomposition columns used by the HAR family."""
    rv = frame["rv5"].to_numpy(dtype=float)
    bv = frame["bv"].to_numpy(dtype=float)
    rsv = frame["rsv"].to_numpy(dtype=float)
    # Continuous part is bipower variation, capped at RV; jump is the remainder.
    cont = np.minimum(bv, rv)
    jump = np.maximum(rv - bv, 0.0)
    rsv_minus = np.minimum(rsv, rv)  # downside, can't exceed total
    rsv_plus = np.maximum(rv - rsv_minus, 0.0)  # upside reconstructed
    out = frame.copy()
    out["cont"] = cont
    out["jump"] = jump
    out["rsv_minus"] = rsv_minus
    out["rsv_plus"] = rsv_plus
    return out


def load_oxford_rv(
    path: str | Path | None = None,
    tickers: list[str] | tuple[str, ...] | None = None,
) -> RealizedDataset:
    """Load the bundled realized-measure panel.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV path; defaults to the bundled ``data/oxford_realized.csv``.
    tickers : sequence of str, optional
        Subset of indices to load; defaults to all eight bundled indices.

    Returns
    -------
    RealizedDataset
        The dated panel with derived jump/semivariance columns attached.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing.
    ValueError
        If a required column is absent or a series fails the units sanity check.
    """
    csv = Path(path) if path is not None else _DEFAULT_CSV
    if not csv.exists():
        raise FileNotFoundError(
            f"realized data not found at {csv}; run scripts/build_realized.py to fetch it"
        )
    df = pd.read_csv(csv, parse_dates=["date"])
    required = {"date", "symbol", *(_MEASURE_COLUMNS)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    wanted = list(tickers) if tickers is not None else [
        t for t in DEFAULT_TICKERS if t in set(df["symbol"])
    ]
    panel: dict[str, pd.DataFrame] = {}
    for tk in wanted:
        sub = df[df["symbol"] == tk].copy()
        if sub.empty:
            raise ValueError(f"ticker {tk!r} not present in {csv}")
        sub = sub.sort_values("date").set_index("date")
        sub = sub.drop(columns=["symbol"])
        rv = sub["rv5"].to_numpy(dtype=float)
        if not np.all(np.isfinite(rv)) or np.any(rv <= 0):
            raise ValueError(f"{tk}: rv5 has non-finite or non-positive values")
        ann_vol = float(np.sqrt(np.mean(rv) * TRADING_DAYS))
        if not (_MIN_ANN_VOL <= ann_vol <= _MAX_ANN_VOL):
            raise ValueError(
                f"{tk}: implied annualised vol {ann_vol:.3f} outside "
                f"[{_MIN_ANN_VOL}, {_MAX_ANN_VOL}] — check data units"
            )
        panel[tk] = _add_derived(sub)
    return RealizedDataset(panel=panel)


def load_sp500_returns(
    path: str | Path | None = None,
    ticker: str = ".SPX",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Daily close-to-close log returns for one index (Track 2 / GARCH).

    Derived from the bundled ``close_price`` column so Track 2 needs no extra
    file. Returns are in **decimal** units (multiply by 100 before fitting an
    `arch` model, as that library expects percentage-scale returns).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV path; defaults to the bundled file.
    ticker : str, default ``".SPX"``
        Which index's price series to difference.
    start, end : str, optional
        Optional ISO date bounds (inclusive) to subsample the series.

    Returns
    -------
    pandas.Series
        Date-indexed daily log returns named ``"return"``.
    """
    csv = Path(path) if path is not None else _DEFAULT_CSV
    df = pd.read_csv(csv, parse_dates=["date"])
    sub = df[df["symbol"] == ticker].sort_values("date").set_index("date")
    if sub.empty:
        raise ValueError(f"ticker {ticker!r} not present in {csv}")
    price = sub["close_price"].astype(float)
    price = price[price > 0]
    ret = np.log(price).diff().dropna()
    ret.name = "return"
    if start is not None:
        ret = ret[ret.index >= pd.Timestamp(start)]
    if end is not None:
        ret = ret[ret.index <= pd.Timestamp(end)]
    return ret
