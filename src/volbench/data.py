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
plausible band as a units guard (the original CSV held volatilities and *was*
squared; the real library is already variance, so the check is what carries
the units contract forward, not a squaring).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: this file is src/volbench/data.py -> repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _REPO_ROOT / "data" / "oxford_realized.csv"
_DEFAULT_VIX_CSV = _REPO_ROOT / "data" / "vix.csv"
_DEFAULT_CRYPTO_CSV = _REPO_ROOT / "data" / "crypto_realized.csv"

TRADING_DAYS: int = 252
CRYPTO_DAYS_PER_YEAR: int = 365  # crypto trades 24/7/365

# Plausibility band for annualised volatility, used to catch a units mistake
# (e.g. loading volatility as variance, or a mis-scaled column). Note: this is a
# coarse guard — it catches order-of-magnitude errors but NOT a 252-vs-365
# annualisation swap (~20% off), which stays inside the band; the correct factor
# is enforced by using TRADING_DAYS / CRYPTO_DAYS_PER_YEAR in the loaders.
_MIN_ANN_VOL: float = 0.03
_MAX_ANN_VOL: float = 0.80

# Canonical eight indices bundled with the package.
DEFAULT_TICKERS: tuple[str, ...] = (
    ".SPX", ".DJI", ".FTSE", ".GDAXI", ".FCHI", ".STOXX50E", ".N225", ".HSI",
)

# Crypto coins (Track 3), built from real intraday bars by scripts/build_crypto.py.
DEFAULT_COINS: tuple[str, ...] = ("BTC", "ETH", "BNB", "SOL")
_MIN_CRYPTO_ANN_VOL: float = 0.10
_MAX_CRYPTO_ANN_VOL: float = 3.00  # crypto can exceed 200% annualised

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


@dataclass(frozen=True)
class AssetClassConfig:
    """Per-asset-class knobs for loading a dated realized-measure panel.

    Asset classes differ only in a handful of parameters: the trading calendar
    (days per year), the plausible annualised-vol band used as a units guard,
    which measure columns the source must carry, and the default symbol
    universe. Capturing them here lets a new class (FX, commodity/rate futures,
    single equities, a VOLARE export) plug into :func:`load_realized_panel` by
    declaring a config rather than writing a bespoke loader, while the
    realized-measure maths and the downstream no-look-ahead / positivity
    invariants stay shared and untouched.

    Attributes
    ----------
    name : str
        Human-readable class label, used in error messages.
    days_per_year : int
        Annualisation factor for the units sanity check (equities/futures 252,
        crypto 365, FX ~252 — document the choice per class).
    ann_vol_band : tuple[float, float]
        ``(min, max)`` plausible annualised volatility; a series whose implied
        vol falls outside flags a units mistake.
    required_columns : tuple[str, ...]
        Measure columns (besides ``date`` / ``symbol``) that must be present.
    default_symbols : tuple[str, ...]
        Symbols loaded when the caller does not name a subset.
    """

    name: str
    days_per_year: int
    ann_vol_band: tuple[float, float]
    required_columns: tuple[str, ...]
    default_symbols: tuple[str, ...]


# Per-class configs for the two classes that have bundled data. A new class is
# added by declaring another AssetClassConfig (units / band / columns / universe)
# and pointing load_realized_panel at its CSV — no new loader code.
EQUITY_INDEX_CONFIG = AssetClassConfig(
    name="equity-index",
    days_per_year=TRADING_DAYS,
    ann_vol_band=(_MIN_ANN_VOL, _MAX_ANN_VOL),
    required_columns=_MEASURE_COLUMNS,
    default_symbols=DEFAULT_TICKERS,
)
CRYPTO_CONFIG = AssetClassConfig(
    name="crypto",
    days_per_year=CRYPTO_DAYS_PER_YEAR,
    ann_vol_band=(_MIN_CRYPTO_ANN_VOL, _MAX_CRYPTO_ANN_VOL),
    required_columns=("rv5", "bv", "medrv", "rk_parzen", "rsv", "rq", "close_price"),
    default_symbols=DEFAULT_COINS,
)


def load_realized_panel(
    path: str | Path,
    config: AssetClassConfig,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> RealizedDataset:
    """Load a dated realized-measure panel for one asset class.

    The shared core behind every per-class loader: read the CSV, check the
    class's required columns are present, and for each requested symbol sort by
    date, attach the jump/semivariance decomposition, and enforce the units
    sanity band (``sqrt(mean(rv5) * days_per_year)`` inside
    ``config.ann_vol_band``). All class-specific behaviour lives in ``config``;
    this function is asset-agnostic.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with at least ``date``, ``symbol`` and ``config.required_columns``.
    config : AssetClassConfig
        Per-class calendar / units / columns / universe.
    symbols : sequence of str, optional
        Subset to load; defaults to the config's symbols present in the file.

    Returns
    -------
    RealizedDataset
        Per-symbol frames with derived jump/semivariance columns attached.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required column is absent or a series fails the units check.
    """
    csv = Path(path)
    if not csv.exists():
        raise FileNotFoundError(f"{config.name} realized data not found at {csv}")
    df = pd.read_csv(csv, parse_dates=["date"])
    required = {"date", "symbol", *config.required_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{config.name}: CSV missing required columns: {sorted(missing)}")

    wanted = list(symbols) if symbols is not None else [
        s for s in config.default_symbols if s in set(df["symbol"])
    ]
    lo, hi = config.ann_vol_band
    panel: dict[str, pd.DataFrame] = {}
    for sym in wanted:
        sub = df[df["symbol"] == sym].copy()
        if sub.empty:
            raise ValueError(f"symbol {sym!r} not present in {csv}")
        sub = sub.sort_values("date").set_index("date").drop(columns=["symbol"])
        rv = sub["rv5"].to_numpy(dtype=float)
        if not np.all(np.isfinite(rv)) or np.any(rv <= 0):
            raise ValueError(f"{sym}: rv5 has non-finite or non-positive values")
        ann_vol = float(np.sqrt(np.mean(rv) * config.days_per_year))
        if not (lo <= ann_vol <= hi):
            raise ValueError(
                f"{sym}: implied annualised vol {ann_vol:.3f} outside "
                f"[{lo}, {hi}] — check data units"
            )
        panel[sym] = _add_derived(sub)
    return RealizedDataset(panel=panel)


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

    Notes
    -----
    A thin wrapper over :func:`load_realized_panel` with
    :data:`EQUITY_INDEX_CONFIG`; see that function for the shared loading logic.
    """
    csv = Path(path) if path is not None else _DEFAULT_CSV
    if not csv.exists():
        raise FileNotFoundError(
            f"realized data not found at {csv}; run scripts/build_realized.py to fetch it"
        )
    return load_realized_panel(csv, EQUITY_INDEX_CONFIG, symbols=tickers)


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


def load_crypto_rv(
    path: str | Path | None = None,
    coins: list[str] | tuple[str, ...] | None = None,
) -> RealizedDataset:
    """Load the crypto realized-measure panel (Track 3).

    Built by ``scripts/build_crypto.py`` from real Binance 5-minute bars, so —
    unlike the equity panel — it carries **realized quarticity** (``rq``),
    enabling HARQ on real data. Crypto trades 24/7, so the units check annualises
    by 365 and allows a wider volatility band.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV path; defaults to the bundled ``data/crypto_realized.csv`` (which is
        fetched, not committed).
    coins : sequence of str, optional
        Subset of coins; defaults to all available.

    Returns
    -------
    RealizedDataset
        Per-coin frames with the derived jump/semivariance columns plus ``rq``.

    Notes
    -----
    A thin wrapper over :func:`load_realized_panel` with :data:`CRYPTO_CONFIG`
    (365-day annualisation, wider vol band, ``rq`` required).
    """
    csv = Path(path) if path is not None else _DEFAULT_CRYPTO_CSV
    if not csv.exists():
        raise FileNotFoundError(
            f"crypto data not found at {csv}; run scripts/build_crypto.py to fetch it"
        )
    return load_realized_panel(csv, CRYPTO_CONFIG, symbols=coins)


def load_vix(
    path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Daily CBOE VIX close — implied volatility index, in annualised percent.

    Public-domain data (FRED series ``VIXCLS``), bundled in ``data/vix.csv``.
    To compare against a daily realized **variance**, convert with
    ``(vix / 100) ** 2 / TRADING_DAYS`` (the implied daily variance). This is the
    input to the variance-risk-premium analysis (``volbench.vrp``).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV path; defaults to the bundled ``data/vix.csv``.
    start, end : str, optional
        Optional ISO date bounds (inclusive).

    Returns
    -------
    pandas.Series
        Date-indexed VIX level named ``"vix"``.
    """
    csv = Path(path) if path is not None else _DEFAULT_VIX_CSV
    if not csv.exists():
        raise FileNotFoundError(
            f"VIX data not found at {csv}; run scripts/build_vix.py to fetch it"
        )
    df = pd.read_csv(csv, parse_dates=["date"])
    s = df.sort_values("date").set_index("date")["vix"].astype(float)
    s.name = "vix"
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s
