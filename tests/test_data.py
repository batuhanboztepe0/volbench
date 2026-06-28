"""Tests for volbench.data: Oxford-Man RV data loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volbench.data import (
    _DEFAULT_CRYPTO_CSV,
    _DEFAULT_CSV,
    CRYPTO_CONFIG,
    DEFAULT_TICKERS,
    EQUITY_INDEX_CONFIG,
    TRADING_DAYS,
    AssetClassConfig,
    RealizedDataset,
    load_oxford_rv,
    load_realized_panel,
    load_sp500_returns,
)

# The Oxford-Man data is not redistributed in the repo; skip these tests until it
# has been fetched with `python scripts/build_realized.py`.
pytestmark = pytest.mark.skipif(
    not _DEFAULT_CSV.exists(),
    reason="bundled data not fetched; run scripts/build_realized.py",
)


# ---------------------------------------------------------------------------
# load_oxford_rv
# ---------------------------------------------------------------------------
def test_load_oxford_rv_returns_dataset():
    ds = load_oxford_rv()
    assert isinstance(ds, RealizedDataset)


def test_load_oxford_rv_exactly_8_tickers():
    ds = load_oxford_rv()
    assert len(ds.tickers) == 8


def test_load_oxford_rv_contains_spx():
    ds = load_oxford_rv()
    assert ".SPX" in ds.tickers


def test_load_oxford_rv_all_default_tickers():
    ds = load_oxford_rv()
    for tk in DEFAULT_TICKERS:
        assert tk in ds.tickers


def test_frame_has_derived_columns():
    ds = load_oxford_rv()
    for tk in ds.tickers:
        df = ds.frame(tk)
        for col in ("rv5", "bv", "cont", "jump", "rsv_minus", "rsv_plus"):
            assert col in df.columns, f"{tk} frame missing column {col!r}"


def test_semivariance_decomposition_sums_to_rv5():
    """rsv_minus + rsv_plus ≈ rv5 (within float tolerance)."""
    ds = load_oxford_rv()
    for tk in ds.tickers:
        df = ds.frame(tk)
        lhs = df["rsv_minus"].to_numpy() + df["rsv_plus"].to_numpy()
        rhs = df["rv5"].to_numpy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8,
                                   err_msg=f"{tk}: rsv_minus+rsv_plus != rv5")


def test_spx_implied_ann_vol_plausible():
    """Implied annualised vol for SPX should be in (0.05, 0.60)."""
    ds = load_oxford_rv()
    rv5 = ds.series(".SPX", "rv5")
    ann_vol = float(np.sqrt(np.mean(rv5) * TRADING_DAYS))
    assert 0.05 < ann_vol < 0.60, f"SPX implied vol {ann_vol:.3f} outside plausible range"


def test_frame_date_indexed():
    import pandas as pd
    ds = load_oxford_rv()
    df = ds.frame(".SPX")
    assert isinstance(df.index, pd.DatetimeIndex)


def test_series_returns_1d_float():
    ds = load_oxford_rv()
    rv = ds.series(".SPX", "rv5")
    assert rv.ndim == 1
    assert rv.dtype == float


# ---------------------------------------------------------------------------
# asset-class abstraction (load_realized_panel + AssetClassConfig)
# ---------------------------------------------------------------------------
def _reference_symbol_frame(csv_path, symbol):
    """Independent oracle: reproduce the loader's per-symbol logic with plain
    pandas/numpy, NOT by calling load_realized_panel. Returns the sorted frame
    and the four derived columns recomputed from scratch, so the assertions below
    pin the loader's behaviour rather than comparing it to itself."""
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    sub = (
        raw[raw["symbol"] == symbol]
        .sort_values("date")
        .set_index("date")
        .drop(columns=["symbol"])
    )
    rv = sub["rv5"].to_numpy(dtype=float)
    bv = sub["bv"].to_numpy(dtype=float)
    rsv = sub["rsv"].to_numpy(dtype=float)
    cont = np.minimum(bv, rv)
    jump = np.maximum(rv - bv, 0.0)
    rsv_minus = np.minimum(rsv, rv)
    rsv_plus = np.maximum(rv - rsv_minus, 0.0)
    return sub, {"cont": cont, "jump": jump, "rsv_minus": rsv_minus, "rsv_plus": rsv_plus}


def test_panel_matches_independent_reference_equity():
    """load_realized_panel reproduces an independent re-implementation (equity)."""
    ds = load_realized_panel(_DEFAULT_CSV, EQUITY_INDEX_CONFIG)
    assert set(ds.tickers) <= set(DEFAULT_TICKERS)
    for tk in ds.tickers:
        ref, derived = _reference_symbol_frame(_DEFAULT_CSV, tk)
        fr = ds.frame(tk)
        np.testing.assert_array_equal(fr.index.to_numpy(), ref.index.to_numpy())
        np.testing.assert_array_equal(fr["rv5"].to_numpy(), ref["rv5"].to_numpy())
        for col, expected in derived.items():
            np.testing.assert_allclose(fr[col].to_numpy(), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not _DEFAULT_CRYPTO_CSV.exists(),
    reason="crypto data not fetched; run scripts/build_crypto.py",
)
def test_panel_matches_independent_reference_crypto():
    """load_realized_panel reproduces an independent re-implementation (crypto),
    including pass-through of the crypto-only ``rq`` column."""
    ds = load_realized_panel(_DEFAULT_CRYPTO_CSV, CRYPTO_CONFIG)
    for c in ds.tickers:
        ref, derived = _reference_symbol_frame(_DEFAULT_CRYPTO_CSV, c)
        fr = ds.frame(c)
        np.testing.assert_array_equal(fr["rv5"].to_numpy(), ref["rv5"].to_numpy())
        np.testing.assert_array_equal(fr["rq"].to_numpy(), ref["rq"].to_numpy())
        for col, expected in derived.items():
            np.testing.assert_allclose(fr[col].to_numpy(), expected, rtol=0, atol=0)


def test_panel_units_band_violation_raises():
    """An impossible vol band makes the units sanity check fail loudly."""
    bad = AssetClassConfig(
        name="bad-band",
        days_per_year=TRADING_DAYS,
        ann_vol_band=(10.0, 20.0),  # no equity index implies 1000%+ vol
        required_columns=EQUITY_INDEX_CONFIG.required_columns,
        default_symbols=(".SPX",),
    )
    with pytest.raises(ValueError, match="check data units"):
        load_realized_panel(_DEFAULT_CSV, bad)


def test_panel_missing_column_raises():
    """A required column the source lacks is reported, not silently dropped."""
    bad = AssetClassConfig(
        name="bad-cols",
        days_per_year=TRADING_DAYS,
        ann_vol_band=EQUITY_INDEX_CONFIG.ann_vol_band,
        required_columns=(*EQUITY_INDEX_CONFIG.required_columns, "not_a_real_column"),
        default_symbols=DEFAULT_TICKERS,
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_realized_panel(_DEFAULT_CSV, bad)


# ---------------------------------------------------------------------------
# load_sp500_returns
# ---------------------------------------------------------------------------
def test_load_sp500_returns_non_empty():
    ret = load_sp500_returns()
    assert len(ret) > 0


def test_load_sp500_returns_finite():
    ret = load_sp500_returns()
    assert np.all(np.isfinite(ret.values))


def test_load_sp500_returns_is_series():
    import pandas as pd
    ret = load_sp500_returns()
    assert isinstance(ret, pd.Series)


def test_load_sp500_returns_name():
    ret = load_sp500_returns()
    assert ret.name == "return"
