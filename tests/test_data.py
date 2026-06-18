"""Tests for volbench.data — Oxford-Man RV data loader."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.data import (
    _DEFAULT_CSV,
    DEFAULT_TICKERS,
    TRADING_DAYS,
    RealizedDataset,
    load_oxford_rv,
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
