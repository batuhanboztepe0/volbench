"""Tests for the crypto realized panel (Track 3).

The crypto CSV is fetched (not committed), so these skip until
``scripts/build_crypto.py`` has been run.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench.data import (
    _DEFAULT_CRYPTO_CSV,
    CRYPTO_DAYS_PER_YEAR,
    RealizedDataset,
    load_crypto_rv,
)
from volbench.models import HARQ

pytestmark = pytest.mark.skipif(
    not _DEFAULT_CRYPTO_CSV.exists(),
    reason="crypto data not fetched; run scripts/build_crypto.py",
)


def test_load_crypto_returns_dataset():
    ds = load_crypto_rv()
    assert isinstance(ds, RealizedDataset)
    assert len(ds.tickers) >= 1


def test_crypto_frames_have_rq_and_derived():
    ds = load_crypto_rv()
    for c in ds.tickers:
        fr = ds.frame(c)
        for col in ("rv5", "bv", "rq", "cont", "jump", "rsv_minus", "rsv_plus"):
            assert col in fr.columns
        assert np.all(fr["rq"].to_numpy() > 0)  # realized quarticity positive


def test_crypto_semivariance_reconstructs_rv():
    ds = load_crypto_rv()
    fr = ds.frame(ds.tickers[0])
    rv = fr["rv5"].to_numpy()
    recon = fr["rsv_minus"].to_numpy() + fr["rsv_plus"].to_numpy()
    assert np.allclose(recon, rv, rtol=1e-6, atol=1e-12)


def test_crypto_units_plausible():
    ds = load_crypto_rv()
    for c in ds.tickers:
        rv = ds.series(c)
        ann_vol = float(np.sqrt(np.mean(rv) * CRYPTO_DAYS_PER_YEAR))
        assert 0.10 <= ann_vol <= 3.0  # crypto annualised vol band


def test_harq_runs_on_real_rq():
    ds = load_crypto_rv()
    fr = ds.frame(ds.tickers[0])
    rv = fr["rv5"].to_numpy()
    fc, origins = HARQ(fr["rq"].to_numpy()).oos_forecast(rv, horizon=1, min_train=400)
    assert fc.shape == origins.shape
    assert np.all(fc > 0)            # clamp guarantees positivity
    assert np.all(np.isfinite(fc))
