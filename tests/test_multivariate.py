"""Tests for volbench.multivariate — CrossHAR, align_panel, spillover_backtest."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.models import (
    LogHAR,
    _test_origins,
    average_future_variance,
)
from volbench.multivariate import CrossHAR, align_panel


# ---------------------------------------------------------------------------
# Shared synthetic-data factory
# ---------------------------------------------------------------------------
def _make_rv(n: int, n_peers: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly-positive target rv (n,) and peer matrix (n, n_peers)."""
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.standard_normal(n) * 0.5 - 5.0)   # log-normal, always positive
    peers = np.exp(rng.standard_normal((n, n_peers)) * 0.5 - 5.0)
    return rv, peers


MIN_TRAIN = 100   # short for speed


# ---------------------------------------------------------------------------
# 1. Output shapes match _test_origins
# ---------------------------------------------------------------------------
def test_crosshar_output_length():
    n, horizon = 400, 1
    rv, peers = _make_rv(n, 3)
    model = CrossHAR(peers)
    fc, origins = model.oos_forecast(rv, horizon, min_train=MIN_TRAIN)
    expected = _test_origins(n, horizon, MIN_TRAIN)
    assert fc.shape == origins.shape
    assert origins.size == expected.size
    assert np.array_equal(origins, expected)


def test_crosshar_output_length_horizon5():
    n, horizon = 500, 5
    rv, peers = _make_rv(n, 2)
    model = CrossHAR(peers)
    fc, origins = model.oos_forecast(rv, horizon, min_train=MIN_TRAIN)
    expected = _test_origins(n, horizon, MIN_TRAIN)
    assert fc.shape == origins.shape
    assert origins.size == expected.size


# ---------------------------------------------------------------------------
# 2. All forecasts strictly positive (log-space model)
# ---------------------------------------------------------------------------
def test_crosshar_all_positive():
    n, horizon = 400, 1
    rv, peers = _make_rv(n, 4)
    model = CrossHAR(peers)
    fc, _ = model.oos_forecast(rv, horizon, min_train=MIN_TRAIN)
    assert np.all(fc > 0), f"Non-positive forecasts: min={fc.min()}"


def test_crosshar_all_finite():
    n, horizon = 400, 1
    rv, peers = _make_rv(n, 2)
    model = CrossHAR(peers)
    fc, _ = model.oos_forecast(rv, horizon, min_train=MIN_TRAIN)
    assert np.all(np.isfinite(fc)), "Non-finite forecasts found"


# ---------------------------------------------------------------------------
# 3. No-lookahead test — corrupting future target/peer rows must not affect
#    forecasts at origins t < n - 1 - horizon
# ---------------------------------------------------------------------------
def test_crosshar_no_lookahead():
    """Multiplying target and peer values after the last training row by ×100
    must NOT change any earlier forecast origin.  If it does, there is a
    lookahead leak."""
    n, horizon = 400, 1
    rv, peers = _make_rv(n, 3, seed=42)
    model_clean = CrossHAR(peers.copy())
    fc_clean, origins_clean = model_clean.oos_forecast(rv.copy(), horizon, min_train=MIN_TRAIN)

    # Corrupt the very last row of target and peers (only origin n-1-horizon
    # looks ahead to that row, so all earlier origins should be unaffected).
    rv_corrupt = rv.copy()
    peers_corrupt = peers.copy()
    rv_corrupt[-1] *= 1000.0
    peers_corrupt[-1] *= 1000.0

    model_corrupt = CrossHAR(peers_corrupt)
    fc_corrupt, origins_corrupt = model_corrupt.oos_forecast(
        rv_corrupt, horizon, min_train=MIN_TRAIN
    )

    assert np.array_equal(origins_clean, origins_corrupt)
    # All origins except possibly the last one should be identical.
    assert np.allclose(fc_clean[:-1], fc_corrupt[:-1], rtol=1e-10), (
        "Corrupting the last row changed earlier-origin forecasts — lookahead detected"
    )


def test_crosshar_no_lookahead_peer_future():
    """Multiplying peer rows AFTER an origin's training window must not affect
    that origin's forecast.  We corrupt all peer rows from index n//2 onward
    and verify that origins < n//2 - horizon are unaffected."""
    n, horizon = 600, 1
    rv, peers = _make_rv(n, 2, seed=7)

    model_clean = CrossHAR(peers.copy())
    fc_clean, origins_clean = model_clean.oos_forecast(rv.copy(), horizon, min_train=MIN_TRAIN)

    peers_corrupt = peers.copy()
    split = n // 2
    peers_corrupt[split:] *= 1000.0

    model_corrupt = CrossHAR(peers_corrupt)
    fc_corrupt, origins_corrupt = model_corrupt.oos_forecast(
        rv.copy(), horizon, min_train=MIN_TRAIN
    )

    # Origins that use training rows entirely before `split - horizon` are unaffected.
    # Origin t trains on rows up to t - horizon; those rows are < split when t < split.
    safe_mask = origins_clean < split
    assert np.allclose(fc_clean[safe_mask], fc_corrupt[safe_mask], rtol=1e-10), (
        "Corrupting future peer rows changed past-origin forecasts — lookahead detected"
    )


# ---------------------------------------------------------------------------
# 4. align_panel returns equal-length arrays
# ---------------------------------------------------------------------------
class _FakeFrame:
    """Minimal stand-in for a date-indexed DataFrame."""
    def __init__(self, dates, rv):
        import pandas as pd
        self._df = pd.DataFrame({"rv5": rv}, index=pd.DatetimeIndex(dates))

    def __getitem__(self, key):
        return self._df[key]

    @property
    def index(self):
        return self._df.index

    def loc(self, idx, col):
        return self._df.loc[idx, col]


def test_align_panel():
    """align_panel must intersect dates and return equal-length rv arrays."""
    from unittest.mock import MagicMock

    import pandas as pd

    dates_a = pd.date_range("2000-01-03", periods=200, freq="B")
    dates_b = pd.date_range("2000-01-10", periods=190, freq="B")  # later start
    common = dates_a.intersection(dates_b)

    rng = np.random.default_rng(0)
    rv_a = np.exp(rng.standard_normal(200) * 0.5 - 5)
    rv_b = np.exp(rng.standard_normal(190) * 0.5 - 5)

    df_a = pd.DataFrame({"rv5": rv_a}, index=dates_a)
    df_b = pd.DataFrame({"rv5": rv_b}, index=dates_b)

    ds = MagicMock()
    ds.frame.side_effect = lambda tk: df_a if tk == "A" else df_b

    out_dates, rv_dict = align_panel(ds, ["A", "B"])
    assert len(out_dates) == len(common)
    assert rv_dict["A"].shape == (len(common),)
    assert rv_dict["B"].shape == (len(common),)
    # Arrays are equal length
    assert rv_dict["A"].shape == rv_dict["B"].shape


# ---------------------------------------------------------------------------
# 5. peer_rv shape validation
# ---------------------------------------------------------------------------
def test_crosshar_shape_mismatch_raises():
    n, horizon = 300, 1
    rv, peers = _make_rv(n, 2)
    # Peers with wrong number of rows
    bad_peers = peers[: n - 5]
    model = CrossHAR(bad_peers)
    with pytest.raises(ValueError, match="peer_rv has"):
        model.oos_forecast(rv, horizon, min_train=MIN_TRAIN)


# ---------------------------------------------------------------------------
# 6. Sanity: pure-noise peers don't catastrophically improve CrossHAR QLIKE
#    (CrossHAR should not be dramatically better than LogHAR with noise peers)
# ---------------------------------------------------------------------------
def test_crosshar_noise_peers_no_spurious_gain():
    """With random-noise peers, CrossHAR should not dominate LogHAR dramatically.

    We use a generous threshold: CrossHAR is not allowed to be more than 30%
    *better* (lower QLIKE) than LogHAR. True spurious overfit would typically
    show much larger gains in-sample but not OOS; here with expanding window
    and real OLS, noise peers should produce roughly equivalent or worse QLIKE.
    """
    from volbench.losses import mean_loss, qlike

    n, horizon = 600, 1
    rng = np.random.default_rng(99)
    rv = np.exp(rng.standard_normal(n) * 0.5 - 5.0)
    # Pure noise peers, completely independent of rv
    peers = rng.random((n, 3)) * 1e-4

    model_loghar = LogHAR()
    model_cross = CrossHAR(peers)

    fc_lh, orig_lh = model_loghar.oos_forecast(rv, horizon, min_train=MIN_TRAIN)
    fc_ch, orig_ch = model_cross.oos_forecast(rv, horizon, min_train=MIN_TRAIN)

    # Restrict to common origins
    common = np.intersect1d(orig_lh, orig_ch)
    lh_mask = np.isin(orig_lh, common)
    ch_mask = np.isin(orig_ch, common)

    target = average_future_variance(rv, horizon)
    realized = target[common]

    qlike_lh = mean_loss(qlike(realized, fc_lh[lh_mask]))
    qlike_ch = mean_loss(qlike(realized, fc_ch[ch_mask]))

    # CrossHAR should not be more than 30% better (lower) than LogHAR with noise peers.
    # In practice they should be comparable or CrossHAR slightly worse.
    if qlike_lh > 0:
        improvement = (qlike_lh - qlike_ch) / qlike_lh
        assert improvement < 0.30, (
            f"CrossHAR improved QLIKE by {improvement:.1%} over LogHAR with pure noise peers "
            f"— likely a lookahead or overfitting bug (qlike_lh={qlike_lh:.6f}, "
            f"qlike_ch={qlike_ch:.6f})"
        )


# ---------------------------------------------------------------------------
# 7. CrossHAR with 1-D peer array (auto-reshape to 2-D)
# ---------------------------------------------------------------------------
def test_crosshar_1d_peer():
    n, horizon = 400, 1
    rv, _ = _make_rv(n, 1)
    peer_1d = np.exp(np.random.default_rng(5).standard_normal(n) * 0.5 - 5.0)
    model = CrossHAR(peer_1d)  # 1-D should auto-reshape
    fc, origins = model.oos_forecast(rv, horizon, min_train=MIN_TRAIN)
    assert np.all(fc > 0)
    assert fc.size == origins.size
