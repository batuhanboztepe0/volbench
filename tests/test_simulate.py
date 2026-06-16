"""Tests for volbench.simulate — intraday path simulator."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.realized import realized_variance
from volbench.simulate import simulate_intraday_path, simulate_many_days


# ---------------------------------------------------------------------------
# IntradayPath fields
# ---------------------------------------------------------------------------
def test_intraday_path_has_documented_fields():
    path = simulate_intraday_path(n_steps=100, seed=0)
    assert hasattr(path, "returns")
    assert hasattr(path, "clean_returns")
    assert hasattr(path, "iv")
    assert hasattr(path, "jv")
    assert hasattr(path, "qv")
    assert hasattr(path, "spot_var")
    assert hasattr(path, "n_jumps")


def test_intraday_path_returns_shape():
    n = 200
    path = simulate_intraday_path(n_steps=n, seed=1)
    assert path.returns.shape == (n,)
    assert path.clean_returns.shape == (n,)
    assert path.spot_var.shape == (n,)


def test_intraday_path_qv_equals_iv_plus_jv():
    path = simulate_intraday_path(n_steps=200, seed=2)
    assert path.qv == pytest.approx(path.iv + path.jv, rel=1e-12)


def test_qv_iv_jv_are_finite():
    path = simulate_intraday_path(n_steps=100, seed=3)
    assert np.isfinite(path.iv)
    assert np.isfinite(path.jv)
    assert np.isfinite(path.qv)


# ---------------------------------------------------------------------------
# Microstructure noise signature effect
# ---------------------------------------------------------------------------
def test_noise_inflates_realized_variance():
    """With microstructure noise, mean RV(observed) > mean RV(clean) on average."""
    n_days = 30
    # No noise
    days_clean = simulate_many_days(
        n_days, seed=5, n_steps=200, noise_ratio=0.0
    )
    # With noise
    days_noisy = simulate_many_days(
        n_days, seed=5, n_steps=200, noise_ratio=2.0
    )
    rv_clean = np.array([realized_variance(r) for r in days_clean["returns"]])
    rv_noisy = np.array([realized_variance(r) for r in days_noisy["returns"]])
    assert rv_noisy.mean() > rv_clean.mean()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_returns():
    path_a = simulate_intraday_path(n_steps=100, seed=99)
    path_b = simulate_intraday_path(n_steps=100, seed=99)
    np.testing.assert_array_equal(path_a.returns, path_b.returns)


def test_different_seeds_give_different_returns():
    path_a = simulate_intraday_path(n_steps=100, seed=1)
    path_b = simulate_intraday_path(n_steps=100, seed=2)
    assert not np.array_equal(path_a.returns, path_b.returns)


# ---------------------------------------------------------------------------
# Jumps
# ---------------------------------------------------------------------------
def test_jumps_with_high_intensity_nonzero_njumps():
    """With high jump intensity, mean n_jumps should be > 0."""
    results = simulate_many_days(
        n_days=20, seed=0, n_steps=100,
        jump_intensity=5.0, jump_size_vol=0.02,
    )
    assert results["n_jumps"].mean() > 0


def test_jumps_with_high_intensity_positive_jv():
    """Jump variation should be positive when jumps occur."""
    results = simulate_many_days(
        n_days=20, seed=0, n_steps=100,
        jump_intensity=5.0, jump_size_vol=0.02,
    )
    assert results["jv"].sum() > 0


def test_no_jumps_zero_jv():
    """Without jumps, jump variation and n_jumps should be zero."""
    path = simulate_intraday_path(n_steps=200, seed=77, jump_intensity=0.0)
    assert path.jv == pytest.approx(0.0)
    assert path.n_jumps == 0


# ---------------------------------------------------------------------------
# simulate_many_days keys
# ---------------------------------------------------------------------------
def test_simulate_many_days_keys():
    results = simulate_many_days(n_days=5, seed=0, n_steps=50)
    expected = {"returns", "clean_returns", "iv", "jv", "qv", "n_jumps"}
    assert set(results.keys()) == expected


def test_simulate_many_days_array_lengths():
    n = 10
    results = simulate_many_days(n_days=n, seed=0, n_steps=50)
    assert len(results["returns"]) == n
    assert len(results["iv"]) == n
    assert len(results["jv"]) == n


def test_simulate_many_days_qv_equals_iv_plus_jv():
    results = simulate_many_days(n_days=15, seed=3, n_steps=50)
    np.testing.assert_allclose(results["qv"], results["iv"] + results["jv"], rtol=1e-12)
