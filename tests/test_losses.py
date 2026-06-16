"""Tests for volbench.losses — forecast evaluation loss functions."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.losses import (
    LOSS_FUNCTIONS,
    RANKING_LOSSES,
    mean_loss,
    mincer_zarnowitz,
    mse_variance,
    mse_volatility,
    qlike,
    rmse_volatility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pos_arrays(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    realized = np.exp(rng.standard_normal(n) * 0.3 + np.log(1e-4))
    forecast = np.exp(rng.standard_normal(n) * 0.3 + np.log(1e-4))
    return realized, forecast


# ---------------------------------------------------------------------------
# qlike
# ---------------------------------------------------------------------------
def test_qlike_zero_at_equality():
    r = np.array([1e-4, 2e-4, 3e-4])
    result = qlike(r, r.copy())
    assert np.allclose(result, 0.0, atol=1e-12)


def test_qlike_positive_otherwise():
    r, f = _pos_arrays()
    # Where r != f, qlike > 0
    loss = qlike(r, f)
    assert np.all(loss >= 0.0)
    # At least some elements should be non-zero
    assert np.any(loss > 0.0)


def test_qlike_length_match_raises():
    with pytest.raises(ValueError):
        qlike(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# mse_variance
# ---------------------------------------------------------------------------
def test_mse_variance_zero_at_equality():
    r = np.array([1e-4, 2e-4, 3e-4])
    result = mse_variance(r, r.copy())
    assert np.allclose(result, 0.0, atol=1e-20)


def test_mse_variance_positive_otherwise():
    r, f = _pos_arrays()
    loss = mse_variance(r, f)
    assert np.all(loss >= 0.0)
    assert np.any(loss > 0.0)


# ---------------------------------------------------------------------------
# mse_volatility
# ---------------------------------------------------------------------------
def test_mse_volatility_zero_at_equality():
    r = np.array([1e-4, 4e-4, 9e-4])
    result = mse_volatility(r, r.copy())
    assert np.allclose(result, 0.0, atol=1e-20)


def test_mse_volatility_positive_otherwise():
    r, f = _pos_arrays()
    loss = mse_volatility(r, f)
    assert np.all(loss >= 0.0)
    assert np.any(loss > 0.0)


# ---------------------------------------------------------------------------
# rmse_volatility
# ---------------------------------------------------------------------------
def test_rmse_volatility_scalar():
    r, f = _pos_arrays()
    result = rmse_volatility(r, f)
    assert isinstance(result, float)
    assert result >= 0.0


def test_rmse_volatility_zero_at_equality():
    r = np.array([1e-4, 2e-4, 3e-4])
    assert rmse_volatility(r, r.copy()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# mean_loss
# ---------------------------------------------------------------------------
def test_mean_loss_ignores_nan():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    assert mean_loss(a) == pytest.approx(7.0 / 3.0)


def test_mean_loss_all_nan_returns_nan():
    a = np.array([np.nan, np.nan])
    assert np.isnan(mean_loss(a))


def test_mean_loss_basic():
    a = np.array([2.0, 4.0, 6.0])
    assert mean_loss(a) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# mincer_zarnowitz
# ---------------------------------------------------------------------------
def test_mz_near_perfect_forecast():
    """Near-perfect forecast: alpha≈0, beta≈1, r2≈1, p_joint high (>0.05)."""
    rng = np.random.default_rng(42)
    n = 300
    realized = np.exp(rng.standard_normal(n) * 0.3 + np.log(1e-4))
    # Tiny perturbation so the regression is not degenerate
    forecast = realized * np.exp(rng.standard_normal(n) * 0.005)

    res = mincer_zarnowitz(realized, forecast)
    assert res["alpha"] == pytest.approx(0.0, abs=1e-6)
    assert res["beta"] == pytest.approx(1.0, abs=0.05)
    assert res["r2"] > 0.99
    assert res["p_joint"] > 0.05


def test_mz_keys():
    r, f = _pos_arrays()
    res = mincer_zarnowitz(r, f)
    assert set(res.keys()) == {"alpha", "beta", "r2", "p_alpha", "p_beta", "p_joint", "n"}


def test_mz_n_matches_input():
    r, f = _pos_arrays(150)
    res = mincer_zarnowitz(r, f)
    assert int(res["n"]) == 150


# ---------------------------------------------------------------------------
# LOSS_FUNCTIONS and RANKING_LOSSES
# ---------------------------------------------------------------------------
def test_loss_functions_keys():
    assert "QLIKE" in LOSS_FUNCTIONS
    assert "MSE-var" in LOSS_FUNCTIONS
    assert "MSE-vol" in LOSS_FUNCTIONS


def test_ranking_losses_tuple():
    assert RANKING_LOSSES == ("QLIKE", "MSE-var")
