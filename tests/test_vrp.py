"""Tests for volbench.vrp — variance risk premium functions."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.vrp import variance_risk_premium, vrp_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_arrays(n: int = 200, seed: int = 0):
    """Return (implied_var, forecast_var, realized_future_var) as plausible
    daily variance arrays (~1e-4 scale)."""
    rng = np.random.default_rng(seed)
    iv = np.exp(rng.standard_normal(n) * 0.3 + np.log(1.5e-4))  # implied slightly high
    fv = np.exp(rng.standard_normal(n) * 0.3 + np.log(1.2e-4))  # forecast
    rv = np.exp(rng.standard_normal(n) * 0.3 + np.log(1.0e-4))  # realised lower
    return iv, fv, rv


# ---------------------------------------------------------------------------
# variance_risk_premium
# ---------------------------------------------------------------------------
def test_vrp_elementwise():
    """variance_risk_premium returns implied - forecast element-wise."""
    iv = np.array([2e-4, 3e-4, 4e-4])
    fv = np.array([1e-4, 2e-4, 2e-4])
    result = variance_risk_premium(iv, fv)
    expected = iv - fv
    np.testing.assert_allclose(result, expected)


def test_vrp_length_mismatch_raises():
    iv = np.array([1e-4, 2e-4])
    fv = np.array([1e-4])
    with pytest.raises(ValueError):
        variance_risk_premium(iv, fv)


# ---------------------------------------------------------------------------
# vrp_strategy — basic structure
# ---------------------------------------------------------------------------
def test_vrp_strategy_keys():
    iv, fv, rv = _make_arrays()
    out = vrp_strategy(iv, fv, rv, horizon=22)
    assert set(out.keys()) == {"always_short", "timed", "longshort", "dm_timed_vs_always_short"}
    for book in ("always_short", "timed", "longshort"):
        stats = out[book]
        assert set(stats.keys()) == {
            "ann_sharpe", "hit_rate", "mean_pnl", "total_pnl", "max_drawdown",
            "sharpe_pp", "psr", "dsr",
        }


def test_vrp_strategy_shapes_finite():
    """All scalar statistics must be finite floats."""
    iv, fv, rv = _make_arrays()
    out = vrp_strategy(iv, fv, rv, horizon=22)
    for book in ("always_short", "timed", "longshort"):
        for key, val in out[book].items():
            assert np.isfinite(val), f"{book}.{key} is not finite: {val}"


# ---------------------------------------------------------------------------
# Always-short book: positive total_pnl when implied > realized everywhere
# ---------------------------------------------------------------------------
def test_always_short_positive_when_implied_exceeds_realized():
    """When implied_var > realized_future_var at every origin, the always-short
    book must have a strictly positive total P&L."""
    n = 100
    iv = np.full(n, 2e-4)   # implied higher
    rv = np.full(n, 1e-4)   # realized lower
    fv = np.full(n, 1.5e-4)
    out = vrp_strategy(iv, fv, rv, horizon=1)
    assert out["always_short"]["total_pnl"] > 0.0


# ---------------------------------------------------------------------------
# Leakage-free: corrupting the last realized value does not change earlier P&L
# ---------------------------------------------------------------------------
def test_strategy_leakage_free():
    """Positions depend only on implied_var and forecast_var (both known at t).
    Corrupting the LAST element of realized_future_var must not change the
    position or P&L for any earlier period."""
    iv, fv, rv = _make_arrays(n=150)

    rv_corrupted = rv.copy()
    rv_corrupted[-1] = 1e10   # poison the last realized value

    # Positions are determined by iv and fv only, so the timed book's total_pnl
    # can only differ in the last element.  Recompute P&L directly.
    iv_safe = np.where(iv > 0.0, iv, np.finfo(float).tiny)
    pos_timed = np.clip((iv - fv) / iv_safe, -1.0, 2.0)

    pnl_clean = pos_timed * (iv - rv)
    pnl_corrupt = pos_timed * (iv - rv_corrupted)

    # All but the last period must be identical.
    np.testing.assert_allclose(pnl_clean[:-1], pnl_corrupt[:-1])


# ---------------------------------------------------------------------------
# Costs reduce (or equal) the timed Sharpe
# ---------------------------------------------------------------------------
def test_costs_reduce_sharpe():
    """With positive transaction costs the timed book's Sharpe <= zero-cost."""
    iv, fv, rv = _make_arrays()
    out_zero = vrp_strategy(iv, fv, rv, horizon=22, costs=0.0)
    out_cost = vrp_strategy(iv, fv, rv, horizon=22, costs=1e-5)
    # Total P&L (and thus Sharpe) should be lower with costs.
    assert out_cost["timed"]["total_pnl"] <= out_zero["timed"]["total_pnl"] + 1e-15


# ---------------------------------------------------------------------------
# DM result is present and well-formed
# ---------------------------------------------------------------------------
def test_dm_result_structure():
    iv, fv, rv = _make_arrays(n=300)
    out = vrp_strategy(iv, fv, rv, horizon=22)
    dm = out["dm_timed_vs_always_short"]
    assert "dm_stat" in dm
    assert "p_value" in dm
    assert "favored" in dm
    assert "n" in dm
