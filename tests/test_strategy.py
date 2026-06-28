"""Tests for volbench.strategy: vol-targeting and regime overlay.

All tests use synthetic data with a fixed seed and run without the Oxford
dataset so they are fast and deterministic.
"""

from __future__ import annotations

import numpy as np

from volbench.strategy import compare_books, regime_overlay, vol_target_backtest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_series(n: int = 800, seed: int = 0):
    """Synthetic time-varying-vol return and variance series.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(returns, forecast_var, rv, jump)``
    """
    rng = np.random.default_rng(seed)
    # Alternate between low (~10 % ann) and high (~30 % ann) vol blocks.
    true_var = np.where(
        np.arange(n) % 100 < 50,
        (0.10 / np.sqrt(252)) ** 2,
        (0.30 / np.sqrt(252)) ** 2,
    )
    returns = rng.normal(0.0, np.sqrt(true_var))
    # Forecast variance: true var with small noise (realistic forecaster)
    noise = rng.uniform(0.8, 1.2, n)
    forecast_var = true_var * noise
    # RV and jump: RV ≈ true variance; occasional spikes in jump
    rv = true_var * rng.uniform(0.9, 1.1, n)
    jump = np.maximum(rng.exponential(true_var * 0.1, n), 0.0)
    return returns, forecast_var, rv, jump


# ---------------------------------------------------------------------------
# 1. Vol-targeting reduces vol variability vs buy-and-hold
# ---------------------------------------------------------------------------

def test_vol_target_reduces_vol_variability():
    """Strategy vol should be closer to the 10% target than buy-and-hold vol."""
    returns, fvar, *_ = _make_series(n=1200, seed=1)
    target = 0.10
    stats = vol_target_backtest(returns, fvar, target_ann_vol=target)

    strat_vol = stats["ann_vol"]
    bh_vol = stats["bh_ann_vol"]
    assert abs(strat_vol - target) < abs(bh_vol - target), (
        f"Strategy vol {strat_vol:.4f} not closer to target {target} "
        f"than B&H vol {bh_vol:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Higher cost lowers net Sharpe
# ---------------------------------------------------------------------------

def test_higher_cost_lowers_net_sharpe():
    """Net Sharpe with higher cost_per_turnover must be <= that with lower cost."""
    returns, fvar, *_ = _make_series(n=1000, seed=2)
    stats_low = vol_target_backtest(returns, fvar, cost_per_turnover=0.0)
    stats_high = vol_target_backtest(returns, fvar, cost_per_turnover=0.005)

    assert stats_high["net_sharpe"] <= stats_low["net_sharpe"] + 1e-10, (
        f"Expected higher cost to lower net Sharpe: "
        f"low={stats_low['net_sharpe']:.4f}, high={stats_high['net_sharpe']:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Regime overlay reduces gross exposure in high-vol windows
# ---------------------------------------------------------------------------

def test_regime_overlay_reduces_exposure_in_high_vol():
    """Overlay must cut average weight in the turbulent half of the series."""
    rng = np.random.default_rng(3)
    n = 400
    # First half: low vol; second half: high vol
    rv = np.concatenate([
        rng.uniform(1e-5, 2e-5, n // 2),
        rng.uniform(1e-4, 2e-4, n // 2),
    ])
    jump = rng.exponential(rv * 0.1)
    weights = np.ones(n)

    adj = regime_overlay(weights, rv, jump)

    # In the high-vol second half the mean adjusted weight should be < 1
    mean_adj_high = float(np.mean(adj[n // 2 :]))
    assert mean_adj_high < 1.0, (
        f"Expected overlay to de-risk in high-vol half, mean={mean_adj_high:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Regime overlay is leakage-free
# ---------------------------------------------------------------------------

def test_regime_overlay_no_lookahead():
    """Corrupting the last rv/jump entry must not change earlier weights."""
    returns, fvar, rv, jump = _make_series(n=300, seed=4)
    weights = np.ones(300)

    adj1 = regime_overlay(weights.copy(), rv.copy(), jump.copy())

    # Corrupt the final observation
    rv2 = rv.copy()
    jump2 = jump.copy()
    rv2[-1] = 1e10
    jump2[-1] = 1e10
    adj2 = regime_overlay(weights.copy(), rv2, jump2)

    # All weights except the last should be unchanged
    np.testing.assert_array_equal(
        adj1[:-1], adj2[:-1],
        err_msg="Earlier weights changed after corrupting the last rv/jump"
    )


# ---------------------------------------------------------------------------
# 5. Output shapes and finiteness
# ---------------------------------------------------------------------------

def test_vol_target_output_shapes_and_finite():
    """All output values must be finite scalars."""
    returns, fvar, *_ = _make_series(n=600, seed=5)
    stats = vol_target_backtest(returns, fvar)

    expected_keys = {
        "gross_sharpe", "net_sharpe", "ann_return", "ann_vol",
        "max_drawdown", "turnover", "avg_leverage",
        "bh_sharpe", "bh_ann_vol", "bh_max_drawdown",
    }
    assert expected_keys <= set(stats.keys()), f"Missing keys: {expected_keys - set(stats.keys())}"
    for k, v in stats.items():
        assert np.isfinite(v), f"Non-finite value for key {k!r}: {v}"


# ---------------------------------------------------------------------------
# 6. Max drawdown is non-positive
# ---------------------------------------------------------------------------

def test_max_drawdown_nonpositive():
    """max_drawdown must be <= 0 for any strategy."""
    returns, fvar, *_ = _make_series(n=500, seed=6)
    stats = vol_target_backtest(returns, fvar)
    assert stats["max_drawdown"] <= 0.0 + 1e-12, (
        f"max_drawdown={stats['max_drawdown']:.6f} should be non-positive"
    )
    assert stats["bh_max_drawdown"] <= 0.0 + 1e-12, (
        f"bh_max_drawdown={stats['bh_max_drawdown']:.6f} should be non-positive"
    )


# ---------------------------------------------------------------------------
# 7. compare_books returns three books with consistent keys
# ---------------------------------------------------------------------------

def test_compare_books_structure():
    """compare_books must return a dict with the three expected book keys."""
    returns, fvar, rv, jump = _make_series(n=700, seed=7)
    result = compare_books(returns, fvar, rv, jump)

    assert set(result.keys()) == {"buy_hold", "vol_target", "vol_target_plus_overlay"}, (
        f"Unexpected keys: {set(result.keys())}"
    )
    required_keys = {
        "gross_sharpe", "net_sharpe", "ann_return", "ann_vol",
        "max_drawdown", "turnover", "avg_leverage",
        "bh_sharpe", "bh_ann_vol", "bh_max_drawdown",
    }
    for book_name, stats in result.items():
        missing = required_keys - set(stats.keys())
        assert not missing, f"{book_name} missing keys: {missing}"
        for k, v in stats.items():
            assert isinstance(v, float), f"{book_name}[{k!r}] is not float: {type(v)}"


def test_all_books_have_honest_sharpe_keys():
    """Every book, including buy_hold, must carry sharpe_pp and psr so code that
    loops over the books (e.g. a JSON writer) does not KeyError on buy_hold."""
    returns, fvar, rv, jump = _make_series(n=700, seed=7)
    result = compare_books(returns, fvar, rv, jump)
    for book_name, stats in result.items():
        assert "sharpe_pp" in stats, f"{book_name} missing sharpe_pp"
        assert "psr" in stats, f"{book_name} missing psr"
