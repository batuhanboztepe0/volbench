"""Tests for volbench.economic — economic-value evaluation layer.

These tests use only synthetic data generated with a fixed seed so they are
fast, deterministic, and independent of the bundled Oxford dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench.economic import (
    black_scholes_price,
    option_pricing_loss,
    var_backtest,
    volatility_targeting,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_varying_vol_series(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate daily returns with time-varying (regime-switching) volatility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(returns, true_variance)`` arrays of length ``n``.
    """
    rng = np.random.default_rng(seed)
    true_var = np.where(
        rng.random(n) < 0.5,
        (0.01 / np.sqrt(252)) ** 2,   # low-vol regime: ~1% ann
        (0.04 / np.sqrt(252)) ** 2,   # high-vol regime: ~4% ann
    )
    returns = rng.normal(0.0, np.sqrt(true_var))
    return returns, true_var


# ---------------------------------------------------------------------------
# 1. Vol-targeting: strategy vol closer to target than B&H vol
# ---------------------------------------------------------------------------

def test_vol_targeting_reduces_vol_variability():
    """Vol-targeting should bring realized vol closer to the 10% target."""
    returns, true_var = _make_varying_vol_series(n=2000, seed=1)
    target = 0.10

    vt = volatility_targeting(returns, true_var, target_ann_vol=target)

    # The strategy's realized annual vol should be closer to the target
    # than buy-and-hold (which is a mix of the two regimes).
    assert abs(vt["ann_vol"] - target) < abs(vt["bh_ann_vol"] - target), (
        f"Strategy vol {vt['ann_vol']:.4f} is not closer to target {target} "
        f"than B&H vol {vt['bh_ann_vol']:.4f}"
    )


def test_vol_targeting_leverage_clipping():
    """Weights must stay in [0, max_leverage]."""
    rng = np.random.default_rng(2)
    n = 500
    returns = rng.normal(0.0, 0.01, n)
    # Very tiny forecast var -> weights would blow up without clipping.
    fvar = np.full(n, 1e-12)

    vt = volatility_targeting(returns, fvar, max_leverage=3.0)
    assert vt["avg_leverage"] <= 3.0 + 1e-9


def test_vol_targeting_output_keys():
    """Return dict must contain all expected keys."""
    rng = np.random.default_rng(3)
    n = 300
    ret = rng.normal(0, 0.01, n)
    fvar = np.full(n, (0.01 / np.sqrt(252)) ** 2)
    result = volatility_targeting(ret, fvar)
    for key in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover",
                "avg_leverage", "bh_sharpe", "bh_ann_vol"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 2. Black-Scholes
# ---------------------------------------------------------------------------

def test_bs_call_monotone_in_sigma():
    """Black-Scholes call price must increase strictly in sigma."""
    sigmas = np.linspace(0.05, 0.80, 20)
    prices = [black_scholes_price(S=100, K=100, T=0.25, r=0.0, sigma=s, call=True)
              for s in sigmas]
    diffs = np.diff(prices)
    assert np.all(diffs > 0), "Call price not monotone in sigma"


def test_bs_call_positive():
    """Black-Scholes call price must be strictly positive for sigma > 0."""
    p = black_scholes_price(S=100, K=100, T=1.0, r=0.02, sigma=0.20, call=True)
    assert p > 0.0


def test_bs_put_call_parity():
    """Put-call parity: C - P = S*exp(0) - K*exp(-rT) when r=0, S=K."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    c = black_scholes_price(S, K, T, r, sigma, call=True)
    p = black_scholes_price(S, K, T, r, sigma, call=False)
    parity = S - K * np.exp(-r * T)
    assert abs((c - p) - parity) < 1e-10, f"Put-call parity violated: {c - p:.6f} vs {parity:.6f}"


def test_bs_zero_maturity():
    """At T=0 the call price equals the intrinsic value."""
    p = black_scholes_price(S=110, K=100, T=0.0, r=0.0, sigma=0.20, call=True)
    assert p == pytest.approx(10.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Option-pricing loss
# ---------------------------------------------------------------------------

def test_option_pricing_loss_zero_when_equal():
    """When forecast == realized the pricing loss must be exactly 0."""
    n = 100
    fvar = np.full(n, (0.20 / np.sqrt(252)) ** 2)
    opl = option_pricing_loss(fvar, fvar, horizon_days=1)
    assert opl["mean_squared_price_error"] == pytest.approx(0.0, abs=1e-20)
    assert opl["rmse_price"] == pytest.approx(0.0, abs=1e-20)
    assert opl["mean_abs_price_error"] == pytest.approx(0.0, abs=1e-20)


def test_option_pricing_loss_positive_when_different():
    """Non-zero forecast error must give a positive option loss."""
    n = 100
    fvar = np.full(n, (0.20 / np.sqrt(252)) ** 2)
    rvar = np.full(n, (0.30 / np.sqrt(252)) ** 2)
    opl = option_pricing_loss(fvar, rvar, horizon_days=1)
    assert opl["rmse_price"] > 0.0


# ---------------------------------------------------------------------------
# 4. VaR backtest — well-specified case
# ---------------------------------------------------------------------------

def test_var_backtest_well_specified_violation_rate():
    """With perfectly calibrated normal VaR, violation rate should be ~alpha."""
    rng = np.random.default_rng(42)
    n = 5000
    alpha = 0.05
    sigma_t = rng.uniform(0.005, 0.02, n)  # time-varying but known
    returns = rng.normal(0.0, sigma_t)
    fvar = sigma_t ** 2

    result = var_backtest(returns, fvar, alpha=alpha)
    assert abs(result["violation_rate"] - alpha) < 0.02, (
        f"Violation rate {result['violation_rate']:.4f} too far from {alpha}"
    )


def test_var_backtest_kupiec_p_high_for_well_specified():
    """Kupiec test should fail to reject (p > 0.05) for a calibrated VaR."""
    rng = np.random.default_rng(43)
    n = 5000
    alpha = 0.05
    sigma = 0.01
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)

    result = var_backtest(returns, fvar, alpha=alpha)
    assert result["kupiec_p"] > 0.05, (
        f"Kupiec p={result['kupiec_p']:.4f} should be > 0.05 for well-specified VaR"
    )


def test_var_backtest_output_keys():
    """var_backtest must return all expected keys."""
    rng = np.random.default_rng(5)
    n = 200
    returns = rng.normal(0, 0.01, n)
    fvar = np.full(n, 1e-4)
    result = var_backtest(returns, fvar)
    for key in ("violation_rate", "expected_rate", "n_violations", "n",
                "kupiec_stat", "kupiec_p", "christoffersen_stat", "christoffersen_p"):
        assert key in result, f"Missing key: {key}"


def test_var_backtest_zero_violations_no_crash():
    """With zero violations all tests should return finite or nan without crashing."""
    n = 200
    # All returns = +0.1, large positive: no return < -VaR.
    returns = np.full(n, 0.10)
    fvar = np.full(n, 1e-4)

    result = var_backtest(returns, fvar)
    assert result["n_violations"] == 0.0
    # kupiec_stat/p should be nan (or finite) but not crash.
    # Just verify the keys are present and numeric types.
    for key in ("kupiec_stat", "kupiec_p", "christoffersen_stat", "christoffersen_p"):
        val = result[key]
        assert val is None or isinstance(val, float), f"Unexpected type for {key}: {type(val)}"
