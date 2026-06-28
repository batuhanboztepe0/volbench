"""Tests for volbench.economic: economic-value evaluation layer.

These tests use only synthetic data generated with a fixed seed so they are
fast, deterministic, and independent of the bundled Oxford dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench.economic import (
    acerbi_szekely_backtest,
    black_scholes_price,
    engle_manganelli_dq,
    expected_shortfall_forecast,
    fz_loss,
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
# 4. VaR backtest: well-specified case
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


# ---------------------------------------------------------------------------
# 5. var_backtest: dq keys present in all modes
# ---------------------------------------------------------------------------

def test_var_backtest_output_keys_include_dq():
    """var_backtest must include dq_stat and dq_pvalue for all dist options."""
    rng = np.random.default_rng(10)
    n = 600
    sigma = 0.01
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)
    for dist in ("normal", "t", "fhs"):
        result = var_backtest(returns, fvar, alpha=0.05, dist=dist)
        assert "dq_stat" in result, f"dq_stat missing for dist={dist!r}"
        assert "dq_pvalue" in result, f"dq_pvalue missing for dist={dist!r}"


# ---------------------------------------------------------------------------
# 6. Student-t VaR fixes normal under-coverage on heavy-tailed data
# ---------------------------------------------------------------------------

def test_t_var_closer_to_alpha_than_normal_on_heavy_tails():
    """On t5-distributed data, Student-t VaR should be closer to alpha than normal VaR.

    The normal quantile (norm.ppf(0.95) ≈ 1.645) is larger than the unit-variance-
    adjusted t5 quantile (~1.56), so normal VaR sets a wider threshold and *under-
    violates* on heavy-tailed data.  Student-t VaR adapts its quantile to the true
    tail shape and produces a violation rate much closer to alpha.
    """
    rng = np.random.default_rng(99)
    n = 6000
    alpha = 0.05
    dof = 5  # heavy tails
    sigma = 0.01
    # Draw standardised t-residuals then scale to have daily vol = sigma.
    z = rng.standard_t(dof, size=n)
    z_std = z / np.sqrt(dof / (dof - 2))  # make unit variance
    returns = sigma * z_std
    fvar = np.full(n, sigma ** 2)

    r_normal = var_backtest(returns, fvar, alpha=alpha, dist="normal")
    r_t = var_backtest(returns, fvar, alpha=alpha, dist="t")

    # Normal VaR over-covers (too conservative) on fat tails, fewer violations than alpha.
    assert r_normal["violation_rate"] < alpha - 0.005, (
        f"Expected normal VaR to under-violate on t5 data; "
        f"got {r_normal['violation_rate']:.4f}"
    )
    # Student-t VaR should be closer to alpha.
    assert abs(r_t["violation_rate"] - alpha) < abs(r_normal["violation_rate"] - alpha), (
        f"t VaR rate {r_t['violation_rate']:.4f} not closer to alpha={alpha} "
        f"than normal rate {r_normal['violation_rate']:.4f}"
    )


def test_t_var_kupiec_high_on_heavy_tails():
    """Kupiec test should not reject Student-t VaR on heavy-tailed data (p > 0.05)."""
    rng = np.random.default_rng(77)
    n = 6000
    alpha = 0.05
    sigma = 0.01
    dof = 5
    z = rng.standard_t(dof, size=n)
    z_std = z / np.sqrt(dof / (dof - 2))
    returns = sigma * z_std
    fvar = np.full(n, sigma ** 2)

    result = var_backtest(returns, fvar, alpha=alpha, dist="t")
    assert result["kupiec_p"] > 0.05, (
        f"Student-t VaR Kupiec p={result['kupiec_p']:.4f} should be > 0.05"
    )


# ---------------------------------------------------------------------------
# 7. FHS VaR violation rate close to alpha on heavy-tailed data
# ---------------------------------------------------------------------------

def test_fhs_var_close_to_alpha_on_heavy_tails():
    """FHS VaR should produce violation rate close to alpha on t5-distributed returns."""
    rng = np.random.default_rng(55)
    n = 6000
    alpha = 0.05
    sigma = 0.01
    dof = 5
    z = rng.standard_t(dof, size=n)
    z_std = z / np.sqrt(dof / (dof - 2))
    returns = sigma * z_std
    fvar = np.full(n, sigma ** 2)

    result = var_backtest(returns, fvar, alpha=alpha, dist="fhs")
    # FHS uses empirical quantile so violation rate should match alpha closely.
    assert abs(result["violation_rate"] - alpha) < 0.015, (
        f"FHS violation rate {result['violation_rate']:.4f} too far from alpha={alpha}"
    )


# ---------------------------------------------------------------------------
# 8. DQ test: well-specified vs clustered violations
# ---------------------------------------------------------------------------

def test_dq_pvalue_high_for_well_specified():
    """DQ p-value should be high (> 0.05) when violations are i.i.d. Bernoulli(alpha)."""
    rng = np.random.default_rng(21)
    n = 2000
    alpha = 0.05
    # Generate i.i.d. violations, no clustering, no autocorrelation.
    viol = (rng.random(n) < alpha).astype(int)
    fvar = np.full(n, 1e-4)

    result = engle_manganelli_dq(viol, fvar, alpha=alpha, lags=4)
    assert result["dq_pvalue"] > 0.01, (
        f"DQ p-value {result['dq_pvalue']:.4f} should be > 0.01 for i.i.d. violations"
    )


def test_dq_pvalue_low_for_clustered_violations():
    """DQ p-value should be low (< 0.10) when violations are strongly clustered."""
    rng = np.random.default_rng(33)
    n = 2000
    alpha = 0.05
    # Markov chain with very high persistence: P(viol|viol) = 0.9.
    viol = np.zeros(n, dtype=int)
    viol[0] = int(rng.random() < alpha)
    for i in range(1, n):
        if viol[i - 1] == 1:
            viol[i] = int(rng.random() < 0.9)
        else:
            viol[i] = int(rng.random() < alpha / (1 - 0.9 * alpha) * alpha)
    fvar = np.full(n, 1e-4)

    result = engle_manganelli_dq(viol, fvar, alpha=alpha, lags=4)
    assert result["dq_pvalue"] < 0.10, (
        f"DQ p-value {result['dq_pvalue']:.4f} should be < 0.10 for clustered violations"
    )


def test_var_backtest_invalid_dist_raises():
    """var_backtest must raise ValueError for unknown dist."""
    rng = np.random.default_rng(7)
    n = 100
    returns = rng.normal(0, 0.01, n)
    fvar = np.full(n, 1e-4)
    with pytest.raises(ValueError, match="dist="):
        var_backtest(returns, fvar, dist="cauchy")


# ---------------------------------------------------------------------------
# 9. Expected Shortfall: normal analytical value
# ---------------------------------------------------------------------------

def test_es_normal_equals_analytical():
    """Normal ES must exactly match the closed-form formula within float precision."""
    from scipy.stats import norm as scipy_norm

    rng = np.random.default_rng(100)
    n = 500
    sigma = 0.015
    alpha = 0.05
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)

    res = expected_shortfall_forecast(returns, fvar, alpha=alpha, dist="normal")

    # Closed-form: ES = -sigma * phi(z_alpha) / alpha  (negative number)
    z_alpha = float(scipy_norm.ppf(alpha))
    es_analytical = -sigma * float(scipy_norm.pdf(z_alpha)) / alpha

    # All ES forecasts are constant (constant variance), so just check the first.
    assert res["es_forecast"][0] == pytest.approx(es_analytical, rel=1e-10), (
        f"Normal ES {res['es_forecast'][0]:.8f} differs from analytical {es_analytical:.8f}"
    )
    # ES must be negative (left-tail loss).
    assert np.all(res["es_forecast"] < 0.0), "ES forecasts must be strictly negative"
    # VaR must be positive.
    assert np.all(res["var_forecast"] > 0.0), "VaR forecasts must be strictly positive"


def test_es_normal_all_distributions_return_negative_es():
    """ES forecasts must be negative for all dist choices."""
    rng = np.random.default_rng(101)
    n = 600
    sigma = 0.01
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)

    for dist in ("normal", "t", "fhs"):
        res = expected_shortfall_forecast(returns, fvar, alpha=0.05, dist=dist)
        assert np.all(res["es_forecast"] < 0.0), (
            f"ES forecasts for dist={dist!r} must be strictly negative"
        )


# ---------------------------------------------------------------------------
# 10. Acerbi-Székely backtest: calibrated vs mis-specified
# ---------------------------------------------------------------------------

def test_acerbi_szekely_z1_near_zero_for_calibrated_model():
    """Z1 should be near 0 (and p > 0.05) for a well-specified normal ES."""
    rng = np.random.default_rng(200)
    n = 5000
    sigma = 0.01
    alpha = 0.05
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)

    res = expected_shortfall_forecast(returns, fvar, alpha=alpha, dist="normal")
    as_res = acerbi_szekely_backtest(
        returns, res["es_forecast"], res["var_forecast"], alpha=alpha, n_boot=2000, seed=42
    )

    assert abs(as_res["Z1"]) < 0.05, (
        f"Z1={as_res['Z1']:.4f} should be near 0 for calibrated ES"
    )
    assert as_res["p"] > 0.05, (
        f"p={as_res['p']:.4f} should be > 0.05 for calibrated ES"
    )


def test_acerbi_szekely_rejects_underforecast_es():
    """Z1 and Z2 should be strongly negative and p small for a halved (under-forecast) ES."""
    rng = np.random.default_rng(201)
    n = 5000
    sigma = 0.01
    alpha = 0.05
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)

    res = expected_shortfall_forecast(returns, fvar, alpha=alpha, dist="normal")
    # Halve the ES magnitude (less negative). ES underestimated.
    es_bad = res["es_forecast"] * 0.5

    as_res = acerbi_szekely_backtest(
        returns, es_bad, res["var_forecast"], alpha=alpha, n_boot=2000, seed=42
    )

    assert as_res["Z1"] < -0.5, (
        f"Z1={as_res['Z1']:.4f} should be << 0 for under-forecast ES"
    )
    assert as_res["Z2"] < -0.5, (
        f"Z2={as_res['Z2']:.4f} should be << 0 for under-forecast ES"
    )
    assert as_res["p"] < 0.05, (
        f"p={as_res['p']:.4f} should be < 0.05 for strongly under-forecast ES"
    )


# ---------------------------------------------------------------------------
# 11. FZ loss: well-specified model strictly lower than mis-specified
# ---------------------------------------------------------------------------

def test_fz_loss_well_specified_lower_than_misspecified():
    """FZ0 mean loss must be strictly lower for the true (VaR, ES) than for an under-forecast pair."""
    rng = np.random.default_rng(300)
    n = 5000
    sigma = 0.01
    alpha = 0.05
    returns = rng.normal(0.0, sigma, n)
    fvar = np.full(n, sigma ** 2)

    res = expected_shortfall_forecast(returns, fvar, alpha=alpha, dist="normal")
    es_good = res["es_forecast"]
    var_good = res["var_forecast"]

    # Mis-specified: ES underestimated by 50 % (less negative than the truth).
    es_bad = es_good * 0.5

    fz_good = fz_loss(returns, var_good, es_good, alpha=alpha)
    fz_bad = fz_loss(returns, var_good, es_bad, alpha=alpha)

    assert fz_good.mean() < fz_bad.mean(), (
        f"FZ well-spec mean={fz_good.mean():.4f} should be < mis-spec mean={fz_bad.mean():.4f}"
    )


def test_fz_loss_returns_array_of_correct_shape():
    """fz_loss must return a 1-D array with the same length as inputs."""
    rng = np.random.default_rng(301)
    n = 300
    returns = rng.normal(0.0, 0.01, n)
    fvar = np.full(n, 1e-4)

    res = expected_shortfall_forecast(returns, fvar, alpha=0.05, dist="normal")
    fz = fz_loss(returns, res["var_forecast"], res["es_forecast"], alpha=0.05)

    assert fz.shape == (n,), f"Expected shape ({n},), got {fz.shape}"
    assert np.all(np.isfinite(fz)), "All FZ0 loss values must be finite"


# ---------------------------------------------------------------------------
# 12. var_backtest backward compatibility with return_es flag
# ---------------------------------------------------------------------------

def test_var_backtest_default_return_es_false():
    """var_backtest with default return_es=False must not include ES keys."""
    rng = np.random.default_rng(400)
    n = 500
    returns = rng.normal(0.0, 0.01, n)
    fvar = np.full(n, 1e-4)

    result = var_backtest(returns, fvar)
    es_keys = {"es_mean", "as_Z1", "as_Z2", "as_p", "fz_mean"}
    assert es_keys.isdisjoint(result.keys()), (
        f"ES keys should be absent when return_es=False; found {es_keys & result.keys()}"
    )


def test_var_backtest_return_es_true_adds_keys():
    """var_backtest with return_es=True must include both original and ES keys."""
    rng = np.random.default_rng(401)
    n = 500
    returns = rng.normal(0.0, 0.01, n)
    fvar = np.full(n, 1e-4)

    result_default = var_backtest(returns, fvar)
    result_es = var_backtest(returns, fvar, return_es=True)

    # All original keys must still be present.
    for key in result_default:
        assert key in result_es, f"Original key {key!r} missing when return_es=True"

    # ES-specific keys must now be present.
    for key in ("es_mean", "as_Z1", "as_Z2", "as_p", "fz_mean"):
        assert key in result_es, f"ES key {key!r} missing when return_es=True"

    # ES mean must be negative (left-tail convention).
    assert result_es["es_mean"] < 0.0, "es_mean must be negative (left-tail ES)"
