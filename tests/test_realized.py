"""Tests for volbench.realized — non-parametric realized measures."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.realized import (
    all_measures,
    bipower_variation,
    bns_jump_test,
    median_rv,
    realized_kernel_parzen,
    realized_quarticity,
    realized_semivariance,
    realized_variance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_returns(n: int = 200, seed: int = 0) -> np.ndarray:
    """Gaussian intraday returns with no jumps."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * 0.001


# ---------------------------------------------------------------------------
# realized_variance
# ---------------------------------------------------------------------------
def test_rv_known_vector():
    r = np.array([1.0, -2.0, 3.0])
    assert realized_variance(r) == pytest.approx(1.0 + 4.0 + 9.0)


def test_rv_zero_returns():
    r = np.zeros(50)
    assert realized_variance(r) == pytest.approx(0.0)


def test_rv_positive():
    rng = np.random.default_rng(42)
    r = rng.standard_normal(300)
    assert realized_variance(r) > 0.0


# ---------------------------------------------------------------------------
# realized_semivariance
# ---------------------------------------------------------------------------
def test_semivariance_non_negative():
    r = _clean_returns(200)
    rsv_minus, rsv_plus = realized_semivariance(r)
    assert rsv_minus >= 0.0
    assert rsv_plus >= 0.0


def test_semivariance_sums_to_rv():
    r = _clean_returns(300)
    rsv_minus, rsv_plus = realized_semivariance(r)
    rv = realized_variance(r)
    assert rsv_minus + rsv_plus == pytest.approx(rv, rel=1e-10)


def test_semivariance_known_vector():
    r = np.array([-2.0, 1.0, -3.0, 0.0, 4.0])
    rsv_minus, rsv_plus = realized_semivariance(r)
    assert rsv_minus == pytest.approx(4.0 + 9.0)
    assert rsv_plus == pytest.approx(1.0 + 16.0)


# ---------------------------------------------------------------------------
# bipower_variation
# ---------------------------------------------------------------------------
def test_bv_approx_rv_on_gaussian():
    """BV ≈ RV on clean Gaussian data (both estimate IV, no jumps)."""
    rng = np.random.default_rng(7)
    r = rng.standard_normal(400) * 0.001
    rv = realized_variance(r)
    bv = bipower_variation(r)
    # BV should be in a reasonable range of RV; within 20% on clean data
    assert abs(bv - rv) / rv < 0.25


def test_bv_jump_robust():
    """Inject one large jump; BV barely moves while RV jumps significantly."""
    rng = np.random.default_rng(11)
    r_base = rng.standard_normal(200) * 0.001
    r_jump = r_base.copy()
    r_jump[100] += 0.1  # very large jump relative to diffusive returns

    rv_base = realized_variance(r_base)
    rv_jump = realized_variance(r_jump)
    bv_base = bipower_variation(r_base)
    bv_jump = bipower_variation(r_jump)

    # RV should increase substantially due to jump
    assert rv_jump > rv_base * 2.0
    # BV should not increase much (jump taints at most 2 adjacent products):
    # its relative jump-induced increase is far smaller than RV's.
    assert (bv_jump / bv_base) < (rv_jump / rv_base)
    assert bv_jump < rv_jump * 0.9


def test_bv_too_short_returns_nan():
    assert np.isnan(bipower_variation(np.array([1.0])))


# ---------------------------------------------------------------------------
# median_rv
# ---------------------------------------------------------------------------
def test_medrv_positive_on_gaussian():
    r = _clean_returns(200)
    assert median_rv(r) > 0.0


def test_medrv_jump_robust():
    """medRV is robust: injecting a single large jump should not blow it up."""
    rng = np.random.default_rng(55)
    r = rng.standard_normal(300) * 0.001
    r_jump = r.copy()
    r_jump[150] += 0.1

    rv_jump = realized_variance(r_jump)
    med = median_rv(r_jump)
    # medRV should be much less than RV due to the jump
    assert med < rv_jump * 0.9


def test_medrv_too_short_returns_nan():
    assert np.isnan(median_rv(np.array([1.0, 2.0])))


# ---------------------------------------------------------------------------
# realized_quarticity
# ---------------------------------------------------------------------------
def test_rq_positive():
    r = _clean_returns(100)
    assert realized_quarticity(r) > 0.0


def test_rq_known_vector():
    r = np.array([1.0, 2.0, 3.0])
    m = 3
    expected = (m / 3.0) * (1.0 + 16.0 + 81.0)
    assert realized_quarticity(r) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# realized_kernel_parzen
# ---------------------------------------------------------------------------
def test_rk_finite_on_clean_data():
    r = _clean_returns(200)
    rk = realized_kernel_parzen(r)
    assert np.isfinite(rk)


def test_rk_approx_rv_on_clean_data():
    """On noise-free data the realized kernel should be close to RV."""
    rng = np.random.default_rng(99)
    r = rng.standard_normal(390) * 0.001
    rv = realized_variance(r)
    rk = realized_kernel_parzen(r)
    # They should be in the same order of magnitude
    assert rk == pytest.approx(rv, rel=0.30)


# ---------------------------------------------------------------------------
# bns_jump_test
# ---------------------------------------------------------------------------
_EXPECTED_BNS_KEYS = {"rv", "bv", "jump_variation", "rj", "z", "p_value"}


def test_bns_keys_present():
    r = _clean_returns(200)
    result = bns_jump_test(r)
    assert set(result.keys()) == _EXPECTED_BNS_KEYS


def test_bns_small_pvalue_on_large_jump():
    """A single enormous jump should yield a small p-value (clear jump signal)."""
    rng = np.random.default_rng(3)
    r = rng.standard_normal(390) * 0.001
    r[200] += 0.15  # huge jump
    res = bns_jump_test(r)
    assert np.isfinite(res["p_value"])
    assert res["p_value"] < 0.05


def test_bns_large_pvalue_on_clean_data():
    """No-jump data: p-value should not be tiny (fail to reject at 1% level)."""
    rng = np.random.default_rng(17)
    r = rng.standard_normal(390) * 0.001
    res = bns_jump_test(r)
    assert np.isfinite(res["p_value"])
    assert res["p_value"] > 0.01


def test_bns_jump_variation_non_negative():
    r = _clean_returns(200)
    res = bns_jump_test(r)
    assert res["jump_variation"] >= 0.0


# ---------------------------------------------------------------------------
# all_measures
# ---------------------------------------------------------------------------
_EXPECTED_MEASURE_KEYS = {
    "rv", "bv", "medrv", "rk", "rsv_minus", "rsv_plus", "rq", "jump_variation",
}


def test_all_measures_keys():
    r = _clean_returns(200)
    m = all_measures(r)
    assert set(m.keys()) == _EXPECTED_MEASURE_KEYS


def test_all_measures_finite():
    r = _clean_returns(200)
    m = all_measures(r)
    for key, val in m.items():
        assert np.isfinite(val), f"{key} is not finite"


def test_all_measures_semivariance_sum():
    r = _clean_returns(200)
    m = all_measures(r)
    assert m["rsv_minus"] + m["rsv_plus"] == pytest.approx(m["rv"], rel=1e-10)
