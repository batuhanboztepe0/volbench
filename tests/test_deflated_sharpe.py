"""Tests for volbench.deflated_sharpe — probabilistic and deflated Sharpe ratios."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)


def _returns(n, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


# ---------------------------------------------------------------------------
# per_period_sharpe
# ---------------------------------------------------------------------------
def test_per_period_sharpe_matches_mean_over_std():
    r = _returns(5000, 0.001, 0.01, 0)
    assert per_period_sharpe(r) == pytest.approx(np.mean(r) / np.std(r, ddof=1), rel=1e-12)


def test_per_period_sharpe_nan_on_degenerate():
    assert np.isnan(per_period_sharpe(np.ones(100)))   # zero variance
    assert np.isnan(per_period_sharpe(np.array([1.0, 2.0])))  # too few obs


# ---------------------------------------------------------------------------
# probabilistic_sharpe_ratio
# ---------------------------------------------------------------------------
def test_psr_in_unit_interval():
    r = _returns(2000, 0.0005, 0.01, 1)
    p = probabilistic_sharpe_ratio(r)
    assert 0.0 <= p <= 1.0


def test_psr_high_for_strong_positive_sharpe():
    # Large positive mean/vol ratio over a long sample -> near-certain > 0.
    r = _returns(5000, 0.002, 0.01, 2)
    assert probabilistic_sharpe_ratio(r) > 0.99


def test_psr_near_half_for_zero_sharpe():
    r = _returns(5000, 0.0, 0.01, 3)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(0.5, abs=0.15)


def test_psr_increases_with_sample_length():
    # Same per-period Sharpe, more (effective) observations -> stronger evidence.
    r = _returns(5000, 0.0008, 0.01, 4)
    short = probabilistic_sharpe_ratio(r, n_eff=200)
    long = probabilistic_sharpe_ratio(r, n_eff=5000)
    assert long > short


def test_psr_lower_with_negative_skew():
    # Two series with the SAME per-period Sharpe; the negatively-skewed one is less
    # credible. Evaluate at a moderate n_eff so PSR is not saturated at 1.
    rng = np.random.default_rng(5)
    base = rng.normal(0.0008, 0.01, 8000)
    # Inject sharp negative-skew shocks, then rescale to base's mean/std (same SR).
    skewed = base.copy()
    shocks = rng.random(8000) < 0.02
    skewed[shocks] -= 0.05
    skewed = (skewed - skewed.mean()) / skewed.std() * base.std() + base.mean()
    assert per_period_sharpe(skewed) == pytest.approx(per_period_sharpe(base), rel=1e-9)
    assert probabilistic_sharpe_ratio(skewed, n_eff=300) < probabilistic_sharpe_ratio(base, n_eff=300)


# ---------------------------------------------------------------------------
# expected_max_sharpe / deflated_sharpe_ratio
# ---------------------------------------------------------------------------
def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe([0.05]) == 0.0


def test_expected_max_sharpe_positive_and_grows_with_trials():
    few = expected_max_sharpe([0.01, 0.05, 0.03])
    many = expected_max_sharpe([0.01, 0.05, 0.03, 0.02, 0.06, 0.04, 0.00, 0.07])
    assert few > 0.0
    assert many > few   # more trials -> higher expected best-under-null


def test_dsr_not_greater_than_undeflated_psr():
    # Deflating for selection can only lower (or equal) the probability.
    r = _returns(5000, 0.001, 0.01, 6)
    trials = [per_period_sharpe(r), 0.02, 0.005, 0.015]
    dsr = deflated_sharpe_ratio(r, trials)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert dsr <= psr + 1e-12


def test_dsr_equals_psr_with_one_trial():
    r = _returns(5000, 0.001, 0.01, 7)
    assert deflated_sharpe_ratio(r, [per_period_sharpe(r)]) == pytest.approx(
        probabilistic_sharpe_ratio(r, benchmark_sr=0.0), rel=1e-12
    )
