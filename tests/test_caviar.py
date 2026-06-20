"""Tests for the CAViaR quantile-regression VaR models."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.caviar import SPECS, caviar_var_forecasts
from volbench.economic import backtest_var_forecasts


def _garch_returns(n: int, seed: int = 0) -> np.ndarray:
    """Simulate a GJR-GARCH(1,1) return path with Student-t innovations."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    h = np.full(n, 1e-4)
    om, a, g, b = 2e-6, 0.05, 0.08, 0.90
    for t in range(1, n):
        eps = r[t - 1]
        h[t] = om + (a + g * (eps < 0)) * eps**2 + b * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(7)
    return r


@pytest.mark.parametrize("spec", SPECS)
def test_specs_produce_valid_forecasts(spec):
    r = _garch_returns(1600, seed=1)
    exog = np.maximum(r, 0) ** 2 + 1e-5 if spec == "REALIZED" else None
    var = caviar_var_forecasts(
        r, alpha=0.05, spec=spec, exog=exog,
        min_train=700, refit_every=300, n_starts=3, seed=0,
    )
    assert var.shape == r.shape
    assert np.all(np.isnan(var[:700]))           # warm-up has no forecast
    scored = var[700:]
    assert np.all(np.isfinite(scored))           # every origin after min_train forecast
    assert np.all(scored > 0)                    # VaR is a positive loss threshold


def test_coverage_near_alpha_on_iid():
    """On iid data CAViaR-SAV should track the unconditional alpha-quantile."""
    rng = np.random.default_rng(7)
    r = rng.standard_normal(2000) * 0.01
    var = caviar_var_forecasts(
        r, alpha=0.05, spec="SAV", min_train=700, refit_every=300, n_starts=3, seed=0,
    )
    bt = backtest_var_forecasts(r, var, alpha=0.05)
    assert abs(bt["violation_rate"] - 0.05) < 0.025   # within 2.5pp of nominal


def test_no_look_ahead():
    """A forecast at origin t must not depend on returns at or after t."""
    r = _garch_returns(1500, seed=3)
    kw = dict(alpha=0.05, spec="AS", min_train=700, refit_every=300, n_starts=3, seed=0)
    var_full = caviar_var_forecasts(r, **kw)

    k = 1100  # perturb the tail; everything strictly before k must be unchanged
    r2 = r.copy()
    r2[k:] += 0.05
    var_pert = caviar_var_forecasts(r2, **kw)

    np.testing.assert_allclose(var_full[700:k], var_pert[700:k], rtol=0, atol=0)


def test_realized_requires_exog():
    r = _garch_returns(900, seed=0)
    with pytest.raises(ValueError):
        caviar_var_forecasts(r, spec="REALIZED", min_train=600, refit_every=300)


def test_invalid_spec():
    r = _garch_returns(900, seed=0)
    with pytest.raises(ValueError):
        caviar_var_forecasts(r, spec="NOPE", min_train=600, refit_every=300)


def test_backtest_var_forecasts_counts_violations():
    """Constant VaR: violation logic and Kupiec/DQ keys behave as expected."""
    rng = np.random.default_rng(0)
    r = rng.standard_normal(1000) * 0.01
    var = np.full(1000, 0.02)          # ~2 sigma flat threshold
    var[:100] = np.nan                 # leading warm-up dropped
    bt = backtest_var_forecasts(r, var, alpha=0.05)
    expected_viol = int(np.sum(r[100:] < -0.02))
    assert bt["n"] == 900.0
    assert bt["n_violations"] == float(expected_viol)
    assert 0.0 <= bt["dq_pvalue"] <= 1.0
    assert 0.0 <= bt["kupiec_p"] <= 1.0
