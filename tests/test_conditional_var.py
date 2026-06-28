"""Tests for volbench.conditional_var: reactive conditional-variance VaR engines."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.conditional_var import ewma_variance_forecast, garch_variance_forecast


def _ret(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def test_ewma_shapes_and_alignment():
    y = _ret(1200)
    fc, org = ewma_variance_forecast(y, min_train=1000)
    assert fc.shape == org.shape
    assert org[0] == 1000 and org[-1] == y.size - 2
    assert np.all(fc > 0.0)


def test_ewma_no_lookahead():
    """Corrupting return[c] must leave every forecast at origins t < c unchanged."""
    y = _ret(1300, 1)
    c = 1150  # interior, above min_train so origins exist on both sides
    y2 = y.copy()
    y2[c] *= 50.0
    fc1, org = ewma_variance_forecast(y, min_train=1000)
    fc2, _ = ewma_variance_forecast(y2, min_train=1000)
    before = org < c
    np.testing.assert_allclose(fc1[before], fc2[before], rtol=1e-12, atol=0.0)
    # The corruption must propagate to later origins (otherwise the test is vacuous).
    assert not np.allclose(fc1[~before], fc2[~before])


def test_ewma_requires_enough_data():
    with pytest.raises(ValueError):
        ewma_variance_forecast(np.ones(100), min_train=1000)


def test_garch_smoke_finite_positive():
    pytest.importorskip("arch")
    y = _ret(1100, 2)
    fc, org = garch_variance_forecast(y, min_train=1000, refit_every=50, o=1)
    assert fc.shape == org.shape
    assert np.all(np.isfinite(fc)) and np.all(fc > 0.0)


def test_gjr_garch_no_lookahead():
    """Corrupting return[c] must leave every GJR-GARCH forecast at origins t < c unchanged."""
    pytest.importorskip("arch")
    y = _ret(1300, 3)
    c = 1150  # interior, above min_train so origins exist on both sides
    y2 = y.copy()
    y2[c] *= 50.0
    fc1, org = garch_variance_forecast(y, min_train=1000, refit_every=50, o=1)
    fc2, _ = garch_variance_forecast(y2, min_train=1000, refit_every=50, o=1)
    before = org < c
    np.testing.assert_allclose(fc1[before], fc2[before], rtol=1e-10, atol=0.0)
    # The corruption must propagate to later origins (otherwise the test is vacuous).
    assert not np.allclose(fc1[~before], fc2[~before])
