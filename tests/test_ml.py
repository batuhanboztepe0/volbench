"""Tests for the leakage-free ML forecasters (Tier 2D)."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.ml import (
    EnsembleForecaster,
    MLForecaster,
    enriched_features,
    enriched_ml,
    plain_features,
    plain_ml,
)
from volbench.models import LogHAR

lgb = pytest.importorskip("lightgbm")


def _rv(n=700, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.standard_normal(n) * 0.4 - 9.0)


def _measures(rv):
    return 0.7 * rv, 0.3 * rv, 0.5 * rv, 0.5 * rv  # cont, jump, rsv_minus, rsv_plus


def test_plain_features_shape():
    rv = _rv()
    X = plain_features(rv, n_lags=5)
    assert X.shape == (rv.size, 8)  # 3 HAR comps + 5 lags
    assert np.isfinite(X[-1]).all()  # finite at the end of the sample


def test_enriched_features_shape_and_peers():
    rv = _rv()
    cont, jump, rm, rp = _measures(rv)
    X = enriched_features(rv, cont, jump, rm, rp)
    assert X.shape[0] == rv.size
    assert X.shape[1] == 3 + 3 + 3 + 1 + 1  # rv/cont/jump HAR + 2 semivariances
    peer = np.column_stack([rv * 1.1, rv * 0.9])
    Xp = enriched_features(rv, cont, jump, rm, rp, peer_rv=peer)
    assert Xp.shape[1] == X.shape[1] + 2


def test_mlforecaster_alignment_and_positive():
    rv = _rv()
    m = plain_ml("lgbm", refit_every=300)
    fc, org = m.oos_forecast(rv, horizon=1, min_train=200)
    assert fc.shape == org.shape
    assert np.all(fc > 0)               # log-space => strictly positive
    assert np.all(np.isfinite(fc))


def test_unknown_learner_raises():
    with pytest.raises(ValueError):
        MLForecaster("does-not-exist", lambda rv: plain_features(rv), name="x")


@pytest.mark.parametrize("kind", ["plain", "enriched"])
def test_ml_no_lookahead(kind):
    """Corrupting an interior observation must not change earlier forecasts."""
    n, h, min_train = 600, 5, 150
    c = 300  # interior corruption index, well inside the origin range

    rv = _rv(n=n, seed=3)
    rv2 = rv.copy()
    rv2[c] *= 1000.0

    if kind == "plain":
        m1 = plain_ml("lgbm", refit_every=200, random_state=0)
        m2 = plain_ml("lgbm", refit_every=200, random_state=0)
    else:
        cont, jump, rm, rp = _measures(rv)
        m1 = enriched_ml("lgbm", cont, jump, rm, rp, refit_every=200, random_state=0)
        cont2, jump2, rm2, rp2 = _measures(rv2)
        m2 = enriched_ml("lgbm", cont2, jump2, rm2, rp2, refit_every=200, random_state=0)

    fc1, org = m1.oos_forecast(rv, h, min_train)
    fc2, org2 = m2.oos_forecast(rv2, h, min_train)
    np.testing.assert_array_equal(org, org2)

    # (a) Forecasts at all origins before the corruption index must be unchanged.
    before = org < c
    assert before.sum() > 0, "No origins before the corruption index"
    np.testing.assert_allclose(
        fc1[before], fc2[before], rtol=1e-10, atol=0.0,
        err_msg="Look-ahead detected: a forecast before the corrupted observation changed",
    )

    # (b) Non-vacuity: at least one origin >= c must have a changed forecast.
    after = org >= c
    assert after.sum() > 0 and not np.allclose(fc1[after], fc2[after], rtol=1e-10, atol=0.0), (
        "Corruption did not affect any later forecast — the probe is vacuous"
    )


def test_ensemble_averages_members():
    rv = _rv()
    members = [LogHAR(), plain_ml("lgbm", refit_every=300)]
    combo = EnsembleForecaster(members, name="Combo")
    fc, org = combo.oos_forecast(rv, horizon=1, min_train=200)
    assert np.all(fc > 0)
    # Combo forecast equals the mean of the members on the common origins.
    indiv = []
    for mem in members:
        f, o = mem.oos_forecast(rv, horizon=1, min_train=200)
        lookup = dict(zip(o.tolist(), f.tolist()))
        indiv.append(np.array([lookup[x] for x in org.tolist()]))
    assert np.allclose(fc, np.mean(np.column_stack(indiv), axis=1))


def test_ensemble_needs_two_members():
    with pytest.raises(ValueError):
        EnsembleForecaster([LogHAR()])
