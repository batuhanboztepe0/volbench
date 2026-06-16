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
    """Corrupting the final observation must not change earlier forecasts."""
    rv = _rv(n=600, seed=3)
    cont, jump, rm, rp = _measures(rv)
    h, min_train = 5, 150

    if kind == "plain":
        m1 = plain_ml("lgbm", refit_every=200, random_state=0)
        m2 = plain_ml("lgbm", refit_every=200, random_state=0)
    else:
        m1 = enriched_ml("lgbm", cont, jump, rm, rp, refit_every=200, random_state=0)
        rv2c, j2, r2m, r2p = _measures(rv.copy())
        m2 = enriched_ml("lgbm", rv2c, j2, r2m, r2p, refit_every=200, random_state=0)

    fc1, org = m1.oos_forecast(rv, h, min_train)
    rv2 = rv.copy()
    rv2[-1] *= 1000.0
    # For enriched, the measure series are derived from rv but bound at build
    # time; corrupting only rv[-1] still cannot reach origins t < n-1-h.
    fc2, _ = m2.oos_forecast(rv2, h, min_train)

    mask = org < (rv.size - 1 - h)
    assert mask.sum() > 0
    assert np.allclose(fc1[mask], fc2[mask])


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
