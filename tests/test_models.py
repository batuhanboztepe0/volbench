"""Tests for volbench.models — volatility forecasters."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.models import (
    EWMA,
    GBRT,
    HAR,
    HARCJ,
    HARJ,
    HARQ,
    SHAR,
    AR1Log,
    HistoricalMean,
    LogHAR,
    MovingAverage,
    RandomWalk,
    _test_origins,
    average_future_variance,
    har_components,
)

MONTH_LAG = 22  # matches models.MONTH_LAG
WEEK_LAG = 5


# ---------------------------------------------------------------------------
# Synthetic series factory
# ---------------------------------------------------------------------------
def _make_rv(n: int = 700, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.standard_normal(n) * 0.4 - 9)


def _make_measures(rv: np.ndarray, seed: int = 1):
    cont = 0.7 * rv
    jump = 0.3 * rv
    rsv_minus = 0.5 * rv
    rsv_plus = 0.5 * rv
    return cont, jump, rsv_minus, rsv_plus


# ---------------------------------------------------------------------------
# average_future_variance
# ---------------------------------------------------------------------------
def test_avg_future_variance_matches_hand_computation():
    rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = average_future_variance(rv, horizon=2)
    # target[0] = mean(rv[1:3]) = (2+3)/2
    assert target[0] == pytest.approx(2.5)
    # target[1] = mean(rv[2:4]) = (3+4)/2
    assert target[1] == pytest.approx(3.5)
    # target[2] = mean(rv[3:5]) = (4+5)/2
    assert target[2] == pytest.approx(4.5)


def test_avg_future_variance_nan_at_boundary():
    rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = average_future_variance(rv, horizon=2)
    # Last two positions should be nan (window incomplete)
    assert np.isnan(target[3])
    assert np.isnan(target[4])


def test_avg_future_variance_horizon1():
    rv = np.array([1.0, 2.0, 3.0, 4.0])
    target = average_future_variance(rv, horizon=1)
    # target[t] = rv[t+1]
    assert target[0] == pytest.approx(rv[1])
    assert target[1] == pytest.approx(rv[2])
    assert target[2] == pytest.approx(rv[3])
    assert np.isnan(target[3])


# ---------------------------------------------------------------------------
# har_components
# ---------------------------------------------------------------------------
def test_har_components_shape():
    rv = _make_rv(100)
    c = har_components(rv)
    assert c.shape == (100, 3)


def test_har_components_daily_equals_rv():
    rv = _make_rv(100)
    c = har_components(rv)
    # Daily component at index t equals rv[t]
    t = MONTH_LAG  # first valid index
    assert c[t, 0] == pytest.approx(rv[t])


def test_har_components_weekly_mean():
    rv = _make_rv(100)
    c = har_components(rv)
    t = MONTH_LAG
    expected_weekly = rv[t - WEEK_LAG + 1: t + 1].mean()
    assert c[t, 1] == pytest.approx(expected_weekly)


def test_har_components_monthly_mean():
    rv = _make_rv(100)
    c = har_components(rv)
    t = MONTH_LAG
    expected_monthly = rv[t - MONTH_LAG + 1: t + 1].mean()
    assert c[t, 2] == pytest.approx(expected_monthly)


def test_har_components_nan_before_warmup():
    rv = _make_rv(100)
    c = har_components(rv)
    # Rows 0..MONTH_LAG-2 should all be nan
    assert np.all(np.isnan(c[: MONTH_LAG - 1, :]))


# ---------------------------------------------------------------------------
# Alignment test: every forecaster returns len(forecast)==len(origins)
# and origins match _test_origins
# ---------------------------------------------------------------------------
MIN_TRAIN = 120
N = 700
HORIZON = 5


@pytest.mark.parametrize("model_cls", [
    RandomWalk,
    HistoricalMean,
    EWMA,
])
def test_alignment_simple_models(model_cls):
    rv = _make_rv(N)
    model = model_cls()
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


def test_alignment_moving_average():
    rv = _make_rv(N)
    model = MovingAverage(window=10)
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


@pytest.mark.parametrize("model_cls", [HAR, LogHAR, AR1Log])
def test_alignment_har_models(model_cls):
    rv = _make_rv(N)
    model = model_cls()
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


def test_alignment_harj():
    rv = _make_rv(N)
    _, jump, _, _ = _make_measures(rv)
    model = HARJ(jump)
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


def test_alignment_harcj():
    rv = _make_rv(N)
    cont, jump, _, _ = _make_measures(rv)
    model = HARCJ(cont, jump)
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


def test_alignment_shar():
    rv = _make_rv(N)
    _, _, rsv_minus, rsv_plus = _make_measures(rv)
    model = SHAR(rsv_minus, rsv_plus)
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


def test_alignment_harq():
    rv = _make_rv(N)
    rng = np.random.default_rng(5)
    rq = np.abs(rng.standard_normal(N)) * 1e-8 + 1e-8
    model = HARQ(rq)
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


def test_alignment_gbrt():
    rv = _make_rv(N)
    model = GBRT(refit_every=50)
    fc, origins = model.oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    expected_origins = _test_origins(N, HORIZON, MIN_TRAIN)
    assert len(fc) == len(origins)
    np.testing.assert_array_equal(origins, expected_origins)


# ---------------------------------------------------------------------------
# Positivity tests for log-space models
# ---------------------------------------------------------------------------
def test_loghar_strictly_positive():
    rv = _make_rv(N)
    fc, _ = LogHAR().oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    assert np.all(fc > 0.0)


def test_ar1log_strictly_positive():
    rv = _make_rv(N)
    fc, _ = AR1Log().oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    assert np.all(fc > 0.0)


def test_gbrt_strictly_positive():
    rv = _make_rv(N)
    fc, _ = GBRT(refit_every=50).oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    assert np.all(fc > 0.0)


def test_logharj_strictly_positive():
    rv = _make_rv(N)
    _, jump, _, _ = _make_measures(rv)
    fc, _ = HARJ(jump, log=True).oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    assert np.all(fc > 0.0)


def test_logharcj_strictly_positive():
    rv = _make_rv(N)
    cont, jump, _, _ = _make_measures(rv)
    fc, _ = HARCJ(cont, jump, log=True).oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    assert np.all(fc > 0.0)


def test_logshar_strictly_positive():
    rv = _make_rv(N)
    _, _, rsv_minus, rsv_plus = _make_measures(rv)
    fc, _ = SHAR(rsv_minus, rsv_plus, log=True).oos_forecast(rv, HORIZON, min_train=MIN_TRAIN)
    assert np.all(fc > 0.0)


# ---------------------------------------------------------------------------
# Canonical no-look-ahead test
# ---------------------------------------------------------------------------
def _run_no_lookahead(model_1, model_2, rv1, rv2, horizon=5, min_train=120):
    """
    Run two model instances (one on original rv, one on corrupted rv where rv[-1]*=1000).
    Check that forecasts at origins t < n-1-h are identical.
    """
    fc1, org1 = model_1.oos_forecast(rv1, horizon, min_train=min_train)
    fc2, org2 = model_2.oos_forecast(rv2, horizon, min_train=min_train)
    n = rv1.size
    # Origins where rv[-1] cannot be in the training window or prediction feature
    mask = org1 < (n - 1 - horizon)
    assert np.any(mask), "No origins satisfy the mask — extend n or reduce horizon"
    np.testing.assert_allclose(
        fc1[mask], fc2[mask], rtol=1e-10, atol=0.0,
        err_msg="Look-ahead detected: forecast changed despite last obs outside training window"
    )


@pytest.mark.parametrize("model_name", ["HAR", "LogHAR", "AR1Log"])
def test_no_lookahead_har_variants(model_name):
    rng = np.random.default_rng(7)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    rv2 = rv.copy()
    rv2[-1] *= 1000.0

    model_map = {"HAR": HAR, "LogHAR": LogHAR, "AR1Log": AR1Log}
    cls = model_map[model_name]
    _run_no_lookahead(cls(), cls(), rv, rv2)


@pytest.mark.parametrize("log", [False, True])
def test_no_lookahead_harj(log):
    rng = np.random.default_rng(8)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    _, jump, _, _ = _make_measures(rv)
    rv2 = rv.copy()
    rv2[-1] *= 1000.0
    jump2 = jump.copy()
    jump2[-1] *= 1000.0
    _run_no_lookahead(HARJ(jump, log=log), HARJ(jump2, log=log), rv, rv2)


@pytest.mark.parametrize("log", [False, True])
def test_no_lookahead_harcj(log):
    rng = np.random.default_rng(9)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    cont, jump, _, _ = _make_measures(rv)
    rv2 = rv.copy()
    rv2[-1] *= 1000.0
    cont2 = cont.copy()
    cont2[-1] *= 1000.0
    jump2 = jump.copy()
    jump2[-1] *= 1000.0
    _run_no_lookahead(HARCJ(cont, jump, log=log), HARCJ(cont2, jump2, log=log), rv, rv2)


@pytest.mark.parametrize("log", [False, True])
def test_no_lookahead_shar(log):
    rng = np.random.default_rng(10)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    _, _, rsv_minus, rsv_plus = _make_measures(rv)
    rv2 = rv.copy()
    rv2[-1] *= 1000.0
    rsv_minus2 = rsv_minus.copy()
    rsv_minus2[-1] *= 1000.0
    rsv_plus2 = rsv_plus.copy()
    rsv_plus2[-1] *= 1000.0
    _run_no_lookahead(
        SHAR(rsv_minus, rsv_plus, log=log),
        SHAR(rsv_minus2, rsv_plus2, log=log),
        rv, rv2,
    )


def test_no_lookahead_gbrt():
    rng = np.random.default_rng(11)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    rv2 = rv.copy()
    rv2[-1] *= 1000.0
    _run_no_lookahead(GBRT(refit_every=50), GBRT(refit_every=50), rv, rv2)
