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
    ARFIMALog,
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
# Interior index at which the no-look-ahead probes corrupt their inputs. It must
# be well inside the origin range so that origins t = _LEAK_IDX - k are themselves
# valid forecast origins for every k in 1..h. Corrupting only the *final*
# observation (the previous design) cannot detect look-ahead within the h-step
# target window, because the origin that would consume rv[-1] does not exist.
_LEAK_IDX = 350


def _run_no_lookahead(model_1, model_2, rv1, rv2, horizon=5, min_train=120):
    """Probe a forecaster for look-ahead by corrupting one interior observation.

    A no-look-ahead forecaster at origin ``t`` uses ``rv`` only up to index ``t``,
    so corrupting ``rv[c]`` must leave every forecast at origins ``t < c``
    unchanged. Any model that peeks ``k >= 1`` steps ahead consumes ``rv[c]`` at
    origin ``t = c - k < c`` and is therefore caught — including leakage of
    ``1..h`` steps *within* the target window, which the old final-observation
    probe missed. ``c`` is inferred from where ``rv1`` and ``rv2`` differ.
    """
    fc1, org1 = model_1.oos_forecast(rv1, horizon, min_train=min_train)
    fc2, org2 = model_2.oos_forecast(rv2, horizon, min_train=min_train)
    np.testing.assert_array_equal(org1, org2)
    diff = np.flatnonzero(np.asarray(rv1) != np.asarray(rv2))
    assert diff.size, "rv2 is not corrupted — the probe would be vacuous"
    c = int(diff.min())
    before = org1 < c
    assert np.any(before), "No origins before the corruption index — adjust _LEAK_IDX/min_train"
    np.testing.assert_allclose(
        fc1[before], fc2[before], rtol=1e-10, atol=0.0,
        err_msg="Look-ahead detected: a forecast before the corrupted observation changed",
    )
    # Non-vacuity: the corruption must propagate to at least one later origin,
    # otherwise the probe would pass for a model that ignores its inputs entirely.
    after = org1 >= c
    assert np.any(after) and not np.allclose(fc1[after], fc2[after], rtol=1e-10, atol=0.0), (
        "Corruption did not affect any later forecast — the probe is vacuous"
    )


@pytest.mark.parametrize("model_name", ["HAR", "LogHAR", "AR1Log"])
def test_no_lookahead_har_variants(model_name):
    rng = np.random.default_rng(7)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0

    model_map = {"HAR": HAR, "LogHAR": LogHAR, "AR1Log": AR1Log}
    cls = model_map[model_name]
    _run_no_lookahead(cls(), cls(), rv, rv2)


@pytest.mark.parametrize("model_cls", [RandomWalk, HistoricalMean, MovingAverage, EWMA])
def test_no_lookahead_simple_models(model_cls):
    rng = np.random.default_rng(13)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0
    _run_no_lookahead(model_cls(), model_cls(), rv, rv2)


@pytest.mark.parametrize("log", [False, True])
def test_no_lookahead_harj(log):
    rng = np.random.default_rng(8)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    _, jump, _, _ = _make_measures(rv)
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0
    jump2 = jump.copy()
    jump2[_LEAK_IDX] *= 1000.0
    _run_no_lookahead(HARJ(jump, log=log), HARJ(jump2, log=log), rv, rv2)


@pytest.mark.parametrize("log", [False, True])
def test_no_lookahead_harcj(log):
    rng = np.random.default_rng(9)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    cont, jump, _, _ = _make_measures(rv)
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0
    cont2 = cont.copy()
    cont2[_LEAK_IDX] *= 1000.0
    jump2 = jump.copy()
    jump2[_LEAK_IDX] *= 1000.0
    _run_no_lookahead(HARCJ(cont, jump, log=log), HARCJ(cont2, jump2, log=log), rv, rv2)


@pytest.mark.parametrize("log", [False, True])
def test_no_lookahead_shar(log):
    rng = np.random.default_rng(10)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    _, _, rsv_minus, rsv_plus = _make_measures(rv)
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0
    rsv_minus2 = rsv_minus.copy()
    rsv_minus2[_LEAK_IDX] *= 1000.0
    rsv_plus2 = rsv_plus.copy()
    rsv_plus2[_LEAK_IDX] *= 1000.0
    _run_no_lookahead(
        SHAR(rsv_minus, rsv_plus, log=log),
        SHAR(rsv_minus2, rsv_plus2, log=log),
        rv, rv2,
    )


def test_no_lookahead_gbrt():
    rng = np.random.default_rng(11)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0
    _run_no_lookahead(GBRT(refit_every=50), GBRT(refit_every=50), rv, rv2)


def test_no_lookahead_harq():
    rng = np.random.default_rng(12)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    rq = np.abs(rng.standard_normal(700)) * 1e-8 + 1e-8
    rv2 = rv.copy()
    rv2[_LEAK_IDX] *= 1000.0
    rq2 = rq.copy()
    rq2[_LEAK_IDX] *= 1000.0
    _run_no_lookahead(HARQ(rq), HARQ(rq2), rv, rv2)


# ---------------------------------------------------------------------------
# ARFIMA tests
# ---------------------------------------------------------------------------
def test_arfima_no_lookahead():
    """Mutating rv[origins[k]+horizon:] must not change forecast[k].

    Covers the fracdiff warm-up region specifically: the fracdiff filter uses
    a causal slice log_rv[:t+1], so only observations <= t are touched.
    We also corrupt values well inside the series (not just the tail) to
    probe whether the filter picks up future information via the warm-up.
    """
    rng = np.random.default_rng(42)
    n = 700
    horizon = 1
    min_train = 120
    rv = np.exp(rng.standard_normal(n) * 0.4 - 9)

    model = ARFIMALog(p=1, d=0.4, trunc=50)
    fc_orig, origins = model.oos_forecast(rv, horizon, min_train=min_train)

    # For each test origin t, corrupt ALL observations after t+horizon.
    # The forecast at k must remain identical since those observations are
    # strictly outside the causal window used at origin t.
    for k in range(0, len(origins), max(1, len(origins) // 20)):
        t = origins[k]
        rv_corrupt = rv.copy()
        rv_corrupt[t + horizon :] *= 1e6  # large corruption after the origin's window
        fc_c, _ = model.oos_forecast(rv_corrupt, horizon, min_train=min_train)
        assert fc_c[k] == pytest.approx(fc_orig[k], rel=1e-10), (
            f"Look-ahead at origin {t}: forecast changed from {fc_orig[k]} "
            f"to {fc_c[k]} after corrupting rv[{t + horizon}:]"
        )

    # Specifically probe the fracdiff warm-up: corrupt a stretch in the
    # middle of the series (between two consecutive origins).
    k_mid = len(origins) // 2
    t_mid = origins[k_mid]
    rv_mid = rv.copy()
    rv_mid[t_mid + horizon :] *= 1e6
    fc_mid, _ = model.oos_forecast(rv_mid, horizon, min_train=min_train)
    assert fc_mid[k_mid] == pytest.approx(fc_orig[k_mid], rel=1e-10), (
        "Fracdiff warm-up look-ahead: forecast changed after corrupting future obs"
    )


def test_arfima_positivity():
    """All ARFIMA forecasts must be strictly positive (Invariant 2)."""
    rng = np.random.default_rng(99)
    rv = np.exp(rng.standard_normal(700) * 0.4 - 9)
    model = ARFIMALog(p=1, d=0.4, trunc=50)
    fc, origins = model.oos_forecast(rv, horizon=1, min_train=120)
    assert fc.size > 0, "No forecasts produced"
    assert np.all(fc > 0.0), f"Non-positive forecast found: min={fc.min()}"


def test_arfima_recovers_long_memory():
    """ARFIMA beats AR1Log on QLIKE for a true ARFIMA(0,0.4,0) log-RV path.

    Simulates log-RV via fractional differencing of Gaussian noise (d=0.4),
    then compares OOS QLIKE of ARFIMALog vs AR1Log.  The fractional-integration
    structure is exactly what the ARFIMA model captures and AR(1) cannot.
    """
    from volbench.losses import qlike

    rng = np.random.default_rng(2024)
    n = 1200  # longer series to let the long-memory signal dominate

    # Simulate ARFIMA(0, 0.4, 0) log-RV: invert fracdiff on white noise.
    # log_rv[t] = sum_k w_k * eps[t-k] where w are fracdiff weights for -d
    d = 0.4
    trunc = 300
    from volbench.models import _fracdiff_weights

    # Weights for the MA-inf representation (inverse of fracdiff operator)
    # = fracdiff weights with d replaced by -d
    w_inv = _fracdiff_weights(-d, trunc)
    eps = rng.standard_normal(n + trunc)
    log_rv_sim = np.array([
        float(np.dot(w_inv[:min(t + 1, trunc)], eps[t : t - min(t + 1, trunc) if min(t + 1, trunc) <= t else None : -1]))
        for t in range(n + trunc)
    ])[trunc:]  # discard warm-up
    # Convert to variance scale
    rv_sim = np.exp(log_rv_sim)

    horizon = 1
    min_train = 300

    arfima = ARFIMALog(p=1, d=0.4, trunc=100)
    ar1 = AR1Log()

    fc_arfima, orig_arfima = arfima.oos_forecast(rv_sim, horizon, min_train=min_train)
    fc_ar1, orig_ar1 = ar1.oos_forecast(rv_sim, horizon, min_train=min_train)

    # Align on common origins
    common = np.intersect1d(orig_arfima, orig_ar1)
    assert common.size >= 100, f"Too few common origins: {common.size}"

    idx_a = np.searchsorted(orig_arfima, common)
    idx_b = np.searchsorted(orig_ar1, common)

    realized_common = np.array([
        rv_sim[t + 1] for t in common
    ])  # h=1: realized = rv[t+1]

    ql_arfima = qlike(realized_common, fc_arfima[idx_a]).mean()
    ql_ar1 = qlike(realized_common, fc_ar1[idx_b]).mean()

    assert ql_arfima < ql_ar1, (
        f"ARFIMA did not beat AR1Log on long-memory data: "
        f"QLIKE_ARFIMA={ql_arfima:.6f}, QLIKE_AR1={ql_ar1:.6f}"
    )
