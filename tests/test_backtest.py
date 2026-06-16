"""Tests for volbench.backtest — end-to-end backtest harness."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.backtest import BacktestResult, run_backtest
from volbench.losses import RANKING_LOSSES
from volbench.models import HAR, HistoricalMean, LogHAR, RandomWalk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_rv(n: int = 700, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.standard_normal(n) * 0.4 - 9)


# ---------------------------------------------------------------------------
# Small end-to-end backtest
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_backtest():
    rv = _make_rv(700)
    return run_backtest(rv, horizon=1, min_train=200, mcs_reps=100, seed=0)


def test_backtest_returns_backtest_result(small_backtest):
    assert isinstance(small_backtest, BacktestResult)


def test_backtest_all_forecasts_share_origins_length(small_backtest):
    res = small_backtest
    n_origins = len(res.origins)
    for name, fc in res.forecasts.items():
        assert len(fc) == n_origins, f"Model {name} has {len(fc)} forecasts, expected {n_origins}"


def test_backtest_mean_losses_has_all_loss_keys(small_backtest):
    res = small_backtest
    assert "QLIKE" in res.mean_losses
    assert "MSE-var" in res.mean_losses
    assert "MSE-vol" in res.mean_losses


def test_backtest_mean_losses_has_all_models(small_backtest):
    res = small_backtest
    for loss_name in res.mean_losses:
        for model_name in res.model_names:
            assert model_name in res.mean_losses[loss_name], (
                f"Model {model_name} missing from mean_losses[{loss_name!r}]"
            )


def test_backtest_mcs_has_ranking_losses(small_backtest):
    res = small_backtest
    for loss_name in RANKING_LOSSES:
        assert loss_name in res.mcs, f"MCS missing loss {loss_name!r}"


def test_backtest_dm_vs_har_excludes_benchmark(small_backtest):
    res = small_backtest
    for loss_name, dm_dict in res.dm_vs_har.items():
        assert res.benchmark not in dm_dict, (
            f"Benchmark {res.benchmark!r} should not appear in dm_vs_har[{loss_name!r}]"
        )


def test_backtest_dm_vs_har_includes_non_benchmark_models(small_backtest):
    res = small_backtest
    for loss_name in RANKING_LOSSES:
        for model_name in res.model_names:
            if model_name != res.benchmark:
                assert model_name in res.dm_vs_har[loss_name], (
                    f"Model {model_name!r} missing from dm_vs_har[{loss_name!r}]"
                )


def test_backtest_realized_length_matches_origins(small_backtest):
    res = small_backtest
    assert len(res.realized) == len(res.origins)


def test_backtest_mcs_results_non_empty(small_backtest):
    for loss_name, mcs_res in small_backtest.mcs.items():
        assert len(mcs_res.included) > 0, f"MCS for {loss_name!r} is empty"


# ---------------------------------------------------------------------------
# ValueError when benchmark not in suite
# ---------------------------------------------------------------------------
def test_backtest_raises_if_benchmark_absent():
    rv = _make_rv(700)
    models = [RandomWalk(), HistoricalMean()]
    with pytest.raises(ValueError, match="benchmark"):
        run_backtest(rv, horizon=1, models=models, min_train=200, benchmark="HAR", mcs_reps=50)


# ---------------------------------------------------------------------------
# Custom model suite
# ---------------------------------------------------------------------------
def test_backtest_custom_model_suite():
    rv = _make_rv(700)
    models = [RandomWalk(), HAR(), LogHAR()]
    res = run_backtest(
        rv, horizon=1, models=models, min_train=200, mcs_reps=50, seed=0, benchmark="HAR"
    )
    assert set(res.model_names) == {"RW", "HAR", "LogHAR"}
    assert "RW" in res.dm_vs_har["QLIKE"]
    assert "LogHAR" in res.dm_vs_har["QLIKE"]
    assert "HAR" not in res.dm_vs_har["QLIKE"]
