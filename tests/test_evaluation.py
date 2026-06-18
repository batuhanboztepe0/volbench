"""Tests for volbench.evaluation — Diebold-Mariano test and MCS."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.evaluation import diebold_mariano, model_confidence_set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _loss_array(n: int = 300, level: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(n)) + level


# ---------------------------------------------------------------------------
# diebold_mariano
# ---------------------------------------------------------------------------
def test_dm_identical_losses_mean_diff_zero():
    loss = _loss_array(300)
    res = diebold_mariano(loss, loss)
    assert res["mean_diff"] == pytest.approx(0.0, abs=1e-12)


def test_dm_identical_losses_favored_zero():
    loss = _loss_array(300)
    res = diebold_mariano(loss, loss)
    # With identical losses favored should be 0
    assert res["favored"] == pytest.approx(0.0)


def test_dm_a_uniformly_smaller_favors_a():
    """When A is uniformly smaller, favored == -1 (A wins) and p_value small."""
    rng = np.random.default_rng(7)
    n = 400
    loss_a = rng.uniform(0.1, 0.5, n)
    loss_b = rng.uniform(1.0, 2.0, n)
    res = diebold_mariano(loss_a, loss_b)
    assert res["favored"] == pytest.approx(-1.0)
    assert res["p_value"] < 0.05


def test_dm_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(100), np.ones(101))


def test_dm_raises_on_too_short():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5), np.ones(5))


def test_dm_result_keys():
    loss = _loss_array(200)
    res = diebold_mariano(loss, loss * 1.01)
    assert set(res.keys()) == {"mean_diff", "dm_stat", "p_value", "favored", "n"}


def test_dm_horizon_sets_newey_west_lag_and_matters():
    """At h>1 the loss differentials are autocorrelated; the HAC path must run.

    Multi-step (overlapping) forecast errors give an MA(h-1) loss differential.
    `horizon=h` must default the Newey-West truncation lag to h-1 (so it matches an
    explicit `lag=h-1`), and that correction must actually change the statistic
    relative to ignoring the autocorrelation (lag 0). Previously no test exercised
    horizon>1, leaving this path unguarded.
    """
    rng = np.random.default_rng(123)
    n = 1000
    eps = rng.standard_normal(n + 4)
    # MA(4) loss differential, as produced by overlapping h=5 forecast errors.
    d = eps[4:] + eps[3:-1] + eps[2:-2] + eps[1:-3] + eps[:-4]
    loss_a = 10.0 + d
    loss_b = np.full(n, 10.0)

    res_h5 = diebold_mariano(loss_a, loss_b, horizon=5)
    res_h5_explicit = diebold_mariano(loss_a, loss_b, horizon=5, lag=4)
    # horizon=5 must default the HAC lag to h-1=4.
    assert res_h5["dm_stat"] == pytest.approx(res_h5_explicit["dm_stat"], rel=1e-12)

    # Ignoring the positive autocorrelation (lag 0) under-estimates the long-run
    # variance and inflates the statistic, so the HAC correction genuinely bites.
    res_lag0 = diebold_mariano(loss_a, loss_b, horizon=5, lag=0)
    assert abs(res_lag0["dm_stat"]) > abs(res_h5["dm_stat"])


def test_dm_n_matches_input():
    n = 250
    loss = _loss_array(n)
    res = diebold_mariano(loss, loss)
    assert int(res["n"]) == n


# ---------------------------------------------------------------------------
# model_confidence_set
# ---------------------------------------------------------------------------
def test_mcs_includes_clearly_best_model():
    """The model with the lowest loss should be in the confidence set."""
    rng = np.random.default_rng(42)
    n = 300
    # "best" model: low-variance losses around 0.2
    losses = {
        "best": rng.uniform(0.1, 0.3, n),
        "medium": rng.uniform(1.0, 2.0, n),
        "worst": rng.uniform(5.0, 10.0, n),
    }
    res = model_confidence_set(losses, alpha=0.10, block_length=5, reps=100, seed=0)
    assert "best" in res.included


def test_mcs_excludes_clearly_worst_model():
    """The model with clearly the highest loss should be eliminated."""
    rng = np.random.default_rng(42)
    n = 300
    losses = {
        "best": rng.uniform(0.1, 0.3, n),
        "medium": rng.uniform(0.5, 1.0, n),
        "worst": rng.uniform(10.0, 20.0, n),
    }
    res = model_confidence_set(losses, alpha=0.10, block_length=5, reps=100, seed=0)
    assert "worst" not in res.included


def test_mcs_pvalues_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 200
    losses = {
        "A": rng.uniform(0.1, 1.0, n),
        "B": rng.uniform(1.0, 5.0, n),
    }
    res = model_confidence_set(losses, alpha=0.10, block_length=5, reps=50, seed=1)
    for name, pv in res.p_values.items():
        assert 0.0 <= pv <= 1.0, f"p_value for {name} is {pv}"


def test_mcs_best_model_pvalue_is_1():
    """The last surviving model's MCS p-value should be 1.0."""
    rng = np.random.default_rng(3)
    n = 200
    losses = {
        "clearly_best": rng.uniform(0.01, 0.02, n),
        "bad": rng.uniform(10.0, 20.0, n),
    }
    res = model_confidence_set(losses, alpha=0.10, block_length=5, reps=100, seed=0)
    assert res.p_values["clearly_best"] == pytest.approx(1.0)


def test_mcs_included_non_empty():
    rng = np.random.default_rng(99)
    n = 200
    losses = {
        "A": rng.uniform(0.1, 1.0, n),
        "B": rng.uniform(0.2, 1.2, n),
    }
    res = model_confidence_set(losses, alpha=0.10, block_length=5, reps=50, seed=0)
    assert len(res.included) > 0


def test_mcs_raises_on_single_model():
    with pytest.raises(ValueError):
        model_confidence_set({"only": np.ones(100)}, reps=10)
