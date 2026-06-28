"""Tests for volbench.evaluation: Diebold-Mariano test and MCS."""

from __future__ import annotations

import numpy as np
import pytest

from volbench.evaluation import clark_west, diebold_mariano, model_confidence_set


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
# clark_west (nested-model test)
# ---------------------------------------------------------------------------
def _cw_setup(n: int = 400, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realized series with a genuinely-better unrestricted forecast."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(1.0, 3.0, n)
    f_unrestricted = y + rng.normal(0.0, 0.05, n)  # close to truth
    f_restricted = y + rng.normal(0.0, 0.30, n)    # further from truth
    return y, f_restricted, f_unrestricted


def test_cw_result_keys():
    y, f_r, f_u = _cw_setup()
    res = clark_west(y, f_r, f_u)
    assert set(res.keys()) == {"mean_adj", "cw_stat", "p_value", "favors_unrestricted", "n"}


def test_cw_favors_unrestricted_when_larger_model_better():
    """A genuinely closer unrestricted forecast yields a positive, significant CW."""
    y, f_r, f_u = _cw_setup()
    res = clark_west(y, f_r, f_u)
    assert res["favors_unrestricted"] == pytest.approx(1.0)
    assert res["cw_stat"] > 0
    assert res["p_value"] < 0.05


def test_cw_one_sided_pvalue_in_unit_interval():
    y, f_r, f_u = _cw_setup()
    res = clark_west(y, f_r, f_u)
    assert 0.0 <= res["p_value"] <= 1.0


def test_cw_identical_forecasts_degenerate():
    """f_r == f_u gives a zero adjusted differential -> nan statistic, no preference."""
    rng = np.random.default_rng(3)
    y = rng.uniform(1.0, 3.0, 300)
    f = y + rng.normal(0.0, 0.1, 300)
    res = clark_west(y, f, f)
    assert np.isnan(res["cw_stat"])
    assert np.isnan(res["p_value"])
    assert res["favors_unrestricted"] == pytest.approx(0.0)


def test_cw_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        clark_west(np.ones(100), np.ones(100), np.ones(101))


def test_cw_raises_on_too_short():
    with pytest.raises(ValueError):
        clark_west(np.ones(5), np.ones(5), np.ones(5))


def test_cw_horizon_sets_newey_west_lag():
    """horizon=h must default the HAC truncation lag to h-1 (matches explicit lag)."""
    y, f_r, f_u = _cw_setup(n=600, seed=5)
    res_h5 = clark_west(y, f_r, f_u, horizon=5)
    res_h5_explicit = clark_west(y, f_r, f_u, horizon=5, lag=4)
    assert res_h5["cw_stat"] == pytest.approx(res_h5_explicit["cw_stat"], rel=1e-12)


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


# ---------------------------------------------------------------------------
# Reference cross-checks: the from-scratch tests must agree with independent
# implementations in statsmodels / scipy / arch. This is what backs the
# "matches reference libraries" claim in the README and report.
# ---------------------------------------------------------------------------
def _ar1(n: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def _nw_lrv_reference(series: np.ndarray, trunc: int) -> float:
    """Newey-West LRV via statsmodels.acovf (biased, demeaned) + Bartlett weights."""
    from statsmodels.tsa.stattools import acovf

    g = acovf(series, nlag=trunc, adjusted=False, demean=True, fft=False)
    return float(g[0] + 2.0 * sum((1.0 - k / (trunc + 1.0)) * g[k] for k in range(1, trunc + 1)))


def test_dm_matches_statsmodels_scipy_reference():
    """DM (+HLN) statistic and p-value match statsmodels.acovf + scipy.stats.t."""
    pytest.importorskip("statsmodels")
    scipy_stats = pytest.importorskip("scipy.stats")
    n, horizon = 240, 5
    loss_a = np.abs(_ar1(n, 0.6, 1)) + 1.00
    loss_b = np.abs(_ar1(n, 0.6, 2)) + 1.05
    out = diebold_mariano(loss_a, loss_b, horizon=horizon)

    d = loss_a - loss_b
    trunc = horizon - 1
    lrv = _nw_lrv_reference(d, trunc)
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n)
    dm_corrected = dm * hln
    p_ref = 2.0 * float(scipy_stats.t.sf(abs(dm_corrected), df=n - 1))

    assert out["dm_stat"] == pytest.approx(dm_corrected, rel=1e-10)
    assert out["p_value"] == pytest.approx(p_ref, rel=1e-10)


def test_clark_west_matches_statsmodels_scipy_reference():
    """Clark-West statistic and one-sided p-value match statsmodels.acovf + scipy normal."""
    pytest.importorskip("statsmodels")
    scipy_stats = pytest.importorskip("scipy.stats")
    n, horizon = 300, 3
    y = np.abs(_ar1(n, 0.5, 7)) + 2.0
    f_r = y + 0.30 * _ar1(n, 0.3, 8)
    f_u = y + 0.20 * _ar1(n, 0.3, 9)
    out = clark_west(y, f_r, f_u, horizon=horizon)

    f_adj = (y - f_r) ** 2 - ((y - f_u) ** 2 - (f_r - f_u) ** 2)
    lrv = _nw_lrv_reference(f_adj, horizon - 1)
    cw = f_adj.mean() / np.sqrt(lrv / n)
    p_ref = float(scipy_stats.norm.sf(cw))

    assert out["cw_stat"] == pytest.approx(cw, rel=1e-10)
    assert out["p_value"] == pytest.approx(p_ref, rel=1e-10)


def test_mcs_included_set_matches_arch():
    """On a clearly separated panel, the MCS included set agrees with arch.bootstrap.MCS."""
    pd = pytest.importorskip("pandas")
    arch_bootstrap = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(3)
    n = 250
    common = np.abs(rng.standard_normal(n))  # shared shock keeps differentials clean
    losses = {
        "best": common + 0.10 * np.abs(rng.standard_normal(n)),
        "mid": common + 1.0 + 0.10 * np.abs(rng.standard_normal(n)),
        "worst": common + 2.0 + 0.10 * np.abs(rng.standard_normal(n)),
    }
    ours = model_confidence_set(losses, alpha=0.10, block_length=10, reps=2000, seed=0)

    arch_mcs = arch_bootstrap.MCS(
        pd.DataFrame(losses), size=0.10, reps=2000, block_size=10, method="R", seed=0
    )
    arch_mcs.compute()
    arch_included = set(arch_mcs.included)

    # Strong separation: both keep only "best".
    assert set(ours.included) == {"best"}
    assert arch_included == {"best"}
    assert set(ours.included) == arch_included
