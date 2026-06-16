"""Expanding-window backtest harness.

:func:`run_backtest` runs a suite of forecasters over a single realized-variance
series, scores them on a **common, intersected** set of test origins (so every
model is judged on identical data — ``ROADMAP.md`` invariant 5), and runs the
pairwise Diebold-Mariano test and the Model Confidence Set on the proxy-robust
losses. It is the single entry point the Track-1 scripts call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce

import numpy as np

from .evaluation import MCSResult, diebold_mariano, model_confidence_set
from .losses import LOSS_FUNCTIONS, RANKING_LOSSES, mean_loss
from .models import DEFAULT_MIN_TRAIN, VolForecaster, average_future_variance, default_models

_DEFAULT_MCS_ALPHA: float = 0.10  # 90% Model Confidence Set
_DEFAULT_BLOCK_LENGTH: int = 10
_DEFAULT_MCS_REPS: int = 2000


@dataclass
class BacktestResult:
    """Container for the outcome of :func:`run_backtest`.

    Attributes
    ----------
    horizon : int
        Forecast horizon scored.
    origins : np.ndarray
        Common test origins shared by every model.
    realized : np.ndarray
        Realized target (average future variance) at the common origins.
    forecasts : dict[str, np.ndarray]
        Per-model forecast vectors aligned to ``origins``.
    losses : dict[str, dict[str, np.ndarray]]
        ``loss_name -> model -> per-observation loss array``.
    mean_losses : dict[str, dict[str, float]]
        ``loss_name -> model -> mean loss``.
    mcs : dict[str, MCSResult]
        Model Confidence Set per ranking loss.
    dm_vs_har : dict[str, dict[str, dict[str, float]]]
        ``loss_name -> model -> Diebold-Mariano result vs the benchmark``.
    model_names : list[str]
        Models scored, in suite order.
    benchmark : str
        Name of the DM reference model.
    """

    horizon: int
    origins: np.ndarray
    realized: np.ndarray
    forecasts: dict[str, np.ndarray]
    losses: dict[str, dict[str, np.ndarray]]
    mean_losses: dict[str, dict[str, float]]
    mcs: dict[str, MCSResult]
    dm_vs_har: dict[str, dict[str, dict[str, float]]]
    model_names: list[str] = field(default_factory=list)
    benchmark: str = "HAR"


def run_backtest(
    rv: np.ndarray,
    horizon: int,
    models: list[VolForecaster] | None = None,
    min_train: int = DEFAULT_MIN_TRAIN,
    mcs_alpha: float = _DEFAULT_MCS_ALPHA,
    mcs_reps: int = _DEFAULT_MCS_REPS,
    block_length: int = _DEFAULT_BLOCK_LENGTH,
    seed: int | None = 0,
    benchmark: str = "HAR",
) -> BacktestResult:
    """Score a model suite on one realized-variance series.

    Parameters
    ----------
    rv : np.ndarray
        Realized-variance series (variance scale).
    horizon : int
        Forecast horizon in days.
    models : list[VolForecaster], optional
        Models to score; defaults to :func:`volbench.models.default_models`.
    min_train : int, default :data:`volbench.models.DEFAULT_MIN_TRAIN`
        First test origin.
    mcs_alpha : float, default 0.10
        MCS significance level (90% set).
    mcs_reps : int, default 2000
        Bootstrap replications for the MCS.
    block_length : int, default 10
        Moving-block-bootstrap block length for the MCS.
    seed : int, optional, default 0
        Seed for the MCS bootstrap.
    benchmark : str, default ``"HAR"``
        Reference model for the Diebold-Mariano comparisons.

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If no common test origins exist or the benchmark is absent.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    suite = models if models is not None else default_models()
    names = [m.name for m in suite]
    if benchmark not in names:
        raise ValueError(f"benchmark {benchmark!r} not in model suite {names}")

    # Run every model, then restrict to the origins they all share (invariant 5).
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in suite:
        fc, origins = model.oos_forecast(rv, horizon, min_train)
        raw[model.name] = (np.asarray(fc, dtype=float), np.asarray(origins, dtype=int))

    common = reduce(np.intersect1d, (o for _, o in raw.values()))
    if common.size == 0:
        raise ValueError("no common test origins across models (series too short?)")

    target = average_future_variance(rv, horizon)
    realized = target[common]

    forecasts: dict[str, np.ndarray] = {}
    for name, (fc, origins) in raw.items():
        lookup = dict(zip(origins.tolist(), fc.tolist()))
        forecasts[name] = np.array([lookup[o] for o in common.tolist()], dtype=float)

    # Per-observation and mean losses for every loss function.
    losses: dict[str, dict[str, np.ndarray]] = {}
    mean_losses: dict[str, dict[str, float]] = {}
    for loss_name, func in LOSS_FUNCTIONS.items():
        per_model = {name: func(realized, forecasts[name]) for name in names}
        losses[loss_name] = per_model
        mean_losses[loss_name] = {name: mean_loss(per_model[name]) for name in names}

    # MCS + Diebold-Mariano on the proxy-robust ranking losses only.
    mcs: dict[str, MCSResult] = {}
    dm_vs_har: dict[str, dict[str, dict[str, float]]] = {}
    for loss_name in RANKING_LOSSES:
        per_model = losses[loss_name]
        mcs[loss_name] = model_confidence_set(
            per_model, alpha=mcs_alpha, block_length=block_length, reps=mcs_reps, seed=seed
        )
        bench_loss = per_model[benchmark]
        dm_vs_har[loss_name] = {
            name: diebold_mariano(per_model[name], bench_loss, horizon=horizon)
            for name in names
            if name != benchmark
        }

    return BacktestResult(
        horizon=horizon,
        origins=common,
        realized=realized,
        forecasts=forecasts,
        losses=losses,
        mean_losses=mean_losses,
        mcs=mcs,
        dm_vs_har=dm_vs_har,
        model_names=names,
        benchmark=benchmark,
    )
