"""Multivariate / volatility-spillover extension for the realized-variance benchmark.

Implements a cross-index HAR model (:class:`CrossHAR`) that augments the standard
log-HAR features with peers' lagged daily log realized variance, together with
alignment helpers and a one-call spillover backtest convenience function.

The no-lookahead invariant is preserved throughout: training at origin ``t`` uses
only rows ``s`` with ``s + horizon <= t``. Peer features at time ``s`` are
contemporaneous-day realized variances of the *peers*, which are valid predictors of
the *future target* so long as the regression target is ``average_future_variance``
(strictly future). The peer features are NOT future target values, so no leakage.

All models work in log-space and map back with the lognormal correction
``exp(mu + 0.5 * resid_var)`` to guarantee strictly positive forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import RealizedDataset
from .evaluation import clark_west
from .models import (
    _LOG_FLOOR,
    DEFAULT_MIN_TRAIN,
    MAX_LAG,
    LogHAR,
    RandomWalk,
    VolForecaster,
    _ols_fit,
    _test_origins,
    average_future_variance,
    har_components,
)

__all__ = ["CrossHAR", "align_panel", "spillover_backtest"]


class CrossHAR(VolForecaster):
    """Log-space HAR augmented with peers' lagged daily log realized variance.

    Features at forecast origin ``t``:
    - Target's own HAR components (log RV_d, log RV_w, log RV_m) from
      :func:`~volbench.models.har_components`.
    - Each peer's daily log RV at time ``t`` (contemporaneous peer observation,
      valid since it predicts the *future* target and does not use future target
      data).

    The model is estimated by OLS on the log target, then mapped back with the
    lognormal bias correction ``exp(mu + 0.5 * resid_var)`` to obtain strictly
    positive variance forecasts.

    Parameters
    ----------
    peer_rv : np.ndarray
        2-D array of shape ``(n, n_peers)`` of peer realized variances aligned
        (same length and dates) with the target rv passed to :meth:`oos_forecast`.
    name : str, default ``"CrossHAR"``
        Display name for this model instance.
    """

    def __init__(self, peer_rv: np.ndarray, name: str = "CrossHAR") -> None:
        self.peer_rv = np.asarray(peer_rv, dtype=float)
        if self.peer_rv.ndim == 1:
            self.peer_rv = self.peer_rv[:, np.newaxis]
        if self.peer_rv.ndim != 2:
            raise ValueError(f"peer_rv must be 1-D or 2-D, got shape {self.peer_rv.shape}")
        self.name = name

    def oos_forecast(
        self, rv: np.ndarray, horizon: int, min_train: int = DEFAULT_MIN_TRAIN
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expanding-window OOS forecast augmented with peer log-RV.

        Parameters
        ----------
        rv : np.ndarray
            Target realized-variance series (length ``n``).
        horizon : int
            Forecast horizon in days.
        min_train : int, default :data:`~volbench.models.DEFAULT_MIN_TRAIN`
            First test origin index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(forecasts, origins)`` where ``forecasts[k]`` is the predicted
            average variance made at ``origins[k]``.

        Raises
        ------
        ValueError
            If ``peer_rv`` length does not match ``rv``.
        """
        rv = np.asarray(rv, dtype=float).ravel()
        n = rv.size
        if self.peer_rv.shape[0] != n:
            raise ValueError(
                f"peer_rv has {self.peer_rv.shape[0]} rows but rv has {n} elements"
            )

        # Precompute target's HAR components (n, 3) and log-peer matrix (n, n_peers).
        comps = har_components(rv)                               # (n, 3)
        log_peer = np.log(np.maximum(self.peer_rv, _LOG_FLOOR)) # (n, n_peers)
        target = average_future_variance(rv, horizon)            # (n,) with nan at tail

        origins = _test_origins(n, horizon, min_train)
        forecasts = np.empty(origins.size)

        for k, t in enumerate(origins):
            # Training rows: s in [MAX_LAG-1, t-horizon] ensures all features
            # are formed (MAX_LAG-1) and the target window is closed by t.
            last_train = t - horizon  # inclusive upper bound
            rows = np.arange(MAX_LAG - 1, last_train + 1)

            # Build train design: [1, log(RV_d), log(RV_w), log(RV_m), log(peer_0), ...]
            log_comps_train = np.log(np.maximum(comps[rows], _LOG_FLOOR))  # (m, 3)
            peer_train = log_peer[rows]                                     # (m, n_peers)
            y_train = target[rows]

            # Drop rows where any feature or target is non-finite.
            valid = (
                np.isfinite(log_comps_train).all(axis=1)
                & np.isfinite(peer_train).all(axis=1)
                & np.isfinite(y_train)
            )
            log_comps_train = log_comps_train[valid]
            peer_train = peer_train[valid]
            y_train = y_train[valid]

            X_train = np.column_stack(
                [np.ones(y_train.size), log_comps_train, peer_train]
            )
            y_log = np.log(np.maximum(y_train, _LOG_FLOOR))
            beta = _ols_fit(X_train, y_log)

            # Prediction at origin t.
            log_comps_pred = np.log(np.maximum(comps[t : t + 1], _LOG_FLOOR))  # (1, 3)
            peer_pred = log_peer[t : t + 1]                                     # (1, n_peers)
            x_pred = np.column_stack(
                [np.ones(1), log_comps_pred, peer_pred]
            )
            mu = float((x_pred @ beta).ravel()[0])

            # Lognormal bias correction: exp(mu + 0.5 * resid_var).
            resid = y_log - X_train @ beta
            s2 = float(resid @ resid) / max(resid.size - X_train.shape[1], 1)
            forecasts[k] = float(np.exp(mu + 0.5 * s2))

        return forecasts, origins


def align_panel(
    dataset: RealizedDataset, tickers: list[str]
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Intersect date indices and return equal-length aligned rv5 arrays.

    Different indices in the Oxford-Man panel have different date coverage.
    This function finds the intersection of trading dates across all requested
    tickers so a target and its peers share a single timeline.

    Parameters
    ----------
    dataset : RealizedDataset
        The realized dataset (from :func:`~volbench.data.load_oxford_rv`).
    tickers : list[str]
        Subset of tickers to align (must all be present in ``dataset``).

    Returns
    -------
    tuple[pandas.DatetimeIndex, dict[str, np.ndarray]]
        ``(dates, rv_dict)`` where ``dates`` is the common date index and
        ``rv_dict`` maps each ticker to its aligned ``rv5`` array of equal
        length.
    """
    if not tickers:
        raise ValueError("tickers must be non-empty")

    # Start with the full date index of the first ticker, then intersect.
    common_dates = dataset.frame(tickers[0]).index
    for tk in tickers[1:]:
        common_dates = common_dates.intersection(dataset.frame(tk).index)

    common_dates = common_dates.sort_values()

    rv_dict: dict[str, np.ndarray] = {}
    for tk in tickers:
        frame = dataset.frame(tk)
        rv_dict[tk] = frame.loc[common_dates, "rv5"].to_numpy(dtype=float)

    return common_dates, rv_dict


def spillover_backtest(
    dataset: RealizedDataset,
    target: str,
    peers: list[str],
    horizon: int,
    min_train: int = DEFAULT_MIN_TRAIN,
    mcs_reps: int = 500,
    seed: int = 0,
) -> dict:
    """Compare LogHAR vs CrossHAR (own+peers) on a target index via DM/MCS.

    Aligns the target and all peer series to their common date intersection,
    then runs an expanding-window backtest comparing:

    - ``LogHAR``: own-index information only (the benchmark).
    - ``CrossHAR``: own + peers' lagged daily log-RV.
    - ``RW``: random-walk sanity floor.

    Parameters
    ----------
    dataset : RealizedDataset
        Realized panel.
    target : str
        Ticker of the index to forecast.
    peers : list[str]
        Ticker(s) of the peer indices to use as additional features.
    horizon : int
        Forecast horizon in days.
    min_train : int, default :data:`~volbench.models.DEFAULT_MIN_TRAIN`
        First test origin.
    mcs_reps : int, default 500
        Bootstrap replications for the Model Confidence Set.
    seed : int, default 0
        Random seed for MCS bootstrap reproducibility.

    Returns
    -------
    dict
        Keys:

        ``target``
            Ticker that was forecast.
        ``n_obs``
            Length of the aligned target series.
        ``n_origins``
            Number of common OOS test origins.
        ``mean_qlike``
            ``dict[model_name -> mean QLIKE]``.
        ``pct_improvement``
            QLIKE improvement of CrossHAR over LogHAR in percent
            (positive = CrossHAR better).
        ``mcs_included``
            List of models in the 90% Model Confidence Set (QLIKE).
        ``dm_crosshar_vs_loghar``
            Diebold-Mariano result dict comparing CrossHAR against LogHAR
            (negative ``mean_diff`` favours CrossHAR).
        ``crosshar_beats_loghar``
            ``True`` if CrossHAR has lower mean QLIKE AND the (descriptive,
            nesting-invalid) QLIKE DM p-value < 0.10.
        ``cw_crosshar_vs_loghar``
            Clark-West (2007) nested-model test of CrossHAR vs LogHAR on the MSE
            channel (the valid significance test; positive favours CrossHAR).
        ``crosshar_improves_cw``
            ``True`` if the Clark-West one-sided p-value < 0.10 and it favours
            CrossHAR. This is the spillover significance verdict.
    """
    all_tickers = [target] + list(peers)
    _dates, rv_dict = align_panel(dataset, all_tickers)

    target_rv = rv_dict[target]
    peer_matrix = np.column_stack([rv_dict[p] for p in peers])  # (n, n_peers)

    models = [
        LogHAR(),
        CrossHAR(peer_matrix),
        RandomWalk(),
    ]

    result = run_backtest(
        target_rv,
        horizon,
        models=models,
        min_train=min_train,
        mcs_reps=mcs_reps,
        seed=seed,
        benchmark="LogHAR",
    )

    mean_qlike: dict[str, float] = {
        name: result.mean_losses["QLIKE"][name] for name in result.model_names
    }

    loghar_q = mean_qlike["LogHAR"]
    crosshar_q = mean_qlike["CrossHAR"]
    pct_improvement = 100.0 * (loghar_q - crosshar_q) / loghar_q if loghar_q > 0 else float("nan")

    mcs_included = result.mcs["QLIKE"].included

    dm_result = result.dm_vs_har["QLIKE"].get("CrossHAR", {})

    crosshar_beats = (
        crosshar_q < loghar_q
        and isinstance(dm_result.get("p_value"), float)
        and dm_result["p_value"] < 0.10
        and dm_result.get("favored", 0.0) == -1.0
    )

    # CrossHAR nests LogHAR (set the peer coefficients to zero), so the QLIKE
    # Diebold-Mariano above is NOT valid for a significance claim (Diebold 2015):
    # the test degenerates under nesting and is biased against the larger model.
    # The Clark-West (2007) MSPE test corrects the nesting bias and is the basis
    # for the "spillover is significant" verdict; ``dm_crosshar_vs_loghar`` is
    # retained only as a descriptive QLIKE effect size.
    cw_result = clark_west(
        result.realized,
        result.forecasts["LogHAR"],    # restricted (nested) forecast
        result.forecasts["CrossHAR"],  # unrestricted (nesting) forecast
        horizon=horizon,
    )
    crosshar_improves_cw = (
        isinstance(cw_result.get("p_value"), float)
        and np.isfinite(cw_result["p_value"])
        and cw_result["p_value"] < 0.10
        and cw_result.get("favors_unrestricted", 0.0) == 1.0
    )

    return {
        "target": target,
        "n_obs": int(target_rv.size),
        "n_origins": int(result.origins.size),
        "mean_qlike": mean_qlike,
        "pct_improvement": float(pct_improvement),
        "mcs_included": mcs_included,
        "dm_crosshar_vs_loghar": dm_result,
        "crosshar_beats_loghar": bool(crosshar_beats),
        "cw_crosshar_vs_loghar": cw_result,
        "crosshar_improves_cw": bool(crosshar_improves_cw),
    }
