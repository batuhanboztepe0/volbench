"""Volatility-forecasting models with a common out-of-sample interface.

Every model implements :meth:`VolForecaster.oos_forecast`, which performs an
*expanding-window* walk-forward over a single realized-variance series and
returns the forecast vector together with the test origins it corresponds to.
All models forecast the **same** target -- the average daily variance over the
next ``horizon`` days,

    target_t = mean(RV_{t+1}, ..., RV_{t+horizon}),

so their losses are directly comparable. Training at an origin ``t`` uses only
observations whose realization window has fully closed by ``t`` (rows ``s`` with
``s + horizon <= t``), which rules out look-ahead bias at every step -- a point
where many implementations leak future information for ``horizon > 1``.

Two families share the interface:

* *Simple* models (RandomWalk, HistoricalMean, MovingAverage, EWMA) need no
  estimation and are computed in closed form / vectorised.
* *Regression* models (AR1Log, HAR, LogHAR, HARQ, GBRT) are refit on the
  expanding window. Cheap linear models refit every step; the gradient-boosted
  model refits every ``refit_every`` steps and is reused in between.

Inputs and outputs are on the **variance** scale (realized variance), not the
volatility scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEEK_LAG: int = 5          # trading days in the HAR weekly component
MONTH_LAG: int = 22        # trading days in the HAR monthly component
MAX_LAG: int = MONTH_LAG   # longest lookback needed to form features
DEFAULT_MIN_TRAIN: int = 500   # minimum origins before the first OOS forecast
RISKMETRICS_LAMBDA: float = 0.94  # RiskMetrics decay for the EWMA model
_LOG_FLOOR: float = 1e-300  # guards log of a non-positive variance
_DEFAULT_GBRT_REFIT: int = 66  # refit cadence (trading quarter) for the tree model


# ---------------------------------------------------------------------------
# Shared feature / target construction
# ---------------------------------------------------------------------------
def average_future_variance(rv: np.ndarray, horizon: int) -> np.ndarray:
    """Direct multi-horizon target: mean variance over the next ``horizon`` days.

    Parameters
    ----------
    rv : np.ndarray
        Realized-variance series.
    horizon : int
        Forecast horizon in days.

    Returns
    -------
    np.ndarray
        Array ``target`` with ``target[t] = mean(rv[t+1 : t+1+horizon])`` and
        ``nan`` where the full window is unavailable.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    n = rv.size
    target = np.full(n, np.nan)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    csum = np.concatenate([[0.0], np.cumsum(rv)])
    for t in range(n - horizon):
        target[t] = (csum[t + 1 + horizon] - csum[t + 1]) / horizon
    return target


def har_components(rv: np.ndarray) -> np.ndarray:
    """Heterogeneous Autoregressive components (Corsi 2009).

    For each ``t`` returns the daily, weekly and monthly realized-variance
    averages computed from information available up to and including ``t``.

    Parameters
    ----------
    rv : np.ndarray
        Realized-variance series.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 3)`` with columns ``[rv_d, rv_w, rv_m]``; rows
        before ``MONTH_LAG - 1`` contain ``nan``.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    n = rv.size
    feats = np.full((n, 3), np.nan)
    csum = np.concatenate([[0.0], np.cumsum(rv)])
    for t in range(MAX_LAG - 1, n):
        rv_d = rv[t]
        rv_w = (csum[t + 1] - csum[t + 1 - WEEK_LAG]) / WEEK_LAG
        rv_m = (csum[t + 1] - csum[t + 1 - MONTH_LAG]) / MONTH_LAG
        feats[t] = (rv_d, rv_w, rv_m)
    return feats


def _test_origins(n: int, horizon: int, min_train: int) -> np.ndarray:
    """Origins at which an OOS forecast is produced.

    Parameters
    ----------
    n : int
        Length of the realized-variance series.
    horizon : int
        Forecast horizon.
    min_train : int
        First origin index (guarantees a minimum training span).

    Returns
    -------
    np.ndarray
        Integer origins ``t`` with ``min_train <= t <= n - 1 - horizon``.
    """
    last = n - horizon  # exclusive end -> last valid origin is n - horizon - 1
    if last <= min_train:
        return np.empty(0, dtype=int)
    return np.arange(min_train, last, dtype=int)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class VolForecaster(ABC):
    """Abstract base class for a volatility forecaster."""

    name: str = "base"

    @abstractmethod
    def oos_forecast(
        self, rv: np.ndarray, horizon: int, min_train: int = DEFAULT_MIN_TRAIN
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expanding-window OOS forecast for one series.

        Parameters
        ----------
        rv : np.ndarray
            Realized-variance series.
        horizon : int
            Forecast horizon in days.
        min_train : int, default :data:`DEFAULT_MIN_TRAIN`
            First test origin.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(forecast, origins)`` where ``forecast[k]`` is the predicted
            average variance made at ``origins[k]``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Simple (closed-form) models
# ---------------------------------------------------------------------------
class RandomWalk(VolForecaster):
    """Random-walk forecast: the next-period variance equals the latest RV.

    The same flat value is used for every horizon (a martingale forecast of the
    average future variance).
    """

    name = "RW"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        origins = _test_origins(rv.size, horizon, min_train)
        return rv[origins], origins


class HistoricalMean(VolForecaster):
    """Expanding-mean forecast: the average of all realized variances to date."""

    name = "HistMean"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        origins = _test_origins(rv.size, horizon, min_train)
        csum = np.cumsum(rv)
        counts = np.arange(1, rv.size + 1)
        expanding_mean = csum / counts
        return expanding_mean[origins], origins


class MovingAverage(VolForecaster):
    """Rolling-mean forecast over a fixed window.

    Parameters
    ----------
    window : int, default :data:`MONTH_LAG`
        Number of trailing days averaged.
    """

    def __init__(self, window: int = MONTH_LAG) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.name = f"MA{window}"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        origins = _test_origins(rv.size, horizon, min_train)
        csum = np.concatenate([[0.0], np.cumsum(rv)])
        w = self.window
        ma = (csum[origins + 1] - csum[origins + 1 - w]) / w
        return ma, origins


class EWMA(VolForecaster):
    """RiskMetrics-style exponentially weighted moving average on the RV series.

    The level recursion ``level_t = lam * level_{t-1} + (1 - lam) * rv_t`` is
    used as a flat forecast across the horizon, mirroring the RiskMetrics
    convention that the multi-step forecast equals the one-step forecast.

    Parameters
    ----------
    lam : float, default :data:`RISKMETRICS_LAMBDA`
        Decay parameter in (0, 1).
    """

    def __init__(self, lam: float = RISKMETRICS_LAMBDA) -> None:
        if not 0.0 < lam < 1.0:
            raise ValueError(f"lam must be in (0, 1), got {lam}")
        self.lam = lam
        self.name = "EWMA"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        n = rv.size
        level = np.empty(n)
        level[0] = rv[0]
        lam = self.lam
        for t in range(1, n):
            level[t] = lam * level[t - 1] + (1.0 - lam) * rv[t]
        origins = _test_origins(n, horizon, min_train)
        return level[origins], origins


# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------
def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares coefficients via :func:`numpy.linalg.lstsq`.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (rows = observations).
    y : np.ndarray
        Target vector.

    Returns
    -------
    np.ndarray
        Coefficient vector.
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


class _LinearHARBase(VolForecaster):
    """Shared expanding-window machinery for (log-)HAR regression models.

    Subclasses set :attr:`log_space`. In level space the model regresses the
    average future variance on a constant and the three HAR components. In log
    space it regresses the log target on the logged components and maps the
    forecast back with the lognormal correction ``exp(xb + 0.5 * s2)`` where
    ``s2`` is the in-sample residual variance.
    """

    log_space: bool = False

    def _design(self, components: np.ndarray) -> np.ndarray:
        """Assemble the regression design matrix from HAR components.

        Parameters
        ----------
        components : np.ndarray
            Array of shape ``(k, 3)`` of ``[rv_d, rv_w, rv_m]`` rows.

        Returns
        -------
        np.ndarray
            Design matrix with a leading intercept column.
        """
        if self.log_space:
            comp = np.log(np.maximum(components, _LOG_FLOOR))
        else:
            comp = components
        return np.column_stack([np.ones(comp.shape[0]), comp])

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        n = rv.size
        comps = har_components(rv)
        target = average_future_variance(rv, horizon)
        origins = _test_origins(n, horizon, min_train)
        forecasts = np.empty(origins.size)

        for k, t in enumerate(origins):
            # Training rows: features known and target window closed by t.
            last_train = t - horizon  # inclusive
            rows = np.arange(MAX_LAG - 1, last_train + 1)
            comp_train = comps[rows]
            y_train = target[rows]
            valid = np.isfinite(comp_train).all(axis=1) & np.isfinite(y_train)
            comp_train = comp_train[valid]
            y_train = y_train[valid]

            X_train = self._design(comp_train)
            y_fit = np.log(np.maximum(y_train, _LOG_FLOOR)) if self.log_space else y_train
            beta = _ols_fit(X_train, y_fit)

            x_pred = self._design(comps[t : t + 1])
            pred = float((x_pred @ beta).ravel()[0])
            if self.log_space:
                resid = y_fit - X_train @ beta
                s2 = float(resid @ resid) / max(resid.size - X_train.shape[1], 1)
                pred = float(np.exp(pred + 0.5 * s2))
            forecasts[k] = pred
        return forecasts, origins


class HAR(_LinearHARBase):
    """Heterogeneous Autoregressive model in level space (Corsi 2009)."""

    name = "HAR"
    log_space = False


class LogHAR(_LinearHARBase):
    """HAR estimated in log space with a lognormal back-transformation."""

    name = "LogHAR"
    log_space = True


class AR1Log(VolForecaster):
    """First-order autoregression on log realized variance (direct horizon).

    Regresses the log average-future-variance on a constant and the current log
    RV, then maps back with the lognormal correction. A parsimonious baseline
    that isolates the daily-persistence component of the HAR family.
    """

    name = "AR1Log"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        n = rv.size
        log_rv = np.log(np.maximum(rv, _LOG_FLOOR))
        target = average_future_variance(rv, horizon)
        origins = _test_origins(n, horizon, min_train)
        forecasts = np.empty(origins.size)

        for k, t in enumerate(origins):
            last_train = t - horizon
            rows = np.arange(0, last_train + 1)
            y_train = target[rows]
            valid = np.isfinite(y_train)
            rows = rows[valid]
            X_train = np.column_stack([np.ones(rows.size), log_rv[rows]])
            y_fit = np.log(np.maximum(target[rows], _LOG_FLOOR))
            beta = _ols_fit(X_train, y_fit)
            x_pred = np.array([1.0, log_rv[t]])
            mu = float(x_pred @ beta)
            resid = y_fit - X_train @ beta
            s2 = float(resid @ resid) / max(resid.size - 2, 1)
            forecasts[k] = float(np.exp(mu + 0.5 * s2))
        return forecasts, origins


class HARQ(VolForecaster):
    """HARQ model of Bollerslev, Patton and Quaedvlieg (2016).

    Augments level-space HAR with the interaction ``rv_d * sqrt(rq_d)``, where
    ``rq_d`` is the daily realized quarticity. The interaction lets the daily
    coefficient shrink when the daily RV is measured imprecisely (high
    quarticity). Requires a realized-quarticity series, which daily public data
    typically lack; in this package HARQ is exercised on the simulation track
    where quarticity is computable from intraday returns.

    Parameters
    ----------
    rq : np.ndarray
        Realized-quarticity series aligned with ``rv``.
    """

    name = "HARQ"

    def __init__(self, rq: np.ndarray) -> None:
        self.rq = np.asarray(rq, dtype=float).ravel()

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        if self.rq.size != rv.size:
            raise ValueError("rq and rv must have the same length")
        n = rv.size
        comps = har_components(rv)
        sqrt_rq = np.sqrt(np.maximum(self.rq, 0.0))
        target = average_future_variance(rv, horizon)
        origins = _test_origins(n, horizon, min_train)
        forecasts = np.empty(origins.size)

        def design(idx: np.ndarray) -> np.ndarray:
            c = comps[idx]
            interaction = c[:, 0] * sqrt_rq[idx]
            return np.column_stack([np.ones(idx.size), c, interaction])

        for k, t in enumerate(origins):
            last_train = t - horizon
            rows = np.arange(MAX_LAG - 1, last_train + 1)
            valid = np.isfinite(comps[rows]).all(axis=1) & np.isfinite(target[rows])
            rows = rows[valid]
            X_train = design(rows)
            beta = _ols_fit(X_train, target[rows])
            x_pred = design(np.array([t]))
            pred = float((x_pred @ beta).ravel()[0])
            # Level-space HARQ can extrapolate to a non-positive or huge value
            # (the RV*sqrt(RQ) interaction is large and noisy, especially on
            # very volatile assets); clamp to the training support so a numerical
            # artifact cannot dominate the QLIKE mean (ROADMAP invariant 2).
            lo = max(float(np.min(target[rows])) * 0.1, _LOG_FLOOR)
            hi = float(np.max(target[rows])) * 10.0
            forecasts[k] = min(max(pred, lo), hi)
        return forecasts, origins


class GBRT(VolForecaster):
    """Gradient-boosted regression trees on HAR features, in log-variance space.

    Uses scikit-learn's :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
    on the three HAR components plus a short block of raw lags. The target is
    the *log* average future variance: gradient boosting is an unbounded
    additive model, so regressing raw variance can yield non-positive forecasts
    at long horizons (and a degenerate QLIKE); modelling the log and mapping
    back with the lognormal correction guarantees positivity and stabilises the
    heavy-tailed scale, matching how :class:`LogHAR` treats the linear case.
    The model is refit every ``refit_every`` origins and reused in between to
    keep the walk-forward tractable. LightGBM is a drop-in replacement. This is
    the package's representative nonlinear / machine-learning forecaster.

    Parameters
    ----------
    refit_every : int, default :data:`_DEFAULT_GBRT_REFIT`
        Number of origins between refits.
    n_lags : int, default :data:`WEEK_LAG`
        Number of raw RV lags appended to the HAR components.
    random_state : int, default 0
        Seed passed to the estimator.
    """

    name = "GBRT"

    def __init__(
        self,
        refit_every: int = _DEFAULT_GBRT_REFIT,
        n_lags: int = WEEK_LAG,
        random_state: int = 0,
    ) -> None:
        self.refit_every = max(1, int(refit_every))
        self.n_lags = max(0, int(n_lags))
        self.random_state = random_state

    def _features(self, rv: np.ndarray) -> np.ndarray:
        """Build the feature matrix: HAR components plus raw lags.

        Parameters
        ----------
        rv : np.ndarray
            Realized-variance series.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(n, 3 + n_lags)`` with ``nan`` in the
            warm-up region.
        """
        comps = har_components(rv)
        n = rv.size
        lags = np.full((n, self.n_lags), np.nan)
        for j in range(1, self.n_lags + 1):
            lags[j:, j - 1] = rv[:-j]
        return np.column_stack([comps, lags])

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "GBRT requires scikit-learn; install it or use a linear model"
            ) from exc

        rv = np.asarray(rv, dtype=float).ravel()
        n = rv.size
        feats = self._features(rv)
        target = average_future_variance(rv, horizon)
        log_target = np.log(np.maximum(target, _LOG_FLOOR))
        origins = _test_origins(n, horizon, min_train)
        forecasts = np.empty(origins.size)

        model = None
        log_resid_var = 0.0
        for k, t in enumerate(origins):
            if k % self.refit_every == 0 or model is None:
                last_train = t - horizon
                rows = np.arange(MAX_LAG - 1, last_train + 1)
                valid = np.isfinite(feats[rows]).all(axis=1) & np.isfinite(target[rows])
                rows = rows[valid]
                model = HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=self.random_state,
                )
                model.fit(feats[rows], log_target[rows])
                in_sample = model.predict(feats[rows])
                resid = log_target[rows] - in_sample
                log_resid_var = float(resid @ resid) / max(resid.size - 1, 1)
            log_pred = float(model.predict(feats[t : t + 1])[0])
            forecasts[k] = float(np.exp(log_pred + 0.5 * log_resid_var))
        return forecasts, origins


# ---------------------------------------------------------------------------
# HAR family using real jump / semivariance measures (Tier 1C)
# ---------------------------------------------------------------------------
def _measure_components(series: np.ndarray) -> np.ndarray:
    """Daily/weekly/monthly HAR averages of an arbitrary realized measure.

    Same construction as :func:`har_components` but for any measure series (the
    continuous part, the jump variation, a semivariance, ...).
    """
    return har_components(series)


def _walk_forward_har(
    rv: np.ndarray,
    log_feats: list[np.ndarray],
    level_feats: list[np.ndarray],
    horizon: int,
    min_train: int,
    log_target: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window OLS walk-forward for a measure-augmented HAR model.

    Regressors are split into two groups so that positivity is respected when a
    component can be zero (jumps, see ``ROADMAP.md`` invariant 2):

    * ``log_feats`` — strictly-positive components entered as ``log(component)``,
    * ``level_feats`` — components kept in levels (jump variation can be exactly
      zero, so it must not be logged).

    When ``log_target`` is ``True`` the model regresses ``log(target)`` and maps
    back with the lognormal correction ``exp(mu + 0.5 * resid_var)``, which keeps
    the variance forecast strictly positive regardless of the level regressors.

    Training at origin ``t`` uses only rows whose realization window has closed
    by ``t`` (``s + horizon <= t``), so there is no look-ahead.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    n = rv.size
    target = average_future_variance(rv, horizon)
    origins = _test_origins(n, horizon, min_train)
    forecasts = np.empty(origins.size)

    log_mat = np.column_stack(log_feats) if log_feats else np.empty((n, 0))
    lvl_mat = np.column_stack(level_feats) if level_feats else np.empty((n, 0))
    finite = (
        np.isfinite(log_mat).all(axis=1)
        & np.isfinite(lvl_mat).all(axis=1)
        & (log_mat > 0.0).all(axis=1)
    )

    def design(idx: np.ndarray) -> np.ndarray:
        cols = [np.ones(idx.size)]
        if log_mat.shape[1]:
            cols.append(np.log(np.maximum(log_mat[idx], _LOG_FLOOR)))
        if lvl_mat.shape[1]:
            cols.append(lvl_mat[idx])
        return np.column_stack(cols)

    for k, t in enumerate(origins):
        last_train = t - horizon
        rows = np.arange(MAX_LAG - 1, last_train + 1)
        valid = finite[rows] & np.isfinite(target[rows])
        rows = rows[valid]
        y_level = target[rows]
        X = design(rows)
        y = np.log(np.maximum(y_level, _LOG_FLOOR)) if log_target else y_level
        beta = _ols_fit(X, y)
        x_pred = design(np.array([t]))
        pred = float((x_pred @ beta).ravel()[0])
        if log_target:
            resid = y - X @ beta
            s2 = float(resid @ resid) / max(resid.size - X.shape[1], 1)
            pred = float(np.exp(pred + 0.5 * s2))
        # Sanity clamp to the training support. Measure-augmented HAR models can
        # extrapolate to a non-positive value (level form) or to a value many
        # orders of magnitude outside the data (when a jump/semivariance
        # regressor is near zero at the origin); such a forecast is a numerical
        # artifact, not a prediction, and would otherwise dominate the QLIKE
        # mean. The log baselines never trigger this; normal forecasts are inside
        # the band and pass through unchanged.
        lo = max(float(np.min(y_level)) * 0.1, _LOG_FLOOR)
        hi = float(np.max(y_level)) * 10.0
        forecasts[k] = min(max(pred, lo), hi)
    return forecasts, origins


class HARJ(VolForecaster):
    """HAR augmented with the daily jump variation (Andersen-Bollerslev-Diebold).

    Adds ``J_d = max(RV_d - BV_d, 0)`` to the level-space HAR regression, testing
    whether separating the jump component from total RV improves forecasts.

    Parameters
    ----------
    jump : np.ndarray
        Daily jump-variation series aligned with ``rv``.
    log : bool, default False
        If ``True``, model log-variance (jump kept in levels for positivity).
    """

    def __init__(self, jump: np.ndarray, log: bool = False) -> None:
        self.jump = np.asarray(jump, dtype=float).ravel()
        self.log = log
        self.name = "LogHAR-J" if log else "HAR-J"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        comp = har_components(rv)
        jd = _measure_components(self.jump)[:, 0]
        if self.log:
            return _walk_forward_har(
                rv, [comp[:, 0], comp[:, 1], comp[:, 2]], [jd], horizon, min_train, True
            )
        return _walk_forward_har(
            rv, [], [comp[:, 0], comp[:, 1], comp[:, 2], jd], horizon, min_train, False
        )


class HARCJ(VolForecaster):
    """HAR with a continuous/jump split (Andersen-Bollerslev-Diebold 2007).

    Replaces total RV with its continuous part (bipower variation) and its jump
    part, each entered at daily/weekly/monthly frequencies.

    Parameters
    ----------
    cont : np.ndarray
        Continuous-variation series (e.g. bipower variation), aligned with ``rv``.
    jump : np.ndarray
        Jump-variation series aligned with ``rv``.
    log : bool, default False
        If ``True``, model log-variance (jump components kept in levels).
    """

    def __init__(self, cont: np.ndarray, jump: np.ndarray, log: bool = False) -> None:
        self.cont = np.asarray(cont, dtype=float).ravel()
        self.jump = np.asarray(jump, dtype=float).ravel()
        self.log = log
        self.name = "LogHAR-CJ" if log else "HAR-CJ"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        cc = _measure_components(self.cont)
        jc = _measure_components(self.jump)
        cont_feats = [cc[:, 0], cc[:, 1], cc[:, 2]]
        jump_feats = [jc[:, 0], jc[:, 1], jc[:, 2]]
        if self.log:
            return _walk_forward_har(rv, cont_feats, jump_feats, horizon, min_train, True)
        return _walk_forward_har(rv, [], cont_feats + jump_feats, horizon, min_train, False)


class SHAR(VolForecaster):
    """Semivariance HAR (Patton & Sheppard 2015).

    Splits the daily component into downside and upside realized semivariances
    (``RSV_minus``, ``RSV_plus``) while keeping weekly and monthly total RV. The
    leverage effect makes downside variation the more informative predictor.

    Parameters
    ----------
    rsv_minus, rsv_plus : np.ndarray
        Downside and upside semivariance series aligned with ``rv``.
    log : bool, default False
        If ``True``, model log-variance (semivariances are positive, so they are
        logged too).
    """

    def __init__(self, rsv_minus: np.ndarray, rsv_plus: np.ndarray, log: bool = False) -> None:
        self.rsv_minus = np.asarray(rsv_minus, dtype=float).ravel()
        self.rsv_plus = np.asarray(rsv_plus, dtype=float).ravel()
        self.log = log
        self.name = "LogSHAR" if log else "SHAR"

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        comp = har_components(rv)
        rsvm_d = _measure_components(self.rsv_minus)[:, 0]
        rsvp_d = _measure_components(self.rsv_plus)[:, 0]
        feats = [rsvm_d, rsvp_d, comp[:, 1], comp[:, 2]]  # RSV-_d, RSV+_d, RV_w, RV_m
        if self.log:
            return _walk_forward_har(rv, feats, [], horizon, min_train, True)
        return _walk_forward_har(rv, [], feats, horizon, min_train, False)


def har_family(
    cont: np.ndarray,
    jump: np.ndarray,
    rsv_minus: np.ndarray,
    rsv_plus: np.ndarray,
) -> list[VolForecaster]:
    """Build the measure-based HAR family for a within-family comparison.

    Includes the level and log baselines plus HAR-J, HAR-CJ and SHAR (each in
    level and log form), all driven by the supplied real realized measures.

    Parameters
    ----------
    cont, jump, rsv_minus, rsv_plus : np.ndarray
        Continuous, jump, downside- and upside-semivariance series aligned with
        the realized-variance series the models will be run on.

    Returns
    -------
    list[VolForecaster]
    """
    return [
        HAR(),
        LogHAR(),
        HARJ(jump),
        HARJ(jump, log=True),
        HARCJ(cont, jump),
        HARCJ(cont, jump, log=True),
        SHAR(rsv_minus, rsv_plus),
        SHAR(rsv_minus, rsv_plus, log=True),
    ]


# ---------------------------------------------------------------------------
# Default model suite for the realized-variance benchmark
# ---------------------------------------------------------------------------
def default_models() -> list[VolForecaster]:
    """Instantiate the standard model suite used in the RV benchmark.

    Returns
    -------
    list[VolForecaster]
        Baselines and competitors, excluding HARQ (no daily quarticity in the
        public data set).
    """
    return [
        RandomWalk(),
        HistoricalMean(),
        MovingAverage(MONTH_LAG),
        EWMA(),
        AR1Log(),
        HAR(),
        LogHAR(),
        GBRT(),
    ]
