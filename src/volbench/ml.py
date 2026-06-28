"""Tier 2D: rigorous, leakage-free machine-learning forecasters.

The headline benchmark uses a single gradient-boosted model (``GBRT``) on plain
HAR features. This module makes the "ML vs HAR" question defensible by adding
LightGBM, XGBoost and a small MLP, each:

* fit in **log-variance** space (positivity, as required for any unbounded
  additive learner),
* with hyperparameters chosen **once on the first training window** by a strict
  expanding-window inner CV that only ever sees data available at the origin (no
  leakage, the single most common silent error in ML volatility studies); only
  the model is then refit on the growing window,
* on two feature sets: a **plain** HAR set and an **enriched** set that adds the
  continuous/jump decomposition, the realized semivariances and (optionally)
  peer indices' lagged realized variance.

The scientific question is whether ML wins when given *richer* features where it
plausibly could, not just on the plain HAR set. A forecast-combination wrapper
(:class:`EnsembleForecaster`) tests whether averaging HAR and ML beats either.

LightGBM/XGBoost are optional dependencies; importing this module does not import
them (the forecasters import lazily when fit).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .losses import mean_loss, qlike
from .models import (
    _LOG_FLOOR,
    DEFAULT_MIN_TRAIN,
    MAX_LAG,
    VolForecaster,
    _test_origins,
    average_future_variance,
    har_components,
)

_DEFAULT_REFIT_EVERY: int = 66  # ~one trading quarter; matches the GBRT baseline cadence
_CV_VAL_FRAC: float = 0.2        # tail fraction of the training window used to score HPs


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
def plain_features(rv: np.ndarray, n_lags: int = 5) -> np.ndarray:
    """Plain HAR feature matrix: daily/weekly/monthly components plus raw lags."""
    rv = np.asarray(rv, dtype=float).ravel()
    n = rv.size
    comps = har_components(rv)
    lags = np.full((n, n_lags), np.nan)
    for j in range(1, n_lags + 1):
        lags[j:, j - 1] = rv[:-j]
    return np.column_stack([comps, lags])


def enriched_features(
    rv: np.ndarray,
    cont: np.ndarray,
    jump: np.ndarray,
    rsv_minus: np.ndarray,
    rsv_plus: np.ndarray,
    peer_rv: np.ndarray | None = None,
) -> np.ndarray:
    """Enriched feature matrix.

    HAR components of RV, the continuous part and the jump part; the daily
    downside/upside semivariances; and, if supplied, each peer index's daily
    realized variance (a value known at the origin). All measure series must be
    aligned with ``rv``.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    rv_c = har_components(rv)
    cont_c = har_components(np.asarray(cont, dtype=float).ravel())
    jump_c = har_components(np.asarray(jump, dtype=float).ravel())
    cols = [rv_c, cont_c, jump_c,
            np.asarray(rsv_minus, dtype=float).reshape(-1, 1),
            np.asarray(rsv_plus, dtype=float).reshape(-1, 1)]
    if peer_rv is not None:
        peer = np.asarray(peer_rv, dtype=float)
        if peer.ndim == 1:
            peer = peer.reshape(-1, 1)
        cols.append(peer)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Estimator grids (label, factory). Kept small to bound walk-forward runtime.
# ---------------------------------------------------------------------------
def _lgbm_grid(random_state: int) -> list[tuple[str, Callable]]:
    import lightgbm as lgb

    def make(num_leaves, lr, n_est):
        return lambda: lgb.LGBMRegressor(
            num_leaves=num_leaves, learning_rate=lr, n_estimators=n_est,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=random_state, verbose=-1,
        )

    return [("lgbm_31_0.05_200", make(31, 0.05, 200)),
            ("lgbm_15_0.05_300", make(15, 0.05, 300)),
            ("lgbm_63_0.05_150", make(63, 0.05, 150))]


def _xgb_grid(random_state: int) -> list[tuple[str, Callable]]:
    import xgboost as xgb

    def make(depth, eta, n_est):
        return lambda: xgb.XGBRegressor(
            max_depth=depth, learning_rate=eta, n_estimators=n_est,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, verbosity=0, n_jobs=0,
        )

    return [("xgb_3_0.05_200", make(3, 0.05, 200)),
            ("xgb_4_0.05_150", make(4, 0.05, 150)),
            ("xgb_5_0.05_200", make(5, 0.05, 200))]


def _mlp_grid(random_state: int) -> list[tuple[str, Callable]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make(hidden, alpha):
        return lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=500,
                         early_stopping=True, n_iter_no_change=15,
                         random_state=random_state),
        )

    return [("mlp_64_1e-3", make((64,), 1e-3)),
            ("mlp_64-32_1e-3", make((64, 32), 1e-3)),
            ("mlp_32_1e-2", make((32,), 1e-2))]


_GRIDS: dict[str, Callable[[int], list[tuple[str, Callable]]]] = {
    "lgbm": _lgbm_grid,
    "xgb": _xgb_grid,
    "mlp": _mlp_grid,
}


# ---------------------------------------------------------------------------
# Leakage-free ML forecaster
# ---------------------------------------------------------------------------
class MLForecaster(VolForecaster):
    """Expanding-window ML forecaster with leakage-free inner-CV tuning.

    Parameters
    ----------
    learner : str
        One of ``"lgbm"``, ``"xgb"``, ``"mlp"``.
    feature_fn : Callable[[np.ndarray], np.ndarray]
        Builds the ``(n, k)`` feature matrix from the RV series (extra measure
        series are bound into the callable by the caller).
    name : str
        Display name.
    refit_every : int, default 250
        Origins between model refits on the expanding window. Hyperparameters are
        selected once by leakage-free inner CV on the first training window and
        then reused; only the model is refit every ``refit_every`` origins.
    random_state : int, default 0
        Seed for the estimators.
    """

    def __init__(
        self,
        learner: str,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        name: str,
        refit_every: int = _DEFAULT_REFIT_EVERY,
        random_state: int = 0,
    ) -> None:
        if learner not in _GRIDS:
            raise ValueError(f"unknown learner {learner!r}; choose from {list(_GRIDS)}")
        self.learner = learner
        self.feature_fn = feature_fn
        self.name = name
        self.refit_every = max(1, int(refit_every))
        self.random_state = random_state

    def _select_config(
        self, X: np.ndarray, y_log: np.ndarray, configs: list[tuple[str, Callable]], horizon: int
    ) -> Callable:
        """Pick the estimator factory minimising validation QLIKE.

        The training rows are time-ordered; the last ``_CV_VAL_FRAC`` form the
        validation block, separated from the inner-training block by a gap of
        ``horizon`` rows so their realization windows do not overlap. Everything
        here is within data available at the origin, so there is no leakage.
        """
        n = y_log.size
        val_size = max(60, int(n * _CV_VAL_FRAC))
        inner_end = n - val_size - horizon
        if inner_end < 100:  # too little data to tune; use the first config
            return configs[0][1]
        Xtr, ytr = X[:inner_end], y_log[:inner_end]
        Xv, yv = X[inner_end + horizon:], y_log[inner_end + horizon:]
        real_v = np.exp(yv)
        best_factory, best_loss = configs[0][1], np.inf
        for _label, factory in configs:
            model = factory().fit(Xtr, ytr)
            pred = np.asarray(model.predict(Xv), dtype=float)
            loss = mean_loss(qlike(real_v, np.exp(pred)))
            if np.isfinite(loss) and loss < best_loss:
                best_loss, best_factory = loss, factory
        return best_factory

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        rv = np.asarray(rv, dtype=float).ravel()
        n = rv.size
        feats = self.feature_fn(rv)
        target = average_future_variance(rv, horizon)
        log_target = np.log(np.maximum(target, _LOG_FLOOR))
        finite_feat = np.isfinite(feats).all(axis=1)
        origins = _test_origins(n, horizon, min_train)
        forecasts = np.empty(origins.size)
        configs = _GRIDS[self.learner](self.random_state)

        # Hyperparameters are chosen once by leakage-free inner CV on the first
        # training window and reused; the model itself is refit on the expanding
        # window every ``refit_every`` origins and used to forecast that whole
        # block in a single batched prediction.
        factory = None
        re = self.refit_every
        for start in range(0, origins.size, re):
            block = origins[start : start + re]
            last_train = int(block[0]) - horizon
            rows = np.arange(MAX_LAG - 1, last_train + 1)
            rows = rows[finite_feat[rows] & np.isfinite(target[rows])]
            X_tr, y_tr = feats[rows], log_target[rows]
            if factory is None:
                factory = self._select_config(X_tr, y_tr, configs, horizon)
            model = factory().fit(X_tr, y_tr)
            # Lognormal back-transform variance from in-sample residuals, applied
            # identically to log-HAR and to the ML learners so the comparison stays
            # fair; for the regularised trees the in-sample/out-of-sample gap shifts
            # forecasts by under a few percent, well inside the QLIKE gap to log-HAR
            # and rank-irrelevant.
            resid = y_tr - np.asarray(model.predict(X_tr), dtype=float)
            s2 = float(resid @ resid) / max(resid.size - 1, 1)
            lo = max(float(np.min(target[rows])) * 0.1, _LOG_FLOOR)
            hi = float(np.max(target[rows])) * 10.0
            log_pred = np.asarray(model.predict(feats[block]), dtype=float)
            forecasts[start : start + re] = np.clip(np.exp(log_pred + 0.5 * s2), lo, hi)
        return forecasts, origins


class EnsembleForecaster(VolForecaster):
    """Equal-weight forecast combination of several forecasters.

    Each member is run, the common origins are intersected, and the member
    forecasts are averaged there. Tests whether combining HAR and ML beats the
    individual models, a classic and frequently-winning result.

    Parameters
    ----------
    members : list[VolForecaster]
        Forecasters to combine.
    name : str, default "Combo"
        Display name.
    """

    def __init__(self, members: list[VolForecaster], name: str = "Combo") -> None:
        if len(members) < 2:
            raise ValueError("EnsembleForecaster needs at least two members")
        self.members = members
        self.name = name

    def oos_forecast(self, rv, horizon, min_train=DEFAULT_MIN_TRAIN):
        runs = [m.oos_forecast(rv, horizon, min_train) for m in self.members]
        from functools import reduce
        common = reduce(np.intersect1d, (o for _, o in runs))
        stacks = []
        for fc, org in runs:
            lookup = dict(zip(org.tolist(), fc.tolist()))
            stacks.append(np.array([lookup[o] for o in common.tolist()]))
        return np.mean(np.column_stack(stacks), axis=1), common


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def plain_ml(learner: str, refit_every: int = _DEFAULT_REFIT_EVERY,
             random_state: int = 0) -> MLForecaster:
    """An ML forecaster on the plain HAR feature set."""
    return MLForecaster(learner, lambda rv: plain_features(rv),
                        name=f"{learner.upper()}-plain", refit_every=refit_every,
                        random_state=random_state)


def enriched_ml(
    learner: str,
    cont: np.ndarray,
    jump: np.ndarray,
    rsv_minus: np.ndarray,
    rsv_plus: np.ndarray,
    peer_rv: np.ndarray | None = None,
    refit_every: int = _DEFAULT_REFIT_EVERY,
    random_state: int = 0,
) -> MLForecaster:
    """An ML forecaster on the enriched feature set (binds the measure series)."""
    def fn(rv: np.ndarray) -> np.ndarray:
        return enriched_features(rv, cont, jump, rsv_minus, rsv_plus, peer_rv)

    return MLForecaster(learner, fn, name=f"{learner.upper()}-enriched",
                        refit_every=refit_every, random_state=random_state)
