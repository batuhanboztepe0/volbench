"""volbench — a reproducible out-of-sample benchmark for realized-volatility forecasting.

Public API is re-exported here for convenience; see the submodules for detail:

- :mod:`volbench.realized`    — non-parametric realized measures
- :mod:`volbench.simulate`    — intraday simulator with known IV/JV
- :mod:`volbench.models`      — forecasters with a common walk-forward interface
- :mod:`volbench.losses`      — proxy-robust losses + calibration
- :mod:`volbench.evaluation`  — Diebold-Mariano and Model Confidence Set
- :mod:`volbench.backtest`    — the expanding-window harness
- :mod:`volbench.data`        — loaders for the bundled realized panel
- :mod:`volbench.economic`    — economic-value evaluation (vol targeting, VaR, options)
- :mod:`volbench.multivariate`— cross-index (spillover) HAR
"""

from __future__ import annotations

import warnings as _warnings

# NumPy 2.0 wheels on macOS link Apple's Accelerate BLAS, whose vectorised
# matmul leaves spurious FPU error flags set even when the result is exact. This
# surfaces as "divide by zero / overflow / invalid value encountered in matmul"
# RuntimeWarnings from perfectly well-conditioned OLS fits (verified: tiny
# coefficients, finite results). Filter only that exact message so genuine
# numerical problems elsewhere still warn; any real matmul overflow would
# produce non-finite values that the finite-checks in evaluation/backtest catch.
_warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
)

from .backtest import BacktestResult, run_backtest
from .conditional_var import ewma_variance_forecast, garch_variance_forecast
from .data import (
    RealizedDataset,
    load_crypto_rv,
    load_oxford_rv,
    load_sp500_returns,
    load_vix,
)
from .deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)
from .economic import (
    black_scholes_price,
    engle_manganelli_dq,
    option_pricing_loss,
    var_backtest,
    volatility_targeting,
)
from .evaluation import MCSResult, diebold_mariano, model_confidence_set
from .losses import (
    LOSS_FUNCTIONS,
    RANKING_LOSSES,
    mean_loss,
    mincer_zarnowitz,
    mse_variance,
    mse_volatility,
    qlike,
    rmse_volatility,
)
from .ml import (
    EnsembleForecaster,
    MLForecaster,
    enriched_features,
    enriched_ml,
    plain_features,
    plain_ml,
)
from .models import (
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
    VolForecaster,
    average_future_variance,
    default_models,
    har_components,
    har_family,
)
from .multivariate import CrossHAR, align_panel, spillover_backtest
from .realized import (
    all_measures,
    bipower_variation,
    bns_jump_test,
    median_rv,
    realized_kernel_parzen,
    realized_quarticity,
    realized_semivariance,
    realized_variance,
)
from .simulate import IntradayPath, simulate_intraday_path, simulate_many_days
from .strategy import compare_books, regime_overlay, vol_target_backtest
from .vrp import variance_risk_premium, vrp_strategy

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # realized
    "realized_variance",
    "realized_semivariance",
    "bipower_variation",
    "median_rv",
    "realized_quarticity",
    "realized_kernel_parzen",
    "bns_jump_test",
    "all_measures",
    # simulate
    "IntradayPath",
    "simulate_intraday_path",
    "simulate_many_days",
    # models
    "VolForecaster",
    "RandomWalk",
    "HistoricalMean",
    "MovingAverage",
    "EWMA",
    "AR1Log",
    "HAR",
    "LogHAR",
    "HARQ",
    "HARJ",
    "HARCJ",
    "SHAR",
    "GBRT",
    "default_models",
    "har_family",
    "average_future_variance",
    "har_components",
    # losses
    "qlike",
    "mse_variance",
    "mse_volatility",
    "rmse_volatility",
    "mean_loss",
    "mincer_zarnowitz",
    "LOSS_FUNCTIONS",
    "RANKING_LOSSES",
    # evaluation
    "diebold_mariano",
    "model_confidence_set",
    "MCSResult",
    # backtest
    "run_backtest",
    "BacktestResult",
    # data
    "load_oxford_rv",
    "load_sp500_returns",
    "load_vix",
    "load_crypto_rv",
    "RealizedDataset",
    # economic
    "volatility_targeting",
    "var_backtest",
    "engle_manganelli_dq",
    "option_pricing_loss",
    "black_scholes_price",
    # multivariate / spillover
    "CrossHAR",
    "align_panel",
    "spillover_backtest",
    # conditional-variance VaR engines (risk layer)
    "ewma_variance_forecast",
    "garch_variance_forecast",
    # deflated / probabilistic Sharpe (honest edge evaluation)
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "per_period_sharpe",
    # variance risk premium (Tier 2D edge)
    "variance_risk_premium",
    "vrp_strategy",
    # vol-targeting strategy + regime overlay
    "vol_target_backtest",
    "regime_overlay",
    "compare_books",
    # machine learning (Tier 2D)
    "MLForecaster",
    "EnsembleForecaster",
    "plain_features",
    "enriched_features",
    "plain_ml",
    "enriched_ml",
]
