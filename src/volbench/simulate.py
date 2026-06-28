"""Intraday price simulator with *known* integrated and jump variation.

The realized estimators in :mod:`volbench.realized` are validated against ground
truth produced here. Each simulated day is a continuous Itô semimartingale with
stochastic volatility, optional finite-activity jumps, and optional additive
microstructure noise, and we keep the exact integrated variance (``IV``) and
jump variation (``JV``) for the path, so an estimator can be compared to what it
is supposed to recover.

Model (one trading day mapped to the unit interval, ``n_steps`` sub-intervals):

* **Spot variance** follows an exponential Ornstein-Uhlenbeck process. With
  ``x_t = log v_t``, ``dx = kappa (mu - x) dt + xi dW``. Parameters are chosen so
  the stationary mean daily variance equals ``ann_vol**2 / 252``.
* **Continuous return** over step ``i`` is ``sqrt(v_i * dt) * z_i``. The path's
  integrated variance is the Riemann sum ``IV = sum_i v_i * dt``.
* **Jumps** arrive as a compound Poisson process with intensity
  ``jump_intensity`` per day and Gaussian sizes; ``JV = sum_k J_k**2``.
* **Microstructure noise** (optional) adds iid Gaussian noise to the log price;
  the noise variance is ``noise_ratio`` times the average per-step signal
  variance, which is what makes realized variance explode at high frequency
  while the realized kernel stays consistent.

The total quadratic variation is ``QV = IV + JV``.

References
----------
- Barndorff-Nielsen & Shephard (2002, 2004); Andersen, Bollerslev, Diebold (2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS: int = 252
_DEFAULT_STEPS: int = 390  # one-minute grid over a 6.5h session


@dataclass
class IntradayPath:
    """One simulated trading day with ground-truth (co)variation.

    Attributes
    ----------
    returns : np.ndarray
        Observed intraday log-returns (includes jumps and microstructure noise).
    clean_returns : np.ndarray
        Returns without microstructure noise (continuous part plus jumps).
    iv : float
        Integrated variance of the continuous part (ground truth for IV).
    jv : float
        Jump variation, ``sum`` of squared jump sizes (ground truth for JV).
    qv : float
        Total quadratic variation, ``iv + jv``.
    spot_var : np.ndarray
        Spot-variance path ``v_i`` (per unit time) used to generate the day.
    n_jumps : int
        Number of jumps realised on the day.
    """

    returns: np.ndarray
    clean_returns: np.ndarray
    iv: float
    jv: float
    qv: float
    spot_var: np.ndarray
    n_jumps: int


def _as_rng(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    """Resolve a NumPy generator from an explicit generator or a seed."""
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def simulate_intraday_path(
    n_steps: int = _DEFAULT_STEPS,
    ann_vol: float = 0.20,
    kappa: float = 5.0,
    vol_of_vol: float = 0.8,
    jump_intensity: float = 0.0,
    jump_size_vol: float = 0.0,
    noise_ratio: float = 0.0,
    leverage: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> IntradayPath:
    """Simulate one trading day and return it with known IV / JV.

    Parameters
    ----------
    n_steps : int, default 390
        Number of intraday sub-intervals (e.g. one-minute bars).
    ann_vol : float, default 0.20
        Target annualised volatility (sets the stationary mean variance).
    kappa : float, default 5.0
        Mean-reversion speed of log-variance (per day).
    vol_of_vol : float, default 0.8
        Volatility of log-variance (``xi``).
    jump_intensity : float, default 0.0
        Expected number of price jumps on the day (Poisson).
    jump_size_vol : float, default 0.0
        Standard deviation of (Gaussian) jump sizes, in return units.
    noise_ratio : float, default 0.0
        Microstructure-noise variance as a multiple of the average per-step
        signal variance. ``0`` disables noise.
    leverage : float, default 0.0
        Correlation between the return and log-variance innovations (the
        leverage effect; negative values make downside moves raise volatility).
    rng : numpy.random.Generator, optional
        Explicit generator (takes precedence over ``seed``).
    seed : int, optional
        Seed used if ``rng`` is not given.

    Returns
    -------
    IntradayPath
        The simulated day with ground-truth variation measures.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if not -1.0 <= leverage <= 1.0:
        raise ValueError("leverage must be in [-1, 1]")
    gen = _as_rng(rng, seed)

    dt = 1.0 / n_steps
    daily_var = (ann_vol ** 2) / TRADING_DAYS

    # Exponential-OU log-variance. Stationary variance of x is xi^2 / (2 kappa);
    # set the mean so E[v] equals the target daily variance.
    xi = vol_of_vol
    stat_var_x = xi * xi / (2.0 * kappa)
    mu_x = np.log(daily_var) - 0.5 * stat_var_x

    # Correlated innovations for returns (z) and log-variance (w).
    z = gen.standard_normal(n_steps)
    w_indep = gen.standard_normal(n_steps)
    w = leverage * z + np.sqrt(max(1.0 - leverage * leverage, 0.0)) * w_indep

    # Start from a stationary draw, then evolve the OU recursion.
    x = np.empty(n_steps)
    x_prev = mu_x + np.sqrt(stat_var_x) * gen.standard_normal()
    for i in range(n_steps):
        x_prev = x_prev + kappa * (mu_x - x_prev) * dt + xi * np.sqrt(dt) * w[i]
        x[i] = x_prev
    spot_var = np.exp(x)  # per unit time

    # Continuous returns and the path's integrated variance.
    cont_returns = np.sqrt(spot_var * dt) * z
    iv = float(np.sum(spot_var * dt))

    # Compound-Poisson jumps.
    jump_returns = np.zeros(n_steps)
    n_jumps = 0
    if jump_intensity > 0.0 and jump_size_vol > 0.0:
        n_jumps = int(gen.poisson(jump_intensity))
        if n_jumps > 0:
            idx = gen.integers(0, n_steps, size=n_jumps)
            sizes = jump_size_vol * gen.standard_normal(n_jumps)
            np.add.at(jump_returns, idx, sizes)
            jv = float(np.sum(sizes ** 2))
        else:
            jv = 0.0
    else:
        jv = 0.0

    clean_returns = cont_returns + jump_returns

    # Additive iid microstructure noise on the log price -> MA(1) in returns.
    if noise_ratio > 0.0:
        omega2 = noise_ratio * daily_var / n_steps
        omega = np.sqrt(omega2)
        u = omega * gen.standard_normal(n_steps + 1)
        noise_returns = np.diff(u)
        returns = clean_returns + noise_returns
    else:
        returns = clean_returns

    return IntradayPath(
        returns=returns,
        clean_returns=clean_returns,
        iv=iv,
        jv=jv,
        qv=iv + jv,
        spot_var=spot_var,
        n_jumps=n_jumps,
    )


def simulate_many_days(
    n_days: int,
    rng: np.random.Generator | None = None,
    seed: int | None = 0,
    **kwargs,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Simulate ``n_days`` independent days for estimator validation.

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    rng : numpy.random.Generator, optional
        Explicit generator (takes precedence over ``seed``).
    seed : int, optional, default 0
        Seed used if ``rng`` is not given.
    **kwargs
        Forwarded to :func:`simulate_intraday_path`.

    Returns
    -------
    dict
        Keys: ``returns`` (list of per-day return arrays), ``clean_returns``
        (list), ``iv``, ``jv``, ``qv`` (float arrays, one entry per day), and
        ``n_jumps`` (int array).
    """
    gen = _as_rng(rng, seed)
    returns: list[np.ndarray] = []
    clean: list[np.ndarray] = []
    iv = np.empty(n_days)
    jv = np.empty(n_days)
    qv = np.empty(n_days)
    n_jumps = np.empty(n_days, dtype=int)
    for d in range(n_days):
        path = simulate_intraday_path(rng=gen, **kwargs)
        returns.append(path.returns)
        clean.append(path.clean_returns)
        iv[d] = path.iv
        jv[d] = path.jv
        qv[d] = path.qv
        n_jumps[d] = path.n_jumps
    return {
        "returns": returns,
        "clean_returns": clean,
        "iv": iv,
        "jv": jv,
        "qv": qv,
        "n_jumps": n_jumps,
    }
