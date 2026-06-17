# volbench

[![CI](https://github.com/batuhanboztepe0/volbench/actions/workflows/ci.yml/badge.svg)](https://github.com/batuhanboztepe0/volbench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-157%2B-brightgreen)

**A reproducible out-of-sample benchmark for realized-volatility forecasting.**

Most volatility tutorials stop at fitting a GARCH or an EWMA and eyeballing the
fit. This repository asks the question that actually matters for deployment:

> Out of sample, which volatility-forecasting models genuinely win — and are
> the differences statistically significant, or just noise?

It answers that with a rigorous walk-forward design, loss functions that are
robust to the noise in a realized-variance proxy (Patton, 2011), and formal
model-comparison tests: pairwise **Diebold-Mariano** and the **Model Confidence
Set** (Hansen, Lunde & Nason, 2011). The realized estimators are
microstructure-grade and validated against simulated ground truth, then applied
to real high-frequency data.

The benchmark runs on **real 5-minute realized data for eight international
equity indices** from the Oxford-Man Institute Realized Library (2000–2022). The
data is not redistributed here (its licence is unclear); a single command —
`python scripts/build_realized.py` — fetches it from the Internet Archive (the
source was discontinued in 2022), after which everything is reproducible offline
from a fixed seed. The data is dated, so it also drives a jump/semivariance HAR
family, a cross-index spillover study, an economic-value layer, and a
calm-vs-crisis regime analysis.

---

## Headline result

Out-of-sample QLIKE, eight indices, expanding-window walk-forward (≈5,000 test
origins per index). Lower is better; **MCS** is the number of indices (out of 8)
where the model survives in the 90% Model Confidence Set; **beats HAR** counts
indices where the model is significantly better than the level-HAR benchmark
(Diebold-Mariano, 5%).

### One-day horizon (h = 1)

| Model       | Avg QLIKE | Avg rank | MCS (of 8) | Beats HAR |
|-------------|----------:|---------:|-----------:|----------:|
| **log-HAR** | **0.187** | **1.12** | **8**      | 6         |
| GBRT (log)  | 0.196     | 2.25     | 1          | 2         |
| HAR         | 0.198     | 2.62     | 2          | —         |
| AR(1)-log   | 0.239     | 4.50     | 0          | 0         |
| EWMA        | 0.244     | 4.88     | 0          | 0         |
| MA(22)      | 0.262     | 6.00     | 0          | 0         |
| Random walk | 0.468     | 6.75     | 0          | 0         |
| Hist. mean  | 0.651     | 7.88     | 0          | 0         |

The same ordering holds at one-week (h = 5) and one-month (h = 22) horizons:
**log-HAR is in the Model Confidence Set for all 8 indices at every horizon**
and is the single best model in the large majority of index-horizon cells. At
h = 5 log-HAR is significantly better than level-HAR on **all 8** indices; the
gradient-boosted model is the consistent runner-up.

### What this says

- **A correctly specified log-space HAR is hard to beat.** Modelling
  log-variance (which respects positivity and the heavy right tail of variance)
  is worth more than model complexity.
- **Machine learning does not win here.** A gradient-boosted model on HAR
  features is competitive but does not displace log-HAR out of sample on daily
  data without microstructure-grade features. This is reported honestly rather
  than tuned away.
- **The naive baselines lose decisively.** The historical mean and the random
  walk are significantly worse than HAR on every index; the simple moving
  average and EWMA are dominated at all horizons.

---

## Beyond the headline

Because the bundled data is dated and carries the full panel of realized
measures, the benchmark goes past "HAR vs everything":

- **HAR family (which variant wins).** Using real bipower variation, jump
  variation and realized semivariances, we compare HAR-J, HAR-CJ
  (continuous/jump split) and SHAR (semivariance HAR) in level and log form.
  **The log variants dominate the level variants, and the semivariance HAR in
  log space (LogSHAR) and the log continuous/jump HAR (LogHAR-CJ) edge plain
  log-HAR** — the downside-semivariance leverage effect carries real predictive
  content. `scripts/run_har_family.py`.
- **Cross-index spillover.** Adding the other seven indices' lagged realized
  variance to a target index's HAR (`CrossHAR`) **significantly improves the
  forecast for all 8 indices** at h = 1 and h = 5 (Diebold-Mariano p < 0.01 in
  most cells; QLIKE improvements of ≈2–8%, largest for .FTSE and .STOXX50E).
  Volatility spillover is real and exploitable out of sample.
  `scripts/run_multivariate.py`.
- **Rigorous ML (does ML win on richer features?).** LightGBM, XGBoost and an
  MLP, each fit in log-variance space with **leakage-free expanding-window
  hyperparameter tuning**, on a plain HAR feature set and an enriched one
  (continuous/jump split + realized semivariances). Even with a fair quarterly
  refit cadence and the richer features, **no ML model displaces log-HAR**, and a
  log-HAR + ML combination does not beat log-HAR alone — the cleanest, most
  defensible form of the "structure beats flexibility" result. `scripts/run_ml.py`.
- **Economic value & risk.** A volatility-targeting strategy, VaR exceedance
  backtests (Kupiec / Christoffersen / Engle–Manganelli DQ), and a Black–Scholes
  option-pricing loss. The statistically-best model is *not* a clean economic
  winner: log-HAR delivers the most accurate option prices, but a simple EWMA
  edges it on vol-targeted Sharpe. Normal VaR under-covers 5% (fat tails), and a
  **Student-t / filtered-historical-simulation VaR fixes coverage** (FHS hits a
  0.050 exceedance rate, DQ p = 0.85). `scripts/run_economic.py`.
- **The edge — variance risk premium.** A good RV forecast monetises through the
  variance risk premium: on the S&P 500, implied vol (VIX) averages 21.5% vs
  16.3% realized — the premium is positive **92% of days**. Selling variance
  earns a Sharpe of 1.45; **timing it with the log-HAR forecast lifts the Sharpe
  to 1.60 and cuts max drawdown by ~65%** (you scale down when your forecast says
  implied is only fairly priced). `scripts/run_vrp.py`.
- **The edge — volatility targeting.** Scaling exposure by `target_vol /
  forecast_vol` (net of costs) holds realized vol near target and roughly
  **halves max drawdown vs buy-and-hold** (−0.40 vs −0.62 across 8 indices); it
  improves Sharpe on US indices, and a jump/regime overlay trims drawdown
  further. A risk-control product, reported honestly — vol targeting is not free
  alpha. `scripts/run_strategy.py`.
- **Regime analysis.** Splitting the 2000–2022 sample into calm vs turbulent
  states and into the GFC and COVID crisis windows, then re-running the MCS in
  each. Log-HAR stays rank-1 with MCS 8/8 in calm, turbulent and GFC regimes;
  the gap to the naive baselines *widens* sharply in crises (the historical mean
  is ≈3.5× worse than log-HAR overall but ≈12× worse during the GFC). The 93-day
  COVID window is too short for the MCS to separate models. `scripts/run_regime.py`.

---

## Estimators are validated, not assumed

The realized estimators in `volbench.realized` are validated on simulated
intraday paths with **known** integrated variance and jump variation
(`scripts/validate_estimators.py`, 4,000 simulated days):

| Check                                   | Result | Target |
|-----------------------------------------|-------:|-------:|
| Realized variance / quadratic variation | 0.999  | 1.0    |
| Bipower variation / integrated variance | 1.042  | 1.0    |
| Median RV / integrated variance         | 1.002  | 1.0    |
| (RV − bipower) / jump variation         | 0.923  | ~1.0   |
| Realized kernel / QV (clean)            | 1.003  | 1.0    |
| **Realized variance / QV (with noise)** | **3.00** (inflated) | — |
| **Realized kernel / QV (with noise)**   | **1.008** (robust)  | 1.0 |
| Jump-test false-positive rate @ 5%      | 0.052  | 0.05   |
| Jump-test detection rate (injected)     | 0.938  | high   |

The microstructure point is the classic **volatility signature plot**
(`results/figures/signature_plot.png`): under additive noise, realized variance
explodes as the sampling frequency rises, while the realized kernel stays on the
true quadratic variation.

---

## Two tracks, kept separate on purpose

- **Track 1 — realized-variance benchmark** (the main result above). Models
  consume a daily realized-variance series and forecast future realized
  variance. Proxy: 5-minute realized variance (low noise).
- **Track 2 — return-based GARCH** (`scripts/run_garch.py`), on real S&P 500
  daily returns (close-to-close from the same library, 2000–2022). Without an RV
  proxy here, one-step variance forecasts are scored against the squared daily
  return. Result ordering is as expected — GJR-GARCH and plain GARCH edge
  RiskMetrics, all far ahead of a constant variance.

These two tracks are **not** directly comparable: different series and —
critically — different proxy quality (a 5-minute RV proxy is far less noisy than
a squared daily return, which is why Track 2's QLIKE levels are an order of
magnitude higher, ≈1.4–2.3 vs ≈0.2). Mixing them would be the exact
apples-to-oranges error this project is built to avoid.

---

## Repository layout

```
volbench/
├── src/volbench/
│   ├── realized.py      # realized estimators: RV, semivariance, bipower,
│   │                    #   medRV, quarticity, realized kernel, BNS jump test
│   ├── simulate.py      # intraday simulator (exp-OU log-variance + jumps +
│   │                    #   optional microstructure noise) with known IV/JV
│   ├── models.py        # forecasters: RW, HistMean, MA, EWMA, AR(1)-log,
│   │                    #   HAR, log-HAR, HARQ, GBRT, + HAR-J/HAR-CJ/SHAR family
│   ├── losses.py        # QLIKE & MSE (Patton-robust), Mincer-Zarnowitz
│   ├── evaluation.py    # Diebold-Mariano (+HLN), Model Confidence Set
│   ├── backtest.py      # expanding-window harness tying it together
│   ├── economic.py      # vol targeting, VaR (normal/t/FHS + Kupiec/Christoffersen/DQ)
│   ├── multivariate.py  # cross-index (spillover) HAR
│   ├── ml.py            # leakage-free LightGBM/XGBoost/MLP + forecast combination
│   ├── vrp.py           # variance risk premium signal + short-variance timing
│   ├── strategy.py      # vol-targeting backtest (with costs) + jump/regime overlay
│   └── data.py          # loaders (Oxford-Man RV panel; SP500 returns; VIX)
├── scripts/             # run_{benchmark,garch,har_family,multivariate,ml,economic,
│   │                    #   vrp,strategy,regime}, validate_estimators, make_figures,
│   │                    #   build_realized, build_vix
├── tests/               # pytest suite (179 tests)
├── data/                # VIX (committed) + provenance; the RV CSV is fetched, not committed
├── results/             # tables, figures, JSON summaries (the deliverable)
├── docs/                # write-up: "why log-HAR is hard to beat"
└── report/              # LaTeX research report
```

## Methodology in brief

- **Target.** Direct multi-horizon: forecast the *average* daily variance over
  the next `h` days, so every model is scored on an identical, comparable target.
- **No look-ahead.** At each origin `t`, regression models train only on
  observations whose realization window closed by `t` (rows `s` with
  `s + h ≤ t`). This is enforced and unit-tested for every model — a common
  silent bug for `h > 1`.
- **Robust losses.** QLIKE and MSE-on-variance are consistent under a noisy
  variance proxy (Patton, 2011); RMSE-on-volatility is reported for reference
  but never used to rank.
- **Significance.** Diebold-Mariano with a Newey-West HAC variance and the
  Harvey-Leybourne-Newbold small-sample correction; the Model Confidence Set
  with a moving-block bootstrap (2,000 replications).

## Quickstart

```bash
pip install -e ".[dev]"        # package + pytest/ruff/mypy
# or: pip install -r requirements.lock   # pinned versions

python scripts/build_realized.py         # STEP 0: fetch the RV data (one-time, ~6 MB,
                                         #   from the Internet Archive); VIX is bundled

make reproduce                 # the full pipeline end to end (or run individually):
python scripts/validate_estimators.py    # results/validation.json
python scripts/run_benchmark.py          # results/summary.json + tables  (Track 1)
python scripts/run_garch.py              # results/garch.json             (Track 2)
python scripts/run_har_family.py         # results/har_family.json
python scripts/run_multivariate.py       # results/multivariate.json
python scripts/run_ml.py                 # results/ml.json  (LightGBM/XGBoost/MLP)
python scripts/run_economic.py           # results/economic.json
python scripts/run_vrp.py                # results/vrp.json  (variance risk premium)
python scripts/run_strategy.py           # results/strategy.json  (vol targeting)
python scripts/run_regime.py             # results/regime.json
python scripts/make_figures.py           # results/figures/*.png
pytest -q                                # 179 tests
```

Minimal programmatic use:

```python
from volbench.data import load_oxford_rv
from volbench.backtest import run_backtest

rv = load_oxford_rv().series(".SPX")          # daily realized variance
res = run_backtest(rv, horizon=1)             # walk-forward all models
print(res.mcs["QLIKE"].included)              # models in the 90% MCS
```

Or the CLI: `volbench run --ticker .SPX --horizon 1`.

## Caveats and honest limitations

- The Oxford-Man library ships realized *measures* but not raw intraday returns,
  and it has no realized-quarticity column, so the jump/semivariance estimators
  and the BNS jump test are validated on simulation, and `HARQ` (which needs
  realized quarticity) runs on the simulation track only.
- The realized kernel is mildly inefficient on near-noiseless data (it equals RV
  in the noise-free limit); it matters under noise, which is exactly where it is
  shown to help. The core benchmark uses the daily RV series directly.
- Track 2's GARCH-on-returns is a deliberately separate reference, not a
  competitor to Track 1's HAR-on-RV, for the proxy and series reasons above.
- Normal VaR in the economic layer under-covers because daily index returns are
  fat-tailed; this is reported as a finding, not hidden.

## References

- Andersen, Bollerslev, Diebold & Labys (2001); Corsi (2009), HAR-RV.
- Andersen, Bollerslev & Diebold (2007), HAR-CJ continuous/jump split.
- Barndorff-Nielsen & Shephard (2004, 2006), bipower variation and jump tests.
- Barndorff-Nielsen, Hansen, Lunde & Shephard (2008), realized kernels.
- Patton (2011), volatility forecast evaluation with imperfect proxies.
- Patton & Sheppard (2015), good and bad volatility / semivariance HAR.
- Hansen, Lunde & Nason (2011), the Model Confidence Set.
- Bollerslev, Patton & Quaedvlieg (2016), HARQ.
- Heber, Lunde, Shephard & Sheppard (2009), Oxford-Man Realized Library.

## License

MIT — see [LICENSE](LICENSE). Bundled data is a redistributed subset of the
Oxford-Man Institute Realized Library for reproducibility; see
[`data/README.md`](data/README.md) for provenance and credit.
