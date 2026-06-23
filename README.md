# volbench

[![CI](https://github.com/batuhanboztepe0/volbench/actions/workflows/ci.yml/badge.svg)](https://github.com/batuhanboztepe0/volbench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-228-brightgreen)

**A reproducible out-of-sample benchmark for realized-volatility forecasting.**

Out of sample, a correctly specified **log-HAR is hard to beat**. It stays in the
90% Model Confidence Set across eight equity indices, crypto, and VOLARE futures and
FX, and machine learning does not displace it. That headline is a textbook result,
reproduced carefully. The model-comparison tests (Diebold-Mariano, Model Confidence
Set) are built from scratch and match reference libraries to machine precision,
look-ahead is unit-tested per model, and a pre-specified cross-asset hypothesis is
reported with its honest, largely negative outcome.

![Where the HAR family stays in the 90% Model Confidence Set, by asset class and horizon: green = a HAR-family model dominates; the lone red cell is gold futures at the monthly horizon](results/figures/transfer_matrix.png)

> New here? The full leaderboard is below; **[Quickstart](#quickstart)** runs the
> whole pipeline end to end, and **[Methodology in brief](#methodology-in-brief)**
> covers the rules that keep it honest.

Most volatility tutorials stop at fitting a GARCH or an EWMA and eyeballing the
fit. This repository asks the question that actually matters for deployment:

> Out of sample, which volatility-forecasting models genuinely win, and are
> the differences statistically significant, or just noise?

It answers that with a rigorous walk-forward design, loss functions that are
robust to the noise in a realized-variance proxy (Patton, 2011), and formal
model-comparison tests: pairwise **Diebold-Mariano** and the **Model Confidence
Set** (Hansen, Lunde & Nason, 2011). The realized estimators are
microstructure-grade and validated against simulated ground truth, then applied
to real high-frequency data.

The benchmark runs on **real 5-minute realized data for eight international
equity indices** from the Oxford-Man Institute Realized Library (2000–2022). The
data is not redistributed here (its licence is unclear); a single command
(`python scripts/build_realized.py`) fetches it from the Internet Archive (the
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

| Model        | Avg QLIKE | Avg rank | MCS (of 8) | Beats HAR |
|--------------|----------:|---------:|-----------:|----------:|
| **log-HAR**  | **0.187** | **1.12** | **8**      | 6         |
| GBRT (log)   | 0.196     | 2.75     | 1          | 2         |
| ARFIMA (log) | 0.197     | 2.75     | 0          | 4         |
| HAR          | 0.198     | 3.38     | 2          | n/a       |
| AR(1)-log    | 0.239     | 5.50     | 0          | 0         |
| EWMA         | 0.244     | 5.88     | 0          | 0         |
| MA(22)       | 0.262     | 7.00     | 0          | 0         |
| Random walk  | 0.468     | 7.75     | 0          | 0         |
| Hist. mean   | 0.651     | 8.88     | 0          | 0         |

The same ordering holds at one-week (h = 5) and one-month (h = 22) horizons:
**log-HAR is in the Model Confidence Set for all 8 indices at every horizon**
and is the single best model in the large majority of index-horizon cells. At
h = 5 log-HAR is significantly better than level-HAR on **all 8** indices. The
gradient-boosted model and the long-memory **ARFIMA** trade places as the closest
non-HAR competitors (GBRT runner-up at h = 5, ARFIMA at h = 22). Neither
displaces log-HAR at any horizon.

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

## Cross-asset generalisation (internal pre-specified protocol): where does log-HAR stop winning?

The equity headline above is a textbook result, rigorously reproduced. The
distinctive question is the next one: **does it generalise, and where does it
break?** It is answered under an *internal, pre-specified* protocol
(`docs/PREREGISTRATION.md`, kept local to the maintainer's tree: author-attested
decision rules fixed in advance as a discipline, **not** externally registered,
timestamped, or committed to this public repo) so the result cannot be quietly
reframed after the fact. A single falsifiable hypothesis
predicts the HAR family *dominates* mature, calendar-synchronised markets and
*degrades* under reversed leverage (crypto), 24/7 calendars, microstructure noise,
or event-driven breaks (commodities, rates). The primary deliverable is the **Q5
transfer matrix**, Liu–Patton–Sheppard's "does anything beat 5-minute RV?"
applied to *models*: where a HAR-family model stays in the 90% Model Confidence Set
(QLIKE), across asset classes and horizons.

![Q5 cross-asset transfer matrix: where the HAR family stays in the 90% MCS, by asset class and horizon](results/figures/transfer_matrix.png)

**What the matrix shows (robust replication, narrow cracks, reported honestly):**

- **log-HAR is robust at the *level*.** A HAR-family model is in the
  90% MCS and single-best across equities (8/8), surviving major crypto (4/4), the
  survivorship-corrected 22-coin universe (18–19/22), US Treasury futures (FV/TY),
  37/39 futures cells, and **38/39 FX cells** (7 majors + 6 EM/secondary pairs).
  These counts are unchanged at the stricter **α = 0.25** MCS (a smaller,
  harder-to-enter set) with no per-instrument verdict flips, so the dominance is
  not an artifact of the 90% level (`results/tables/transfer_matrix_alpha_sensitivity.csv`).
- **The *primary* pre-specified prediction was FALSIFIED.** The headline bet was
  that **rates futures (FV/TY)** would break HAR around the FOMC/auction calendar.
  They did not. HAR dominates both at every horizon. This is logged as a falsified
  prediction in the internal protocol's §9 amendment log, not reframed.
- **The only genuine degrade is gold (GC) @ h = 22** (the single red cell above),
  where the MCS collapses to `{EWMA}` (EWMA beats the best HAR variant by ~25%). It
  is adversarially verified as robust (stable across 7 seeds and B ∈ {2k…20k}) **but**
  isolated (no metals gradient) and **partly mechanical** (long-horizon
  estimation-risk immunity of a parameter-light smoother), at a *secondary* horizon.
  It does **not** satisfy the hypothesis's predicted-mechanism clause.
- **Equity-tuned refinements do not transfer to log-HAR.** **HARQ** never DM-beats
  log-HAR outside equities (0/22 crypto, 0/13 futures, 0/13 FX). It *does* beat
  *plain* (un-logged) HAR in a minority of cells (≈2/13 futures, 2/13 FX, 5/22
  crypto at h = 1), so the quarticity correction carries some value off-equity, but
  the log transform already captures it; against log-HAR the refinement adds
  nothing. **LogSHAR's** downside-semivariance edge vanishes in crypto (0/22 at
  h = 1) but *partly* carries to FX (4–7/13). Only the *refinements* fail, not
  log-HAR itself.

**Honest verdict.** The confirmatory outcome is **mixed/partial, leaning
"replication at scale"**, a failure mode the protocol named in advance: one
unpredicted, secondary, partly-mechanical degrade plus a falsified primary
prediction, not the clean cross-asset *discovery* the study hoped for. The
contribution is **breadth + rigor + honest negatives**, not a new model.
`scripts/run_volare_futures.py`, `run_volare_fx.py`, `run_crypto_expanded.py`,
`build_transfer_matrix.py`.

**Risk layer: VaR (siloed, an honest open problem).** A direct-quantile **CAViaR**
(Engle–Manganelli) layer with a same-window GARCH/GJR-GARCH/EWMA comparison shows
normal VaR under-covers (~9.1% at nominal 5%), a Student-t tail does not help, and
FHS improves coverage (~6.5%) but fails the dynamic-quantile (DQ) test. On a common
window **GJR-GARCH is the best VaR engine** (DQ pass 3/8); CAViaR-AS matches coverage
but not DQ (1/8). The **leverage channel**, not the modelling paradigm
(direct-quantile vs variance), is what matters. Full DQ adequacy stays unsolved.
`scripts/run_caviar.py`.

---

## Beyond the headline

Because the bundled data is dated and carries the full panel of realized
measures, the benchmark goes past "HAR vs everything":

- **HAR family (which variant wins).** Using real bipower variation, jump
  variation and realized semivariances, we compare HAR-J, HAR-CJ
  (continuous/jump split) and SHAR (semivariance HAR) in level and log form.
  **The log variants dominate the level variants, and the semivariance HAR in
  log space (LogSHAR) and the log continuous/jump HAR (LogHAR-CJ) edge plain
  log-HAR**. The downside-semivariance leverage effect carries real predictive
  content. `scripts/run_har_family.py`.
- **Cross-index spillover.** Adding the other seven indices' lagged realized
  variance to a target index's HAR (`CrossHAR`) lowers QLIKE for all 8 indices
  (≈2–8% at h = 1, largest for .FTSE and .STOXX50E). Because `CrossHAR` *nests*
  `LogHAR`, the standard Diebold-Mariano test is not valid for this comparison
  (Diebold 2015), so significance is judged by the **Clark-West (2007)
  nested-model test** on the MSE channel: `CrossHAR` significantly improves on
  `LogHAR` for **7 of 8 indices at h = 1** (all but .SPX) and 4 of 8 at h = 5.
  Volatility spillover is real and largely exploitable out of sample, strongest
  at the daily horizon. `scripts/run_multivariate.py`.
- **Rigorous ML (does ML win on richer features?).** LightGBM, XGBoost and an
  MLP, each fit in log-variance space with **leakage-free expanding-window
  hyperparameter tuning**, on a plain HAR feature set and an enriched one
  (continuous/jump split + realized semivariances). Even with a fair quarterly
  refit cadence and the richer features, **no ML model displaces log-HAR**, and a
  log-HAR + ML combination does not beat log-HAR alone. This is the cleanest, most
  defensible form of the "structure beats flexibility" result. `scripts/run_ml.py`.
- **Economic value & risk.** A volatility-targeting strategy, VaR exceedance
  backtests (Kupiec / Christoffersen / Engle–Manganelli DQ), and a Black–Scholes
  option-pricing loss. The statistically-best model is *not* a clean economic
  winner: log-HAR delivers the most accurate option prices, but a simple EWMA
  edges it on vol-targeted Sharpe. On real index data normal VaR **under-covers**
  the 5% tail (log-HAR: ~9.1% exceedances, averaged over 8 indices); a **Student-t
  tail does not help** (~9.1%, because at the 5% level the unit-variance t quantile is
  *less* extreme than the normal), while **FHS substantially improves
  unconditional coverage** (~6.5%, calibrated out-of-sample on a warm-up block).
  None fully passes the Engle–Manganelli dynamic-quantile test on real data, so
  the residual miss is in the conditional dynamics, not just the tail shape.
  `scripts/run_economic.py`.
- **The edge: variance risk premium.** A good RV forecast monetises through the
  variance risk premium: on the S&P 500, implied vol (VIX) averages 21.5% vs
  16.3% realized. The premium is positive **92% of days**. Selling variance
  earns a Sharpe of 1.45; **timing it with the log-HAR forecast lifts the Sharpe
  to 1.60 and cuts max drawdown by ~65%** (you scale down when your forecast says
  implied is only fairly priced). These Sharpes are gross of transaction costs and
  computed on overlapping 22-day variance-swap payoffs (so the absolute level is
  inflated vs a non-overlapping annualisation; read the *lift* over the naive book
  and the drawdown cut, both of which survive realistic costs). On the honest,
  **non-overlapping** payoffs the per-swap Sharpe is ~0.38, and the edge stays
  decisive after a **Deflated Sharpe** test (PSR ≈ DSR ≈ 0.9996, deflating for the
  three book configurations tried). The variance risk premium is real, not a
  selection artifact. `scripts/run_vrp.py`.
- **The edge: volatility targeting.** Scaling exposure by `target_vol /
  forecast_vol` (net of costs) holds realized vol near target and **cuts max
  drawdown by 20–49% (median ~34%) vs buy-and-hold** (−0.40 vs −0.62 across 8
  indices, close to halving only on the US indices); it improves Sharpe on US
  indices, and a jump/regime overlay trims drawdown further. A **Probabilistic
  Sharpe** check makes the nuance explicit: the targeted Sharpe is credibly > 0
  only on the US indices (PSR ≈ 0.96 for SPX/DJI) and is **indistinguishable from
  zero** on FTSE/CAC/STOXX (PSR < 0.5). A risk-control product, reported honestly.
  Vol targeting is not free alpha. `scripts/run_strategy.py`.
- **Crypto generality test (Track 3).** Computed from real Binance 5-minute
  bars for BTC/ETH/BNB/SOL (69%–134% annualised vol), so the *full* estimator
  suite (including realized quarticity) runs on real data for the first time.
  **Log-HAR is #1 and in the MCS for all four coins at every horizon**: the
  headline generalises to a 24/7, fat-tailed asset class. Two honest contrasts
  with equities: **HARQ (now testable on real quarticity) does not transfer**
  (crypto's heavy-tailed RQ makes it the worst model), and **cross-coin spillover
  is weak** (CrossHAR only marginally beats log-HAR on BTC, p≈0.09). A real-data
  signature plot confirms the microstructure-noise inflation on BTC. *Caveat:* the
  four coins are large assets that **survive today** (a dead coin such as LUNA is
  not in the panel), so this is a "log-HAR generalises to surviving major coins"
  result, not a claim over the full cross-section.
  `scripts/build_crypto.py`, `scripts/run_crypto.py`.
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
| Realized variance / quadratic variation | 1.000  | 1.0    |
| Bipower variation / integrated variance (clean) | 0.998 | 1.0 |
| Median RV / integrated variance         | 0.998  | 1.0    |
| Bipower variation / integrated variance (with jumps) | 1.044 | n/a (finite-*M* bias) |
| (RV − bipower) / jump variation         | 0.926  | ~1.0   |
| Realized kernel / QV (clean)            | 1.003  | 1.0    |
| **Realized variance / QV (with noise)** | **3.00** (inflated) | n/a |
| **Realized kernel / QV (with noise)**   | **1.008** (robust)  | 1.0 |
| Jump-test false-positive rate @ 5%      | 0.052  | 0.05   |
| Jump-test detection rate (injected)     | 0.938  | high   |

Bipower variation is jump-robust only *asymptotically*: on jump-free paths it is
unbiased (0.998), but on jump-contaminated days at *M* = 390 it carries a known
finite-sample upward bias (1.044). That is exactly why the headline forecasters
model log-variance rather than relying on a level jump correction.

The microstructure point is the classic **volatility signature plot**
(`results/figures/signature_plot.png`): under additive noise, realized variance
explodes as the sampling frequency rises, while the realized kernel stays on the
true quadratic variation.

---

## Two tracks, kept separate on purpose

- **Track 1: realized-variance benchmark** (the main result above). Models
  consume a daily realized-variance series and forecast future realized
  variance. Proxy: 5-minute realized variance (low noise).
- **Track 2: return-based GARCH** (`scripts/run_garch.py`), on real S&P 500
  daily returns (close-to-close from the same library, 2000–2022). Without an RV
  proxy here, one-step variance forecasts are scored against the squared daily
  return. Result ordering is as expected: GJR-GARCH and plain GARCH edge
  RiskMetrics, all far ahead of a constant variance.

These two tracks are **not** directly comparable: different series and,
critically, different proxy quality (a 5-minute RV proxy is far less noisy than
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
│   └── data.py          # loaders (Oxford-Man RV; SP500 returns; VIX; crypto RV)
├── scripts/             # run_{benchmark,garch,har_family,multivariate,ml,economic,
│   │                    #   vrp,strategy,regime,crypto}, validate_estimators,
│   │                    #   make_figures, build_{realized,vix,crypto}
├── tests/               # pytest suite (228 tests)
├── data/                # VIX (committed) + provenance; RV and crypto CSVs are fetched
├── results/             # tables, figures, JSON summaries (the deliverable)
├── docs/                # write-up: "why log-HAR is hard to beat"
└── report/              # LaTeX research report
```

## Methodology in brief

- **Target.** Direct multi-horizon: forecast the *average* daily variance over
  the next `h` days, so every model is scored on an identical, comparable target.
- **No look-ahead.** At each origin `t`, regression models train only on
  observations whose realization window closed by `t` (rows `s` with
  `s + h ≤ t`). This is enforced and unit-tested for every model. It is a common
  silent bug for `h > 1`.
- **Robust losses.** QLIKE and MSE-on-variance are consistent under a noisy
  variance proxy (Patton, 2011); RMSE-on-volatility is reported for reference
  but never used to rank.
- **Significance.** Diebold-Mariano with a Newey-West HAC variance and the
  Harvey-Leybourne-Newbold small-sample correction; the Model Confidence Set
  with a moving-block bootstrap (2,000 replications for the equity benchmark;
  10,000 for the pre-specified cross-asset arms).

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
python scripts/build_crypto.py           # data/crypto_realized.csv (Binance 5-min bars)
python scripts/run_crypto.py             # results/crypto.json  (Track 3: BTC/ETH/BNB/SOL)
python scripts/make_figures.py           # results/figures/*.png
pytest -q                                # 228 tests
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

MIT licence. See [LICENSE](LICENSE). Bundled data is a redistributed subset of the
Oxford-Man Institute Realized Library for reproducibility; see
[`data/README.md`](data/README.md) for provenance and credit.

## AI tool usage

This project was built with substantial help from AI coding assistants (Claude).
The author set the research questions, the methodology, and the pre-specified
decision rules, reviewed and verified all code and results, and is responsible for
the content and any errors. AI assistance covered implementation, refactoring,
documentation, and analysis support. Every statistical result is reproducible from
the committed code and a fixed seed, so the claims rest on the artifacts, not on
the tool that helped write them.
