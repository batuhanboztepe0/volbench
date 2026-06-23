# Bundled data — provenance and units

## `oxford_realized.csv`

A subset of the **Oxford-Man Institute Realized Library** (final public release,
v0.3), used as the equity realized-volatility track. The library reports daily
realized measures computed from high-frequency intraday prices for international
equity indices.

- **Original source:** Oxford-Man Institute of Quantitative Finance, Realized
  Library, `https://realized.oxford-man.ox.ac.uk/`. The Institute closed and the
  library was discontinued in 2022; the file shipped here was retrieved from the
  Internet Archive snapshot of the official ZIP
  (`oxfordmanrealizedvolatilityindices.zip`, captured 2022-03-01).
- **Reference:** Heber, Lunde, Shephard & Sheppard (2009), *Oxford-Man Institute's
  Realized Library*. Please cite the library if you use this data.
- **Coverage:** 2000-01-03 to 2022-02-25, daily.
- **Indices bundled (8):** `.SPX` (S&P 500), `.DJI` (Dow Jones), `.FTSE`
  (FTSE 100), `.GDAXI` (DAX), `.FCHI` (CAC 40), `.STOXX50E` (EURO STOXX 50),
  `.N225` (Nikkei 225), `.HSI` (Hang Seng). The full library has 31 indices;
  `scripts/build_realized.py` re-downloads the original and re-extracts any
  subset.

### Columns

| Column          | Meaning | Units |
|-----------------|---------|-------|
| `date`          | Trading day | `YYYY-MM-DD` |
| `symbol`        | Index symbol | — |
| `rv5`           | 5-minute realized **variance** | daily, decimal returns² |
| `rv5_ss`        | 5-minute realized variance, subsampled | daily, decimal returns² |
| `bv`            | Bipower variation (5-min) | daily, decimal returns² |
| `medrv`         | Median realized variance (5-min) | daily, decimal returns² |
| `rk_parzen`     | Realized kernel, Parzen weights | daily, decimal returns² |
| `rsv`           | Realized **downside** semivariance (5-min) | daily, decimal returns² |
| `close_price`   | Index close level | price |
| `open_to_close` | Open-to-close log return | decimal |
| `nobs`          | Intraday observations used that day | count |

### Units — important

`rv5` and the other measures are **variances** (return², decimal), *not*
volatilities. They are loaded as-is by `volbench.data.load_oxford_rv`, which
**does not** square them and sanity-checks that the implied annualised
volatility (`sqrt(mean(rv5) * 252)`) lands in a plausible 3%–80% band. The
upside semivariance is reconstructed as `rsv_plus = rv5 - rsv` (both share the
same 5-minute return grid, so `rsv_minus + rsv_plus = rv5`).

There is **no realized-quarticity column** in the library, so the `HARQ` model is
exercised on the simulation track only (where quarticity is computable from
simulated intraday returns).

### License / redistribution

The Oxford-Man Realized Library was distributed for academic and research use.
Its redistribution licence is unclear, so the `oxford_realized.csv` panel is
**not committed** to the repository — `scripts/build_realized.py` fetches it from
the Internet Archive. If you build on this data, cite Heber et al. (2009) and the
Oxford-Man Institute.

## `crypto_realized.csv` (Track 3)

Daily realized measures for four crypto assets (BTC, ETH, BNB, SOL), computed
from **real Binance 5-minute klines** by `scripts/build_crypto.py`. Because the
exchange exposes free intraday bars, this panel — unlike the equity one — is
built from genuine high-frequency data, so it carries the *full* estimator suite
including **realized quarticity** (`rq`), which enables HARQ on real data.

- **Source:** Binance public REST API (`/api/v3/klines`, 5-minute interval).
- **Construction:** for each UTC day (crypto trades 24/7, ~288 bars/day, no
  overnight gap) we compute the measures with `volbench.realized`. Only daily
  measures are stored, not the raw bars.
- **Redistribution:** exchange-data terms are unclear, so the panel is **not
  committed**; `build_crypto.py` regenerates it. Tests skip when it is absent.
- **Survivorship:** the four coins are large assets still trading today, hardcoded
  in `build_crypto.py`. A coin that has since died (e.g. LUNA) is not included, so
  Track-3 results are over *surviving major* coins, not the full cross-section.
- **Units:** all measures are daily variances (decimal returns²). Annualise with
  365 (24/7 trading), not 252.
- **Columns:** `date`, `symbol`, `rv5`, `bv`, `medrv`, `rk_parzen`, `rsv`
  (downside), `rsv_plus` (upside), `rq` (realized quarticity), `close_price`,
  `open_to_close`, `nobs`.

## `crypto_expanded_realized.csv` (survivorship-corrected crypto universe)

Daily realized measures for a 22-coin crypto universe: 20 coins still live as of the study end date, plus LUNA (Terra collapse, May 2022) and FTT (FTX collapse, November 2022). These two dead coins are what makes the survivorship correction possible.

- **Source:** Binance Vision public archive (`data.binance.vision`), 5-minute klines. The Vision bucket retains delisted symbols, unlike the live Binance REST API used by `build_crypto.py`. Realized measures are computed with the same `volbench.realized` aggregation as `crypto_realized.csv`, so the two panels are directly comparable.
- **Builder:** `scripts/build_crypto_expanded.py`. Run it once to generate the file; it is not committed (exchange-data redistribution terms are unclear).
- **Universe (22 coins):** BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, TRX, LINK, MATIC, LTC, BCH, ATOM, XLM, ETC, ALGO, VET, FIL (live), plus LUNA and FTT (dead). Coin start dates vary (earliest: 2019-01-01); dead coins end at their respective collapse dates.
- **Survivorship note:** The 20 live coins were large-cap as of approximately 2022-01-01. Coins that existed but are not on this list are not included, so the live subset is still subject to selection bias. Only the two explicitly dead coins are survivorship-corrected.
- **Units and columns:** same as `crypto_realized.csv` (daily variances, decimal returns², 365-day annualisation). Columns: `date`, `symbol`, `rv5`, `bv`, `medrv`, `rk_parzen`, `rsv`, `rsv_plus`, `rq`, `close_price`, `open_to_close`, `nobs`.
- **Timestamp normalisation:** Binance Vision switched `open_time` from milliseconds to microseconds in 2025. The builder normalises both to milliseconds.

## `volare_futures_realized.csv` and `volare_forex_realized.csv` (cross-asset arms)

Daily realized measures from **VOLARE** (VOLatility Archive for Realized
Estimates) for 13 futures contracts and 13 FX pairs. These two panels feed the
pre-specified cross-asset transfer matrix (the futures and FX confirmatory arms).

- **Source:** VOLARE token-authenticated REST API (`https://volare.unime.it/api`),
  pulled by `scripts/build_volare.py --fetch futures` and `--fetch forex`. A (free)
  VOLARE account is required.
- **Redistribution:** VOLARE requests citation but grants no explicit
  redistribution licence (the arXiv CC-BY tag covers the paper, not the data, and
  the portal states only a citation requirement). So, like the Oxford-Man and
  crypto panels, these CSVs are **not committed**; `build_volare.py` re-fetches
  them, and the cross-asset scripts skip when they are absent. If you use the data
  you MUST cite VOLARE:
  > Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., &
  > Spagnolo, F. (2026). VOLatility Archive for Realized Estimates (VOLARE).
  > arXiv:2602.19732. https://doi.org/10.48550/arXiv.2602.19732. VOLARE page:
  > https://volare.unime.it.
- **Coverage (futures, 13):** `C` (corn), `CL` (WTI crude), `ES` (E-mini S&P 500),
  `EU` (euro FX future), `FV` (5-yr T-note), `GC` (gold), `HG` (copper), `NG`
  (natural gas), `NQ` (E-mini Nasdaq), `S` (soybeans), `SI` (silver), `TY` (10-yr
  T-note), `W` (wheat). 2009-09-28 to 2026-05-29, daily.
- **Coverage (FX, 13):** `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`, `USDCAD`,
  `USDCHF`, `USDJPY`, `USDKRW`, `USDNOK`, `USDPLN`, `USDSEK`, `USDSGD`, `ZARUSD`
  (7 majors plus 6 secondary/EM). 2009-09-25 to 2026-05-29, daily.
- **Units:** all measures are daily variances (decimal returns²), the same
  convention as the equity panel. They are loaded as-is, not squared.
- **Columns:** `date`, `symbol`, `rv5` (5-min realized variance), `bv` (bipower
  variation), `medrv` (median realized variance), `rk_parzen` (realized kernel,
  Parzen weights), `rsv` (downside semivariance), `rq` (realized quarticity),
  `close_price`, `open_to_close` (log return), `nobs` (intraday observations used).

## `vix.csv`

CBOE Volatility Index (VIX), daily close, used by the variance-risk-premium
study (`volbench.vrp`).

- **Source:** Federal Reserve Bank of St. Louis (FRED), series `VIXCLS`
  (`https://fred.stlouisfed.org/series/VIXCLS`), which republishes CBOE's index.
- **License:** public domain (FRED data) — committed directly to the repo;
  `scripts/build_vix.py` refreshes it.
- **Coverage:** 2000-01-03 to 2022-12-30, daily.
- **Columns:** `date` (`YYYY-MM-DD`), `vix` (annualised implied volatility, in
  percent). Convert to an implied **daily variance** with `(vix/100)**2 / 252`
  before comparing to realized variance.
