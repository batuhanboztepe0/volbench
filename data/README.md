# Bundled data — provenance and units

## `oxford_realized.csv`

A subset of the **Oxford-Man Institute Realized Library** (final public release,
v0.3), redistributed here for reproducibility. The library reports daily
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
