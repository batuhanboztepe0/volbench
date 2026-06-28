# volbench: References

*Consolidated citation list for the volbench benchmark and the cross-asset
break-HAR study.*

**Verification key:**
`✓✓` bibliographic details independently confirmed in the 2026-06-20 literature
fact-check (title/authors/venue/DOI checked) ·
`✓` standard, well-established reference (author/year/title/venue reliable;
confirm exact volume/pages at typesetting) ·
`⚠` incomplete or needs confirmation; do **not** cite as-is.

---

## 1. Core methodology: HAR family, losses, comparison tests

- `✓` Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
  Volatility." *Journal of Financial Econometrics*, 7(2), 174–196.
- `✓` Andersen, T.G., Bollerslev, T., Diebold, F.X., Labys, P. (2003). "Modeling
  and Forecasting Realized Volatility." *Econometrica*, 71(2), 579–625. (log-RV)
- `✓` Patton, A.J. (2011). "Volatility forecast comparison using imperfect
  volatility proxies." *Journal of Econometrics*, 160(1), 246–256. (proxy-robust losses)
- `✓` Hansen, P.R., Lunde, A., Nason, J.M. (2011). "The Model Confidence Set."
  *Econometrica*, 79(2), 453–497. (MCS)
- `✓` Diebold, F.X., Mariano, R.S. (1995). "Comparing Predictive Accuracy."
  *Journal of Business & Economic Statistics*, 13(3), 253–263.
- `✓` Harvey, D., Leybourne, S., Newbold, P. (1997). "Testing the equality of
  prediction mean squared errors." *International Journal of Forecasting*, 13(2),
  281–291. (HLN small-sample correction to DM)
- `✓` Bollerslev, T., Patton, A.J., Quaedvlieg, R. (2016). "Exploiting the errors:
  A simple approach for improved volatility forecasting." *Journal of
  Econometrics*, 192(1), 1–18. (HARQ)
- `✓` Patton, A.J., Sheppard, K. (2015). "Good Volatility, Bad Volatility: Signed
  Jumps and the Persistence of Volatility." *Review of Economics and Statistics*,
  97(3), 683–697. (semivariance / SHAR)
- `✓` Andersen, T.G., Bollerslev, T., Diebold, F.X. (2007). "Roughing It Up:
  Including Jump Components in the Measurement, Modeling, and Forecasting of Return
  Volatility." *Review of Economics and Statistics*, 89(4), 701–720. (continuous/jump, HAR-CJ)
- `✓` Mincer, J., Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts." In
  *Economic Forecasts and Expectations*, NBER. (Mincer–Zarnowitz regression)
- `✓` Corsi, F., Reno, R. (2012). "Discrete-time volatility forecasting with
  persistent leverage effect and the link with continuous-time volatility
  modeling." *Review of Financial Studies*, 25(5), 1336–1369. (leverage-HAR)

## 2. Realized estimators

- `✓✓` Andersen, T.G., Bollerslev, T. (1998). "Answering the Skeptics: Yes,
  Standard Volatility Models Do Provide Accurate Forecasts." *International
  Economic Review*, 39(4), 885–905.
- `✓` Barndorff-Nielsen, O.E., Shephard, N. (2004). "Power and Bipower Variation
  with Stochastic Volatility and Jumps." *Journal of Financial Econometrics*, 2(1), 1–37.
- `✓` Barndorff-Nielsen, O.E., Shephard, N. (2006). "Econometrics of Testing for
  Jumps in Financial Economics Using Bipower Variation." *Journal of Financial
  Econometrics*, 4(1), 1–30. (BNS jump test)
- `✓` Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., Shephard, N. (2008).
  "Designing Realized Kernels to Measure the Ex Post Variation of Equity Prices in
  the Presence of Noise." *Econometrica*, 76(6), 1481–1536. (realized kernels)

## 3. ML vs HAR debate

- `✓✓` Audrino, F., Chassot, J. (2024). "HARd to Beat: The Overlooked Impact of
  Rolling Windows in the Era of Machine Learning." arXiv:2406.08041. (properly-fitted
  HAR beats ML across 1,455 stocks; volbench aligns with this side)
- `✓✓` Christensen, K., Siggaard, M., Veliyev, B. (2023). "A Machine Learning
  Approach to Volatility Forecasting." *Journal of Financial Econometrics*, 21(5),
  1680–1727. (ML beats HAR, the opposing side)

## 4. Cross-asset / spillover

- `✓✓` Zhang, C., Pu, X., Cucuringu, M., Dong, X. (2025). "Forecasting realized
  volatility with spillover effects: Perspectives from graph neural networks."
  *International Journal of Forecasting*, 41(1), 377–397. (graph/GHAR spillover)
- `✓` Bollerslev, T., Hood, B., Huss, J., Pedersen, L.H. (2018). "Risk Everywhere:
  Modeling and Managing Volatility." *Review of Financial Studies*, 31(7), 2729–2773.
- `✓✓` Mallory, M.L. (2026). "Two-Step Regularized HARX to Measure Volatility
  Spillovers in Multi-Dimensional Systems." arXiv:2601.03146. (HAR-ElasticNet on 6
  futures; finds univariate HAR matches on point forecasts, which corroborates HAR robustness)

## 5. Crypto volatility

- `✓✓` Qiu, Y., Wang, Z., Xie, T., Zhang, X. (2021). "Forecasting Bitcoin realized
  volatility by exploiting measurement error under model uncertainty." *Journal of
  Empirical Finance*, 62, 179–201. DOI:10.1016/j.jempfin.2021.03.003.
  (endorses a **model-averaged HARQ-type** estimator; volbench contradicts the
  *direction* for plain HARQ)
- `✓✓` Qiu, Y., Wang, Y., Xie, T. (2021). "Forecasting Bitcoin realized volatility
  by measuring the spillover effect among cryptocurrencies." *Economics Letters*,
  208, 110092. DOI:10.1016/j.econlet.2021.110092. (positive short-horizon spillover)
- `✓✓` Yi, …, He, …, Zhang, … (2022). "Out-of-sample prediction of Bitcoin realized
  volatility: Do other cryptocurrencies help?" *North American Journal of Economics
  and Finance*, 62, 101731. DOI:10.1016/j.najef.2022.101731. (positive cross-crypto
  predictors with DM/MCS)
- `✓` Korkusuz, B., Sahiner, M. (2025). "Coin impact on cross-crypto realized
  volatility and dynamic cryptocurrency volatility connectedness." *Financial
  Innovation*, 11(1). DOI:10.1186/s40854-025-00881-x. (HAR vs LSTM under the MCS,
  plus a TVP-VAR / Diebold-Yilmaz connectedness arm. Adding large-cap RV improves
  mid-cap (XRP/LTC) forecasts but not large-cap (BTC/ETH). Consistent with the weak
  large-cap spillover gain found here, but its method is connectedness, not a DM
  horse-race.)
- `✓✓` Brini, A., Lenz, J. (2024). "A Comparison of Cryptocurrency
  Volatility-benchmarking New and Mature Asset Classes." arXiv:2404.04962.
  (positive high-frequency leverage effect in crypto; volatility-benchmarking
  across asset classes)

## 6. Economic / risk layer (VaR, Sharpe, ES)

- `✓` Bollerslev, T. (1986). "Generalized autoregressive conditional
  heteroskedasticity." *Journal of Econometrics*, 31(3), 307–327. (GARCH)
- `✓` Glosten, L.R., Jagannathan, R., Runkle, D.E. (1993). "On the relation
  between the expected value and the volatility of the nominal excess return on
  stocks." *Journal of Finance*, 48(5), 1779–1801. (GJR-GARCH)
- `✓` Engle, R.F., Manganelli, S. (2004). "CAViaR: Conditional Autoregressive Value
  at Risk by Regression Quantiles." *Journal of Business & Economic Statistics*,
  22(4), 367–381. (CAViaR + DQ test)
- `✓` Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk
  Measurement Models." *Journal of Derivatives*, 3(2), 73–84. (unconditional coverage)
- `✓` Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." *International
  Economic Review*, 39(4), 841–862. (conditional coverage)
- `✓` Acerbi, C., Székely, B. (2014). "Back-testing expected shortfall." *Risk*,
  27(11), 76–81.
- `✓` Fissler, T., Ziegel, J.F. (2016). "Higher order elicitability and Osband's
  principle." *Annals of Statistics*, 44(4), 1680–1707. (FZ loss for ES)
- `✓` Bailey, D.H., López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal
  of Portfolio Management*, 40(5), 94–107. (PSR / DSR)

## 7. Positioning: breadth-as-contribution benchmarks

- `✓✓` Liu, L.Y., Patton, A.J., Sheppard, K. (2015). "Does anything beat 5-minute
  RV? A comparison of realized measures across multiple asset classes." *Journal of
  Econometrics*, 187(1), 293–311. DOI:10.1016/j.jeconom.2015.07.006.
  (the breadth-not-model analog volbench emulates)
- `✓✓` Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., &
  Spagnolo, F. (2026). VOLatility Archive for Realized Estimates (VOLARE).
  arXiv:2602.19732 [q-fin.ST]. https://doi.org/10.48550/arXiv.2602.19732.
  (preprint; an open realized-estimator data infrastructure, not a forecasting model)
- `✓` Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2020). "The M4 Competition:
  100,000 time series and 61 forecasting methods." *International Journal of
  Forecasting*, 36(1), 54–74. (M5: 2022, *IJF* 38(4), 1346–1364)

## 8. Pre-registration / methodology of science

- `✓✓` (2023). "Registered reports adoption across scientific domains".
  *Scientometrics*. DOI:10.1007/s11192-023-04896-y. (≤1% adoption in economics vs
  7% psychology; supports the "near-absent in fin-econ / first-mover" framing)
- `✓` Arpinon, T., Espinosa, R. (2023). "A practical guide to registered reports for
  economists." *Journal of the Economic Science Association*.

---

## 9. Data sources

- **Oxford-Man Realized Library.** Heber, G., Lunde, A., Shephard, N., Sheppard, K.
  (2009). "Oxford-Man Institute's Realized Library," University of Oxford. `✓`
  *Used for:* 8 equity indices (.SPX, .FTSE, .N225, …), 2000–2022, ~5,000
  OOS origins/index (Track 1 headline). 5-minute RV proxy.
- **Binance Vision.** Binance public historical market data
  (`data.binance.vision`), 5-minute klines; retains delisted symbols. `✓`
  *Used for:* crypto Track 3: 4 coins (BTC/ETH/BNB/SOL) and the expanded,
  survivorship-corrected 22-coin universe (20 live + LUNA/FTT dead). Builder:
  `scripts/build_crypto_expanded.py`.
- **VOLARE (VOLatility Archive for Realized Estimates).** Accessed via the VOLARE
  REST API (`https://volare.unime.it/api`); fetched, not redistributed (VOLARE
  requests citation; no explicit redistribution licence found). `✓✓`
  *Used for:* 13 futures contracts (rates FV/TY; commodity CL/NG/GC/SI/HG/C/S/W;
  equity-index ES/NQ; FX EU) and 13 FX pairs (7 major, 6 EM/secondary),
  2009-09-28 to 2026-05-29. Access: `scripts/build_volare.py --fetch futures` and
  `--fetch forex`.
  **Required attribution.** VOLARE asks that you cite both the paper and the page:
  Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., & Spagnolo, F.
  (2026). VOLatility Archive for Realized Estimates (VOLARE). arXiv:2602.19732
  [q-fin.ST]. https://doi.org/10.48550/arXiv.2602.19732. VOLARE page:
  https://volare.unime.it.
- **CBOE VIX.** CBOE Volatility Index daily series. `✓` *Used for:* the variance
  risk premium (implied vs realized) edge layer.
- **S&P 500 daily returns.** *Used for:* Track 2 (return-based GARCH reference,
  scored on squared daily return; never compared to Track 1). `✓`

---

*Last updated 2026-06-20. `✓` marks a citation whose page and volume numbers were
confirmed against the source.*
