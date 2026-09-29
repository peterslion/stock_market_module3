# Question 5: What data is missing?

Which new indicators should be added, given the correlation results and the decision tree?

The best tree has test precision 0.629 at depth 4. Deeper trees raise validation precision to 0.946 and lower test precision. The label is only partly explained by the current columns, so the next gain is new information rather than a deeper tree.

What the current results already use:

- The root split is US core CPI (`cpi_core_yoy` around 2%), then US yields (`DGS10`, `DGS1`) and `FEDFUNDS`.
- The same tree then splits on WTI, gold, Bitcoin, the S&P 500, and the Dow.
- Linear correlations of those macro and commodity series with `is_positive_growth_30d_future` are small, about 0.03 to 0.06, but they are the strongest columns that are not the future return itself.
- Seasonality shows up as `October_w4`, `November_w3`, and a July split (`Month_7`).

What the file does not contain is a local macro block for Europe and India, a real yield, a market-wide volatility series, and an earnings calendar. The panel is US, EU, and India stocks, while the macro columns are US-only. `ticker_type` and `growth_epi_*` are the only India-specific context.

## Indicators to add

1. **US 10-year real yield and breakeven inflation.** The tree rebuilds a real-rate regime by stacking `cpi_core_yoy`, `DGS10`, and `FEDFUNDS`. One series would replace that stack. Source: FRED `DFII10` (10-year TIPS yield) and `T10YIE` (10-year breakeven).

2. **ECB deposit rate, euro-area core inflation, and the Bund 10-year yield, plus the RBI repo rate and India CPI.** EU names such as ASML, SAP, MC.PA, and TTE, and India names such as RELIANCE.NS, TCS.NS, and HDFC Bank, are scored with US inflation and the Fed. Source: FRED and the ECB Statistical Data Warehouse for Europe; the Reserve Bank of India for the repo rate and India CPI.

3. **EURUSD and USDINR growth at 30, 90, and 365 days.** Local-currency prices are joined to a US rate regime with no exchange rate in between. Source: FRED `DEXUSEU` and `DEXINUS`.

4. **VIX, VSTOXX, and India VIX.** Stock-level `atr` and `volatility` are already in the file and are weakly related to the 30-day sign. The tree uses oil, gold, and Bitcoin as stand-ins for risk appetite. Source: FRED `VIXCLS`, STOXX for VSTOXX, and NSE for India VIX.

5. **Days until the next earnings announcement, and the last earnings surprise.** October and November weeks are the strongest `month_wom` dummies, and July is an explicit tree split. A week-of-month dummy cannot tell earnings season for NVDA from earnings season for ITC.NS. Source: the company filings already reachable through Yahoo Finance, which is the price source used to build this file.

A column that can be built without a new download is relative growth: stock `growth_30d` divided by `growth_snp500_30d`, `growth_dax_30d`, or `growth_epi_30d` according to `ticker_type`. The tree sees the stock and the market index separately. It never sees whether the stock is ahead of its own market.
