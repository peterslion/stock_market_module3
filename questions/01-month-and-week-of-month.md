# Question 1: Dummies for Month and Week-of-Month

What is the absolute correlation of the most correlated week-of-month dummy with the binary outcome `is_positive_growth_30d_future`?

October and November are potentially important seasonal months. This question goes further by generating dummy variables for both the month and the week of the month (starting from 1). The first week of October is coded as `October_w1`.

## Steps

1. Week of month is `(day - 1) // 7 + 1`.
2. `month_wom` combines the month name and that week, for example `October_w1` or `November_w2`.
3. Categorical features:

   - `Month`
   - `Weekday`
   - `Ticker`
   - `ticker_type`
   - `month_wom`

4. `pandas.get_dummies()` encodes those fields. This file produces 115 dummy columns, including 60 week-of-month indicators.
5. `DataFrame.corr()` measures the Pearson correlation of each feature with `is_positive_growth_30d_future`.
6. Keep the week-of-month dummies, take the absolute correlation, and sort descending.

The new dummies stay in the dataset for later questions.

## Answer

**0.021**

The largest absolute correlation is `October_w4` (raw value `0.021180`). The ranked list is `results/month_wom_correlations.csv`.

Source data: [stocks_df_combined_2026_09_18.parquet.brotli](https://drive.google.com/uc?id=1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5).
