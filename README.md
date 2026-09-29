# Month and week-of-month growth correlations

Question write-ups:

- [Question 1: month and week-of-month dummies](questions/01-month-and-week-of-month.md)
- [Question 2: new hand rules](questions/02-hand-rules.md)
- [Question 3: depth-10 tree](questions/03-pred5-clf10.md)
- [Question 4: best tree depth](questions/04-best-tree-depth.md)
- [Question 5: missing data](questions/05-missing-data.md)

This analysis measures which week-of-month seasonal dummy is most linearly
associated with the binary label `is_positive_growth_30d_future`.

Source file: `stocks_df_combined_2026_09_18.parquet.brotli`
([Google Drive](https://drive.google.com/uc?id=1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5)).

## Answer

**0.021**

That is the absolute Pearson correlation, rounded to three decimal places, of
the strongest week-of-month dummy. The label is `October_w4`
(correlation `0.021180`). October and November weeks occupy the top of the
ranking, which matches the earlier observation that those months matter for
the sign of 30-day-ahead growth.

## How the features are built

Week of month starts at 1:

```text
(day - 1) // 7 + 1
```

`month_wom` joins the English month name to that week, for example
`October_w1` or `November_w2`. Categorical fields passed to
`pandas.get_dummies` are:

- `Month` (recoded from the month-start timestamp to the month name)
- `Weekday`
- `Ticker`
- `ticker_type`
- `month_wom`

On this file that is 115 dummy columns: 12 months, 7 weekdays, 33 tickers,
3 ticker types, and 60 month-week labels (12 months × 5 weeks). The original
categorical columns stay in the frame, and the dummy columns are kept so later
steps can use them as features. The original month-start timestamp is preserved
as `month_start`.

## Run

```bash
pip install -r requirements.txt
python analysis/month_wom_correlation.py
```

The script downloads the source parquet into `data/` when it is missing, writes
`data/stocks_with_month_wom_dummies.parquet`, and saves:

- `results/q1_month_wom_answer.json`
- `results/month_wom_correlations.csv` (every week label such as `October_w4`, sorted by `abs_corr`)

`pandas.get_dummies` still stores those indicators as `month_wom_October_w4` in the dataset. The correlation table drops that prefix and keeps the week label.

Large data files under `data/` are not committed.
