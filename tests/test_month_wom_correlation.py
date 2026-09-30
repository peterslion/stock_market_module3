"""Checks for the month / week-of-month dummy correlation."""

import pandas as pd

from analysis.month_wom_correlation import (
    correlation_with_target,
    month_wom_dummy_columns,
    prepare_dataset,
    rounded_absolute_correlation,
    week_of_month,
)


def test_week_of_month_matches_the_day_formula():
    dates = pd.to_datetime(
        ["2024-10-01", "2024-10-07", "2024-10-08", "2024-10-31", "2024-02-29"]
    )
    assert week_of_month(pd.Series(dates)).tolist() == [1, 1, 2, 5, 5]


def test_october_week4_has_the_highest_absolute_correlation():
    raw = pd.read_parquet(
        "data/stocks_df_combined_2026_09_18.parquet.brotli",
        columns=["Date", "Month", "Weekday", "Ticker", "ticker_type", "Volume", "is_positive_growth_30d_future"],
    )
    prepared = prepare_dataset(raw)
    assert prepared["Date"].min() >= pd.Timestamp("2000-01-01")
    assert len(month_wom_dummy_columns(prepared.columns)) == 60
    assert set(prepared["month_week"]) == set(prepared["month_wom"])
    assert "October_d13" in set(prepared["month_day"])
    assert "October_w1" in set(prepared["month_wom"])

    correlations = correlation_with_target(prepared)
    assert correlations.index[0] == "October_w4"
    assert "month_wom_" not in correlations.index[0]
    assert rounded_absolute_correlation(correlations) == 0.025
