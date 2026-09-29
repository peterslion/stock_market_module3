"""Checks for the month / week-of-month dummy correlation."""

import pandas as pd

from analysis.month_wom_correlation import (
    CATEGORICAL_FEATURES,
    add_categorical_dummies,
    add_month_wom,
    correlation_with_target,
    month_wom_dummy_columns,
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
        columns=["Date", "Month", "Weekday", "Ticker", "ticker_type", "is_positive_growth_30d_future"],
    )
    prepared = add_categorical_dummies(add_month_wom(raw))
    dummy_columns = [
        column
        for column in prepared.columns
        if column.startswith(tuple(f"{name}_" for name in CATEGORICAL_FEATURES))
    ]
    assert len(month_wom_dummy_columns(prepared.columns)) == 60
    assert len(dummy_columns) == 115
    assert "October_w1" in set(prepared["month_wom"])

    correlations = correlation_with_target(prepared)
    assert correlations.index[0] == "month_wom_October_w4"
    assert rounded_absolute_correlation(correlations) == 0.021
