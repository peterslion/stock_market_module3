"""Month and week-of-month dummy correlation with future 30-day growth.

Builds categorical indicators for Month, Weekday, Ticker, ticker_type, and
month_wom (for example ``October_w1``), then reports the month_wom dummy
with the largest absolute Pearson correlation to
``is_positive_growth_30d_future``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DRIVE_FILE_ID = "1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5"
RAW_FILENAME = "stocks_df_combined_2026_09_18.parquet.brotli"
PREPARED_FILENAME = "stocks_with_month_wom_dummies.parquet"

TARGET = "is_positive_growth_30d_future"
CATEGORICAL_FEATURES = [
    "Month",
    "Weekday",
    "Ticker",
    "ticker_type",
    "month_wom",
    "month_week",
    "month_day",
]
OHLCV = ["Open", "High", "Low", "Close", "Adj Close_x", "Volume"]
CUSTOM_NUMERICAL = [
    "SMA10",
    "SMA20",
    "growing_moving_average",
    "high_minus_low_relative",
    "volatility",
    "ln_volume",
]
TECHNICAL_INDICATORS = [
    "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc", "bop", "cci", "cmo", "dx",
    "macd", "macdsignal", "macdhist", "macd_ext", "macdsignal_ext", "macdhist_ext",
    "macd_fix", "macdsignal_fix", "macdhist_fix", "mfi", "minus_di", "mom", "plus_di",
    "dm", "ppo", "roc", "rocp", "rocr", "rocr100", "rsi", "slowk", "slowd", "fastk",
    "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr", "ad", "adosc", "obv",
    "atr", "natr", "ht_dcperiod", "ht_dcphase", "ht_phasor_inphase", "ht_phasor_quadrature",
    "ht_sine_sine", "ht_sine_leadsine", "ht_trendmod", "avgprice", "medprice", "typprice",
    "wclprice",
]
MACRO = [
    "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
    "FEDFUNDS", "DGS1", "DGS5", "DGS10",
]
SAMPLE_START = "2000-01-01"


def raw_data_path() -> Path:
    return DATA_DIR / RAW_FILENAME


def prepared_data_path() -> Path:
    return DATA_DIR / PREPARED_FILENAME


def download_raw_data(destination: Path | None = None) -> Path:
    """Download the source parquet from Google Drive if it is not local yet."""
    destination = destination or raw_data_path()
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={DRIVE_FILE_ID}&export=download&confirm=t"
    )
    import urllib.request

    print(f"Downloading source data to {destination} ...")
    urllib.request.urlretrieve(url, destination)
    return destination


def growth_columns(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in columns if column.startswith("growth_") and "future" not in column]


def to_predict_columns(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in columns if "future" in column]


def technical_pattern_columns(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in columns if "cdl" in column]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Column groups from the notebook's dataset-preparation snippet."""
    columns = list(df.columns)
    categorical = ["Month", "Weekday", "Ticker", "ticker_type"]
    growth = growth_columns(columns)
    patterns = technical_pattern_columns(columns)
    to_drop = [
        column
        for column in ["Year", "Date", "index_x", "index_y", "index", "Quarter", "Adj Close_y"]
        + categorical
        + OHLCV
        if column in df.columns
    ]
    numerical = growth + [column for column in TECHNICAL_INDICATORS if column in df.columns]
    numerical += patterns + [column for column in CUSTOM_NUMERICAL if column in df.columns]
    numerical += [column for column in MACRO if column in df.columns]
    return {
        "GROWTH": growth,
        "TO_PREDICT": to_predict_columns(columns),
        "TECHNICAL_PATTERNS": patterns,
        "MACRO": [column for column in MACRO if column in df.columns],
        "NUMERICAL": numerical,
        "TO_DROP": to_drop,
    }


def add_ln_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zero volume before the log so the transform stays finite."""
    out = df.copy()
    out["ln_volume"] = out["Volume"].replace(0, np.nan).fillna(1e-9).map(np.log)
    return out


def ticker_date_span(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Ticker")["Date"].agg(["min", "max", "count"])


def limit_to_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["Date"] >= SAMPLE_START].copy()


def week_of_month(dates: pd.Series) -> pd.Series:
    """Week of month starting at 1. Week 1 is days 1-7, week 2 is days 8-14, and so on."""
    return (dates.dt.day - 1) // 7 + 1


def add_month_wom(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``month_wom`` labels such as ``October_w1`` and code ``Month`` as a name.

    The source file stores ``Month`` as the first timestamp of each month. Dummy
    encoding needs the English month name (12 levels), which is what produces
    about 115 categorical indicators together with week-of-month.
    The original month-start timestamp is kept as ``month_start``.
    """
    out = df.copy()
    if "month_start" not in out.columns:
        out["month_start"] = out["Month"]
    out["Month"] = out["Date"].dt.month_name()
    week = week_of_month(out["Date"]).astype(str)
    out["month_wom"] = out["Month"] + "_w" + week
    out["month_week"] = out["month_wom"]
    out["month_day"] = out["Month"] + "_d" + out["Date"].dt.day.astype(str)
    return out


def add_categorical_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Append dummy columns and keep the original categorical fields.

    ``columns`` is passed explicitly so integer fields such as ``Weekday`` are
    encoded along with the string fields. Without that argument, pandas only
    dummy-encodes object, string, and category columns.
    """
    encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype="int8")
    # Drop any previously generated dummy columns so the function is idempotent.
    existing = [column for column in encoded.columns if column in df.columns]
    base = df.drop(columns=existing) if existing else df
    return pd.concat([base, encoded], axis=1)


def month_wom_dummy_columns(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in columns if column.startswith("month_wom_")]


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of each month_wom dummy with the binary target.

    ``DataFrame.corr`` is pairwise. These dummies and the target are complete,
    so the month_wom correlations do not depend on other columns.
    """
    feature_columns = month_wom_dummy_columns(df.columns)
    if not feature_columns:
        raise ValueError("No month_wom dummy columns found. Call add_categorical_dummies first.")

    correlated = df[feature_columns + [TARGET]].corr(numeric_only=True)[TARGET]
    result = correlated.drop(labels=[TARGET]).rename("corr").to_frame()
    result["abs_corr"] = result["corr"].abs()
    result = result.sort_values("abs_corr", ascending=False)
    # get_dummies names columns month_wom_October_w4. Report the week label itself.
    result.index = result.index.str.removeprefix("month_wom_")
    result.index.name = "feature"
    return result


def highest_absolute_correlation(correlations: pd.DataFrame) -> float:
    return float(correlations["abs_corr"].iloc[0])


def rounded_absolute_correlation(correlations: pd.DataFrame, digits: int = 3) -> float:
    return float(np.round(highest_absolute_correlation(correlations), digits))


def prepare_dataset(source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the post-2000 frame with calendar labels and categorical dummies.

    Preparation follows the notebook: log volume, the GROWTH / TO_PREDICT /
    TECHNICAL_PATTERNS / MACRO / NUMERICAL / TO_DROP lists, a per-ticker date
    summary, and the ``Date >= 2000-01-01`` sample.
    """
    if source is None:
        source = pd.read_parquet(download_raw_data())
    prepared = add_ln_volume(source)
    sets = feature_sets(prepared)
    prepared.attrs["feature_sets"] = sets
    span = ticker_date_span(prepared)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    span.to_csv(RESULTS_DIR / "ticker_date_span.csv")
    featured = add_month_wom(limit_to_sample(prepared))
    featured.attrs["feature_sets"] = sets
    return add_categorical_dummies(featured)


def write_results(correlations: pd.DataFrame) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(RESULTS_DIR / "month_wom_correlations.csv")

    top_feature = str(correlations.index[0])
    raw_abs = highest_absolute_correlation(correlations)
    answer = {
        "target": TARGET,
        "most_correlated_month_wom_dummy": top_feature,
        "correlation": float(correlations.iloc[0]["corr"]),
        "absolute_correlation": raw_abs,
        "absolute_correlation_rounded_3dp": rounded_absolute_correlation(correlations),
        "n_month_wom_dummies": int(len(correlations)),
    }
    (RESULTS_DIR / "q1_month_wom_answer.json").write_text(json.dumps(answer, indent=2) + "\n")
    return answer


def main() -> None:
    print("Loading and preparing features ...")
    prepared = prepare_dataset()
    dummy_columns = [
        column
        for column in prepared.columns
        if column.startswith(tuple(f"{name}_" for name in CATEGORICAL_FEATURES))
    ]
    month_wom_columns = month_wom_dummy_columns(prepared.columns)
    print(f"rows: {len(prepared):,}")
    print(f"categorical dummies: {len(dummy_columns)}")
    print(f"month_wom dummies: {len(month_wom_columns)}")

    correlations = correlation_with_target(prepared)
    answer = write_results(correlations)

    prepared_path = prepared_data_path()
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing prepared dataset to {prepared_path} ...")
    prepared.to_parquet(prepared_path, index=False)

    print()
    print(correlations.head(10).to_string())
    print()
    print(
        "Highest absolute correlation: "
        f"{answer['absolute_correlation_rounded_3dp']:.3f} "
        f"({answer['most_correlated_month_wom_dummy']})"
    )


if __name__ == "__main__":
    main()
