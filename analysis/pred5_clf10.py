"""Depth-10 decision tree and the test rows only that tree gets right.

Hand rules pred0-pred2 come from the module notebook. pred3 and pred4 are the
2026 macro rules. The tree is fit on train plus validation with
``DecisionTreeClassifier(max_depth=10, random_state=42)`` and then applied to
every row. Features are the notebook numerical set plus dummies for Month
(calendar month number), Weekday, Ticker, ticker_type, and month_wom.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from analysis.hand_rules import (
    DATA_DIR,
    PRED3,
    PRED4,
    RAW_FILENAME,
    RESULTS_DIR,
    TARGET,
    temporal_split,
)

PRED0 = "pred0_manual_cci"
PRED1 = "pred1_manual_prev_g1"
PRED2 = "pred2_manual_prev_g1_and_snp"
PRED5 = "pred5_clf_10"
HAND_RULES = [PRED0, PRED1, PRED2, PRED3, PRED4]

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
CUSTOM_NUMERICAL = [
    "SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative", "volatility", "ln_volume",
]
MACRO = [
    "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
    "FEDFUNDS", "DGS1", "DGS5", "DGS10",
]
CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]


def numerical_features(columns: list[str]) -> list[str]:
    growth = [column for column in columns if column.startswith("growth_") and "future" not in column]
    patterns = [column for column in columns if "cdl" in column]
    return growth + TECHNICAL_INDICATORS + patterns + CUSTOM_NUMERICAL + MACRO


def prepare_modeling_frame(source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows from 2000-01-01 with the five hand rules and the temporal split."""
    if source is None:
        source = pd.read_parquet(DATA_DIR / RAW_FILENAME)
    frame = source[source["Date"] >= "2000-01-01"].copy()
    frame["ln_volume"] = frame["Volume"].replace(0, np.nan).fillna(1e-9).map(np.log)
    frame["Weekday"] = frame["Weekday"].astype(str)
    # Notebook encodes Month as the month number. month_wom keeps the English week label.
    frame["month_wom"] = frame["Date"].dt.month_name() + "_w" + ((frame["Date"].dt.day - 1) // 7 + 1).astype(str)
    frame["Month"] = frame["Date"].dt.month.astype(str)
    frame = temporal_split(frame, frame["Date"].min(), frame["Date"].max())
    frame[PRED0] = (frame["cci"] > 200).astype(int)
    frame[PRED1] = (frame["growth_30d"] > 1).astype(int)
    frame[PRED2] = ((frame["growth_30d"] > 1) & (frame["growth_snp500_30d"] > 1)).astype(int)
    frame[PRED3] = ((frame["DGS10"] <= 4) & (frame["DGS5"] <= 1)).astype(int)
    frame[PRED4] = ((frame["DGS10"] > 4) & (frame["FEDFUNDS"] <= 4.795)).astype(int)
    return frame


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Numerical features plus categorical dummies, with infinities and NaNs set to 0."""
    features = numerical_features(list(frame.columns))
    dummies = pd.get_dummies(frame[CATEGORICAL], columns=CATEGORICAL, dtype="int8")
    matrix = pd.concat(
        [frame[features].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0)


def finite_row_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where every numerical feature is finite. Missing values count as not finite."""
    features = numerical_features(list(frame.columns))
    return np.isfinite(frame[features].to_numpy(dtype=float)).all(axis=1)


def remove_infinite_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a non-finite value in any numerical feature.

    ``np.isfinite`` is false for both infinities and missing values, matching
    the notebook helper. Outlier percentiles are not applied: the notebook
    keeps those rows.
    """
    return frame.loc[finite_row_mask(frame)].copy()


def fit_pred5(frame: pd.DataFrame) -> pd.Series:
    """Fit a depth-10 tree on finite train+validation rows and predict those rows."""
    kept = remove_infinite_values(frame)
    matrix = feature_matrix(kept)
    y = kept[TARGET].astype(int).reset_index(drop=True)
    train_rows = kept["split"].isin(["train", "validation"]).to_numpy()
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(matrix.loc[train_rows], y.loc[train_rows])
    predicted = np.full(len(frame), np.nan)
    predicted[finite_row_mask(frame)] = clf.predict(matrix)
    return pd.Series(predicted, index=frame.index, name=PRED5)


def only_pred5_is_correct(frame: pd.DataFrame) -> pd.Series:
    """1 when pred5 matches the target and every hand rule does not."""
    target = frame[TARGET].to_numpy()
    pred5_correct = frame[PRED5].to_numpy() == target
    hands_wrong = np.ones(len(frame), dtype=bool)
    for column in HAND_RULES:
        hands_wrong &= frame[column].to_numpy() != target
    return pd.Series((pred5_correct & hands_wrong).astype(int), index=frame.index, name="only_pred5_is_correct")


def count_unique_correct_on_test(frame: pd.DataFrame) -> int:
    test = frame["split"].eq("test")
    return int(frame.loc[test, "only_pred5_is_correct"].sum())


def build_predictions(source: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = prepare_modeling_frame(source)
    frame[PRED5] = fit_pred5(frame)
    frame["only_pred5_is_correct"] = only_pred5_is_correct(frame)
    return frame


def write_results(count: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "prediction": PRED5,
        "test_only_pred5_is_correct": count,
        "max_depth": 10,
        "random_state": 42,
        "trained_on": ["train", "validation"],
    }
    (RESULTS_DIR / "q3_pred5_answer.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    frame = build_predictions()
    count = count_unique_correct_on_test(frame)
    write_results(count)
    print(f"TEST rows where only pred5_clf_10 is correct: {count}")


if __name__ == "__main__":
    main()
