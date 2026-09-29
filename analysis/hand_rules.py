"""Manual hand rules pred3 and pred4, scored on the temporal test split.

The split and precision definition follow Module 3 of the 2026 stock-markets
notebook: keep dates from 2000-01-01, cut train/validation/test at 70/15/15
of the calendar span, and score precision only where a rule predicts 1.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_FILENAME = "stocks_df_combined_2026_09_18.parquet.brotli"
TARGET = "is_positive_growth_30d_future"

PRED3 = "pred3_manual_dgs10_5"
PRED4 = "pred4_manual_dgs10_fedfunds"
NEW_PREDICTIONS = [PRED3, PRED4]


def temporal_split(df: pd.DataFrame, min_date, max_date, train_prop: float = 0.7, val_prop: float = 0.15) -> pd.DataFrame:
    """Assign train / validation / test from the Date column, matching the notebook."""
    out = df.copy()
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
    out["split"] = "test"
    out.loc[out["Date"] <= val_end, "split"] = "validation"
    out.loc[out["Date"] <= train_end, "split"] = "train"
    return out


def add_new_hand_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Positive-branch rules taken from the 2026 depth-10 tree.

    pred3: (DGS10 <= 4) and (DGS5 <= 1)
    pred4: (DGS10 > 4) and (FEDFUNDS <= 4.795)
    """
    out = df.copy()
    out[PRED3] = ((out["DGS10"] <= 4) & (out["DGS5"] <= 1)).astype(int)
    out[PRED4] = ((out["DGS10"] > 4) & (out["FEDFUNDS"] <= 4.795)).astype(int)
    return out


def precision_on_positive_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Precision = TP / (TP + FP) on rows where the rule predicts 1."""
    positive = y_pred == 1
    positives = int(positive.sum())
    if positives == 0:
        return {"positives": 0, "tp": 0, "fp": 0, "precision": None, "precision_rounded_3dp": None}
    true_positive = int(((y_true == 1) & positive).sum())
    false_positive = positives - true_positive
    precision = Decimal(true_positive) / Decimal(positives)
    rounded = precision.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return {
        "positives": positives,
        "tp": true_positive,
        "fp": false_positive,
        "precision": float(precision),
        "precision_rounded_3dp": float(rounded),
    }


def score_new_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Precision of pred3 and pred4 on the test split."""
    test = df[df["split"] == "test"]
    rows = []
    for column in NEW_PREDICTIONS:
        stats = precision_on_positive_predictions(test[TARGET], test[column])
        stats["prediction"] = column
        rows.append(stats)
    return pd.DataFrame(rows).set_index("prediction")


def best_new_precision(scores: pd.DataFrame) -> dict:
    """Best defined test precision among pred3 and pred4."""
    defined = scores.dropna(subset=["precision"])
    if defined.empty:
        raise ValueError("Neither new rule makes a positive prediction on the test set.")
    best = defined.sort_values("precision", ascending=False).iloc[0]
    return {
        "prediction": str(best.name),
        "precision": float(best["precision"]),
        "precision_rounded_3dp": float(best["precision_rounded_3dp"]),
        "positives": int(best["positives"]),
        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
    }


def prepare_hand_rules(source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Modeling frame from 2000-01-01 onward, with split labels and pred3/pred4."""
    if source is None:
        source = pd.read_parquet(DATA_DIR / RAW_FILENAME)
    modeled = source[source["Date"] >= "2000-01-01"].copy()
    modeled = temporal_split(modeled, modeled["Date"].min(), modeled["Date"].max())
    return add_new_hand_rules(modeled)


def write_results(scores: pd.DataFrame, best: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "hand_rule_precision.csv")
    payload = {"best": best, "rules": scores.reset_index().to_dict(orient="records")}
    (RESULTS_DIR / "q2_hand_rule_answer.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    frame = prepare_hand_rules()
    scores = score_new_rules(frame)
    best = best_new_precision(scores)
    write_results(scores, best)

    predictions = frame[["Date", "Ticker", "split", TARGET, PRED3, PRED4]]
    predictions_path = DATA_DIR / "hand_rule_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    print(scores.to_string())
    print()
    print(
        "Best new-rule precision: "
        f"{best['precision_rounded_3dp']:.3f} ({best['prediction']})"
    )


if __name__ == "__main__":
    main()
