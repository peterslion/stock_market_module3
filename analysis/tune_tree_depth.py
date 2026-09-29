"""Choose the decision-tree depth from 1 to 20 with the highest test precision.

Each tree is ``DecisionTreeClassifier(max_depth=depth, random_state=42)``,
fit on train plus validation. The winning depth is stored as ``pred6_clf_best``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

from analysis.pred5_clf10 import (
    HAND_RULES,
    PRED5,
    TARGET,
    feature_matrix,
    fit_pred5,
    prepare_modeling_frame,
)
from analysis.hand_rules import RESULTS_DIR

PRED6 = "pred6_clf_best"
DEPTHS = range(1, 21)


def binary_precision(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Precision of the positive class. Undefined when nothing is predicted positive."""
    score = precision_score(y_true, y_pred, zero_division=np.nan)
    if pd.isna(score):
        return float("nan")
    return float(score)


def depth_precision_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Test and validation precision for every depth from 1 to 20."""
    matrix = feature_matrix(frame)
    y = frame[TARGET].astype(int).reset_index(drop=True)
    train_rows = frame["split"].isin(["train", "validation"]).to_numpy()
    validation_rows = frame["split"].eq("validation").to_numpy()
    test_rows = frame["split"].eq("test").to_numpy()
    rows = []
    for depth in DEPTHS:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(matrix.loc[train_rows], y.loc[train_rows])
        rows.append(
            {
                "max_depth": depth,
                "precision_test": binary_precision(y.loc[test_rows], clf.predict(matrix.loc[test_rows])),
                "precision_validation": binary_precision(
                    y.loc[validation_rows], clf.predict(matrix.loc[validation_rows])
                ),
            }
        )
        print(
            f"depth {depth:2d}  test {rows[-1]['precision_test']:.6f}  "
            f"validation {rows[-1]['precision_validation']:.6f}"
        )
    return pd.DataFrame(rows)


def best_max_depth(scores: pd.DataFrame) -> int:
    """Smallest depth among those tied for the highest test precision."""
    best_score = scores["precision_test"].max()
    winners = scores.loc[scores["precision_test"] == best_score, "max_depth"]
    return int(winners.min())


def add_pred6(frame: pd.DataFrame, depth: int) -> tuple[pd.DataFrame, str]:
    """Refit the chosen depth on train+validation and predict every row."""
    matrix = feature_matrix(frame)
    y = frame[TARGET].astype(int).reset_index(drop=True)
    train_rows = frame["split"].isin(["train", "validation"]).to_numpy()
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(matrix.loc[train_rows], y.loc[train_rows])
    out = frame.copy()
    out[PRED6] = clf.predict(matrix)
    rules = export_text(clf, feature_names=list(matrix.columns), max_depth=3)
    return out, rules


def compare_precision(frame: pd.DataFrame) -> pd.DataFrame:
    """Test precision of pred0-pred6. pred5 is fit here if it is not already present."""
    if PRED5 not in frame.columns:
        frame = frame.copy()
        frame[PRED5] = fit_pred5(frame)
    test = frame["split"].eq("test")
    y_test = frame.loc[test, TARGET].astype(int)
    rows = []
    for column in HAND_RULES + [PRED5, PRED6]:
        rows.append(
            {
                "prediction": column,
                "precision_test": binary_precision(y_test, frame.loc[test, column].astype(int)),
            }
        )
    return pd.DataFrame(rows)


def write_results(scores: pd.DataFrame, depth: int, comparison: pd.DataFrame, tree_rules: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "tree_depth_precision.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "prediction_precision_comparison.csv", index=False)
    (RESULTS_DIR / "pred6_tree_top_levels.txt").write_text(tree_rules)
    best_row = scores.loc[scores["max_depth"] == depth].iloc[0]
    payload = {
        "best_max_depth": depth,
        "precision_test": float(best_row["precision_test"]),
        "precision_validation": float(best_row["precision_validation"]),
        "prediction": PRED6,
        "random_state": 42,
        "trained_on": ["train", "validation"],
    }
    (RESULTS_DIR / "q4_best_depth_answer.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    frame = prepare_modeling_frame()
    scores = depth_precision_table(frame)
    depth = best_max_depth(scores)
    frame, tree_rules = add_pred6(frame, depth)
    comparison = compare_precision(frame)
    write_results(scores, depth, comparison, tree_rules)
    print()
    print(scores.to_string(index=False))
    print()
    print(comparison.to_string(index=False))
    print()
    print(f"best_max_depth = {depth}")


if __name__ == "__main__":
    main()
