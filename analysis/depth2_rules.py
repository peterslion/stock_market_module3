"""Hand rules read from the top two levels of clf_10.

The tree is DecisionTreeClassifier(max_depth=10, random_state=42), fit on
train plus validation after infinities and missing values are set to 0.
plot_tree(..., max_depth=2) stops at depth 2. A branch is kept when the
majority class at that node is the positive label.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from analysis.hand_rules import RESULTS_DIR, TARGET, precision_on_positive_predictions
from analysis.pred5_clf10 import feature_matrix, prepare_modeling_frame


def positive_branches(clf: DecisionTreeClassifier, feature_names: list[str], stop_depth: int = 2) -> list[dict]:
    """Paths that plot_tree would paint as the positive class at stop_depth."""
    tree = clf.tree_
    found: list[dict] = []

    def walk(node: int, depth: int, conditions: list[dict]) -> None:
        is_leaf = tree.children_left[node] == -1
        if depth == stop_depth or is_leaf:
            counts = tree.value[node][0]
            if int(np.argmax(counts)) == 1:
                found.append(
                    {
                        "conditions": conditions,
                        "node_samples": int(tree.n_node_samples[node]),
                        "negative": float(counts[0]),
                        "positive": float(counts[1]),
                    }
                )
            return
        feature = feature_names[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        walk(tree.children_left[node], depth + 1, conditions + [{"feature": feature, "op": "<=", "threshold": threshold}])
        walk(tree.children_right[node], depth + 1, conditions + [{"feature": feature, "op": ">", "threshold": threshold}])

    walk(0, 0, [])
    return found


def apply_branch(matrix: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    """1 on rows that follow every split on the branch. The matrix already uses 0 for missing values."""
    mask = np.ones(len(matrix), dtype=bool)
    values = matrix
    for condition in conditions:
        column = values[condition["feature"]].to_numpy()
        threshold = condition["threshold"]
        if condition["op"] == "<=":
            mask &= column <= threshold
        else:
            mask &= column > threshold
    return pd.Series(mask.astype(int), index=matrix.index)


def rule_text(conditions: list[dict]) -> str:
    parts = []
    for condition in conditions:
        parts.append(f"{condition['feature']} {condition['op']} {condition['threshold']:.6g}")
    return " and ".join(parts)


def score_branches(frame: pd.DataFrame, matrix: pd.DataFrame, branches: list[dict]) -> list[dict]:
    test_rows = frame["split"].eq("test").to_numpy()
    y_test = frame.loc[frame["split"].eq("test"), TARGET].astype(int).reset_index(drop=True)
    scored = []
    for index, branch in enumerate(branches, start=1):
        prediction = apply_branch(matrix, branch["conditions"])
        stats = precision_on_positive_predictions(y_test, prediction.loc[test_rows].reset_index(drop=True))
        stats["prediction"] = f"pred_depth2_branch_{index}"
        stats["rule"] = rule_text(branch["conditions"])
        stats["conditions"] = branch["conditions"]
        stats["node_samples"] = branch["node_samples"]
        scored.append(stats)
    return scored


def main() -> None:
    frame = prepare_modeling_frame()
    matrix = feature_matrix(frame)
    y = frame[TARGET].astype(int).reset_index(drop=True)
    train_rows = frame["split"].isin(["train", "validation"]).to_numpy()
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(matrix.loc[train_rows], y.loc[train_rows])
    branches = positive_branches(clf, list(matrix.columns))
    scored = score_branches(frame, matrix, branches)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "q2_depth2_tree_rules.json").write_text(json.dumps(scored, indent=2) + "\n")
    (RESULTS_DIR / "clf10_tree_top_levels.txt").write_text(
        export_text(clf, feature_names=list(matrix.columns), max_depth=2)
    )
    for row in scored:
        print(row["prediction"], row["precision_rounded_3dp"], "positives", row["positives"])
        print(" ", row["rule"])


if __name__ == "__main__":
    main()
