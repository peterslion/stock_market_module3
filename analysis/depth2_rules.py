"""Hand rules read from the top two levels of clf_10.

The tree is DecisionTreeClassifier(max_depth=10, random_state=42), fit on the
finite train and validation rows. plot_tree(..., max_depth=2) shows two
branches whose majority class is Positive.
"""

from __future__ import annotations

import json

from analysis.hand_rules import RESULTS_DIR, TARGET, precision_on_positive_predictions, prepare_hand_rules

# Exact thresholds from clf_10 at depth 2.
CPI_CORE_YOY = 0.05965891666710377
GROWTH_GOLD_365D = 0.8382211625576019
GROWTH_DJI_30D = 0.9582904577255249

PRED_LOW_CPI_HIGH_GOLD = "pred_low_cpi_high_gold"
PRED_HIGH_CPI_WEAK_DJI = "pred_high_cpi_weak_dji"
DEPTH2_RULES = [PRED_LOW_CPI_HIGH_GOLD, PRED_HIGH_CPI_WEAK_DJI]


def add_depth2_rules(df):
    """Positive branches of the depth-2 view of clf_10."""
    out = df.copy()
    out[PRED_LOW_CPI_HIGH_GOLD] = (
        (out["cpi_core_yoy"] <= CPI_CORE_YOY) & (out["growth_gold_365d"] > GROWTH_GOLD_365D)
    ).astype(int)
    out[PRED_HIGH_CPI_WEAK_DJI] = (
        (out["cpi_core_yoy"] > CPI_CORE_YOY) & (out["growth_dji_30d"] <= GROWTH_DJI_30D)
    ).astype(int)
    return out


def score_depth2_rules(df):
    test = df[df["split"] == "test"]
    rows = []
    for column in DEPTH2_RULES:
        stats = precision_on_positive_predictions(test[TARGET], test[column])
        stats["prediction"] = column
        rows.append(stats)
    return rows


def main() -> None:
    frame = add_depth2_rules(prepare_hand_rules())
    scores = score_depth2_rules(frame)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "q2_depth2_tree_rules.json").write_text(json.dumps(scores, indent=2) + "\n")
    for row in scores:
        print(row["prediction"], row["precision_rounded_3dp"], "positives", row["positives"])


if __name__ == "__main__":
    main()
