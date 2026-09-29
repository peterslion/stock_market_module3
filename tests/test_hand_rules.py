"""Precision of the new DGS10 hand rules on the temporal test split."""

import pandas as pd

from analysis.hand_rules import (
    PRED3,
    PRED4,
    best_new_precision,
    prepare_hand_rules,
    score_new_rules,
)


def test_pred4_is_the_only_rule_with_test_positives_and_precision_is_0_503():
    frame = prepare_hand_rules()
    scores = score_new_rules(frame)
    best = best_new_precision(scores)

    assert scores.loc[PRED3, "positives"] == 0
    assert pd.isna(scores.loc[PRED3, "precision"])
    assert scores.loc[PRED4, "positives"] > 0
    assert best["prediction"] == PRED4
    assert best["precision_rounded_3dp"] == 0.503
