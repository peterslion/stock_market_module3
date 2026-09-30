"""Unique correct test predictions of the depth-10 tree."""

import pandas as pd

from analysis.pred5_clf10 import (
    HAND_RULES,
    PRED5,
    TARGET,
    build_predictions,
    count_unique_correct_on_test,
    only_pred5_is_correct,
)


def test_only_pred5_requires_every_hand_rule_to_be_wrong():
    frame = pd.DataFrame(
        {
            TARGET: [1, 1, 0],
            PRED5: [1, 1, 0],
            HAND_RULES[0]: [0, 1, 0],
            HAND_RULES[1]: [0, 0, 1],
            HAND_RULES[2]: [0, 0, 1],
            HAND_RULES[3]: [0, 0, 1],
            HAND_RULES[4]: [0, 0, 1],
        }
    )
    flags = only_pred5_is_correct(frame)
    assert flags.tolist() == [1, 0, 0]


def test_test_set_unique_correct_count_is_2622():
    frame = build_predictions()
    assert count_unique_correct_on_test(frame) == 2622
