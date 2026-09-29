"""Selection of the decision-tree depth with the highest test precision."""

import pandas as pd

from analysis.tune_tree_depth import best_max_depth


def test_best_depth_is_the_smallest_depth_tied_for_the_highest_test_precision():
    scores = pd.DataFrame(
        {
            "max_depth": [1, 2, 3, 4],
            "precision_test": [0.50, 0.58, 0.61, 0.61],
        }
    )
    assert best_max_depth(scores) == 3
