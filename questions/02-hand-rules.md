# Question 2: New hand rules on macro variables

What is the precision score for the best of the new predictions (`pred3` or `pred4`), rounded to three digits after the decimal point?

The rules are the positive branches from the 2026 depth-10 tree, not the 2025 thresholds:

- `pred3_manual_dgs10_5`: `(DGS10 <= 4) & (DGS5 <= 1)`
- `pred4_manual_dgs10_fedfunds`: `(DGS10 > 4) & (FEDFUNDS <= 4.795)`

Rows from 2000-01-01 onward are split by calendar span into train 70%, validation 15%, and test 15%. Precision is `TP / (TP + FP)` on the test rows where the rule predicts 1.

## Answer

**0.503**

`pred3` makes no positive predictions on the test set (16 Sep 2022 through 18 Sep 2026), because `DGS5` stays above 1. `pred4` predicts positive growth on 15,921 test rows, with 8,001 true positives and 7,920 false positives. `8001 / 15921 = 0.502544`, which rounds to **0.503**.

## Rules from the visualized depth-10 tree

`plot_tree(clf_10, max_depth=2)` shows two branches whose majority class is Positive. The thresholds are the exact splits from `DecisionTreeClassifier(max_depth=10, random_state=42)`.

- `pred_low_cpi_high_gold`: `cpi_core_yoy <= 0.059659` and `growth_gold_365d > 0.838221`
- `pred_high_cpi_weak_dji`: `cpi_core_yoy > 0.059659` and `growth_dji_30d <= 0.958290`

On the same test set, precision is **0.563** for the gold branch (31,047 positive predictions) and **0.808** for the Dow branch (718 positive predictions). The better of these two visualized rules is **0.808**.
