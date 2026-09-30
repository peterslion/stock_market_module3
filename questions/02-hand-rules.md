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

`plot_tree(clf_10, max_depth=2)` is read from `DecisionTreeClassifier(max_depth=10, random_state=42)` after infinities and missing values are set to 0. Three nodes at depth 2 have a positive majority class:

- `pred_depth2_branch_1`: `cpi_core_yoy <= 0.0227954` and `DGS10 <= 1.745`
- `pred_depth2_branch_2`: `cpi_core_yoy <= 0.0227954` and `DGS10 > 1.745`
- `pred_depth2_branch_3`: `cpi_core_yoy > 0.0227954` and `FEDFUNDS > 4.965`

The remaining depth-2 node, high core CPI and `FEDFUNDS <= 4.965`, has a negative majority class, so it is not a positive rule.

On the test set, core CPI stays above 0.0227954, so the first two rules make no positive predictions. `pred_depth2_branch_3` predicts positive growth on 11,760 test rows, with 7,321 true positives and 4,439 false positives. `7321 / 11760 = 0.622534`, which rounds to **0.623**. That is the precision of the best positive branch from the depth-2 view.
