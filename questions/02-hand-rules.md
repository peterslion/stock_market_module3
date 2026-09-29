# Question 2: New hand rules on macro variables

What is the precision score for the best of the new predictions (`pred3` or `pred4`), rounded to three digits after the decimal point?

The rules are the positive branches from the 2026 depth-10 tree, not the 2025 thresholds:

- `pred3_manual_dgs10_5`: `(DGS10 <= 4) & (DGS5 <= 1)`
- `pred4_manual_dgs10_fedfunds`: `(DGS10 > 4) & (FEDFUNDS <= 4.795)`

Rows from 2000-01-01 onward are split by calendar span into train 70%, validation 15%, and test 15%. Precision is `TP / (TP + FP)` on the test rows where the rule predicts 1.

## Answer

**0.503**

`pred3` makes no positive predictions on the test set (16 Sep 2022 through 18 Sep 2026), because `DGS5` stays above 1. `pred4` predicts positive growth on 15,921 test rows, with 8,001 true positives and 7,920 false positives. `8001 / 15921 = 0.502544`, which rounds to **0.503**.
