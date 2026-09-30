# Question 3: Unique correct predictions from a 10-level decision tree

How many TEST records have `pred5_clf_10` correct while every hand rule `pred0` through `pred4` is incorrect?

The classifier is `DecisionTreeClassifier(max_depth=10, random_state=42)`, fit on train plus validation. Infinities and missing feature values are set to 0, and no rows are dropped. The feature set is the notebook's numerical columns plus dummies for `Month` (month number), `Weekday`, `Ticker`, `ticker_type`, and `month_wom`.

Hand rules:

- `pred0_manual_cci`: `cci > 200`
- `pred1_manual_prev_g1`: `growth_30d > 1`
- `pred2_manual_prev_g1_and_snp`: `growth_30d > 1` and `growth_snp500_30d > 1`
- `pred3_manual_dgs10_5`: `DGS10 <= 4` and `DGS5 <= 1`
- `pred4_manual_dgs10_fedfunds`: `DGS10 > 4` and `FEDFUNDS <= 4.795`

## Answer

**1801**
