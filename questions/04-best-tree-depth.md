# Question 4: Hyperparameter tuning for a decision tree

What is the optimal tree depth, from 1 to 20, for a `DecisionTreeClassifier`?

Each depth is trained as `DecisionTreeClassifier(max_depth=depth, random_state=42)` on the finite train plus validation rows. Rows with a missing or infinite numerical feature are removed first. Percentile outliers are kept. The optimal depth is the one with the highest precision on the remaining test rows. That model is saved as `pred6_clf_best`.

## Answer

**5**

Test precision at depth 5 is **0.648**. That is the highest of the twenty depths, and it is above 0.58. Validation precision keeps rising through depth 20.

Test precision on the finite test rows:

| Prediction | Test precision |
|---|---:|
| pred0_manual_cci | 0.560 |
| pred1_manual_prev_g1 | 0.579 |
| pred2_manual_prev_g1_and_snp | 0.571 |
| pred3_manual_dgs10_5 | no positive test predictions |
| pred4_manual_dgs10_fedfunds | 0.503 |
| pred5_clf_10 | 0.582 |
| pred6_clf_best | 0.648 |
