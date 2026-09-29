# Question 4: Hyperparameter tuning for a decision tree

What is the optimal tree depth, from 1 to 20, for a `DecisionTreeClassifier`?

Each depth is trained as `DecisionTreeClassifier(max_depth=depth, random_state=42)` on train plus validation. The optimal depth is the one with the highest precision on the test set. That model is saved as `pred6_clf_best`.

## Answer

**4**

Test precision at depth 4 is **0.629**. That is the highest of the twenty depths, and it is above 0.58. Validation precision keeps rising through depth 20, while test precision peaks at 4 and then declines.

Test precision of the earlier predictions:

| Prediction | Test precision |
|---|---:|
| pred0_manual_cci | 0.565 |
| pred1_manual_prev_g1 | 0.579 |
| pred2_manual_prev_g1_and_snp | 0.573 |
| pred3_manual_dgs10_5 | no positive test predictions |
| pred4_manual_dgs10_fedfunds | 0.503 |
| pred5_clf_10 | 0.591 |
| pred6_clf_best | 0.629 |
