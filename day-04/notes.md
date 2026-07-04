# Day 04 - Decision Trees & Random Forests

## Split criteria
```
Gini(S)    = 1 - sum_k p_k^2
Entropy(S) = -sum_k p_k log p_k
Gain       = I(parent) - (|L|/n) I(L) - (|R|/n) I(R)
```
Regression trees minimize within-node variance (MSE). Greedy CART: sort each feature, sweep thresholds with running class counts: O(d n log n) per node.

## Random Forest variance argument
For B trees with variance s^2 and pairwise correlation rho:
`Var(mean) = rho s^2 + (1 - rho) s^2 / B`.
Bagging shrinks the second term; **feature subsampling (sqrt(d))** lowers rho. Out-of-bag samples (~36.8% = 1/e) give a free validation estimate.

## Architecture
```
data -> bootstrap_1 -> tree_1 (random feature subset per split) \
     -> bootstrap_2 -> tree_2                                     > average probs -> argmax
     -> bootstrap_B -> tree_B                                    /
```
## Trade-offs
| | Single tree | Random Forest |
|---|---|---|
| Variance | high | low |
| Interpretability | high | low (use importances/SHAP) |
| Extrapolation | none | none |
| Needs scaling | no | no |

## Interview Questions
1. Why do deep trees overfit and how does bagging help? *(Low bias/high variance; averaging reduces variance.)*
2. Impurity importance bias? *(Favors high-cardinality features; prefer permutation importance/SHAP.)*
3. Why can't trees extrapolate? *(Piecewise constant.)*
4. Gini vs entropy? *(Rarely matters; Gini is cheaper.)*
5. How are missing values handled? *(Surrogate splits / learned default direction in GBDT.)*
6. Explain OOB error. *(Each tree evaluated on samples it never saw.)*
