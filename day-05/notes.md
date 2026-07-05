# Day 05 - Gradient Boosted Trees, XGBoost, LightGBM

## Functional gradient descent
```
F_0 = argmin_c sum L(y_i, c)
for m = 1..M:
    r_i = -dL(y_i, F)/dF |_{F = F_{m-1}(x_i)}      # pseudo-residuals
    h_m = fit tree to (x_i, r_i)
    F_m = F_{m-1} + eta * h_m
```
Squared loss -> r = y - F. Logistic loss -> r = y - sigma(F).

## XGBoost second-order objective
`L ~ sum[g_i f(x_i) + (1/2) h_i f(x_i)^2] + gamma T + (lambda/2) sum w_j^2`, with g, h = first/second derivatives.
Optimal leaf weight `w_j* = -G_j / (H_j + lambda)`; split gain
`1/2 [G_L^2/(H_L+l) + G_R^2/(H_R+l) - (G_L+G_R)^2/(H_L+H_R+l)] - gamma`.

## XGBoost vs LightGBM
| | XGBoost | LightGBM |
|---|---|---|
| Tree growth | level-wise (default) | leaf-wise (best-first) |
| Split finding | exact / histogram | histogram + GOSS + EFB |
| Categoricals | one-hot / experimental | native |
| Speed on big data | good | usually faster |
| Overfit risk | lower | higher on small data (tune num_leaves, min_data_in_leaf) |

## Pipeline
```
X,y -> F0 -> [grad/hess -> shallow tree -> shrink -> add] x M -> early stop on val set
```
## Interview Questions
1. Bagging vs boosting? *(Variance reduction vs bias reduction; parallel vs sequential.)*
2. Role of learning rate vs n_estimators? *(Shrinkage; lower eta needs more trees but generalizes better.)*
3. Why second-order info? *(Newton step; better leaf values and gain.)*
4. How does XGBoost handle missing values? *(Learns default direction per split.)*
5. Key hyperparameters to tune? *(eta, max_depth/num_leaves, min_child_weight, subsample, colsample, lambda.)*
6. Why are GBDTs strong on tabular data? *(Handle heterogeneous features, no scaling, capture interactions.)*
