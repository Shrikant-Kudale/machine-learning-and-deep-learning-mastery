# Day 01 - Linear Regression: Cost Functions & Gradient Descent

## Model
`y_hat = X theta`, with X including a bias column. Cost: `J(theta) = (1/n) ||X theta - y||^2`.

## Derivation
```
J = (1/n)(X theta - y)^T (X theta - y)
dJ/dtheta = (2/n) X^T (X theta - y)
Set to 0  ->  X^T X theta = X^T y  ->  theta* = (X^T X)^{-1} X^T y
Hessian = (2/n) X^T X  (PSD)  ->  J is convex; unique minimum iff X has full column rank.
```
**GD update:** `theta <- theta - lr * (2/n) X^T (X theta - y)`.
Convergence for lr < 2/L where L = lambda_max(Hessian); rate depends on condition number kappa = lambda_max/lambda_min, which is why we **standardize features**.

**MLE view:** with Gaussian noise `y ~ N(X theta, sigma^2)`, maximizing likelihood == minimizing squared error.

## Pipeline
```
raw X -> StandardScaler (fit on train only) -> [Normal Eq | Batch GD | Mini-batch SGD] -> MSE on holdout
```

## Trade-offs
| Method | Cost | Pros | Cons |
|---|---|---|---|
| Normal equation | O(n d^2 + d^3) | exact, no lr | bad for large d, ill-conditioned X^T X |
| Batch GD | O(n d) / step | scales, simple | needs lr tuning, scaling |
| Mini-batch SGD | O(b d) / step | streaming, GPU friendly | noisy, needs lr schedule |

## Interview Questions
1. Why does feature scaling speed up GD? *(Reduces condition number; contours become round.)*
2. When is X^T X not invertible and what do you do? *(Collinearity or d > n; use ridge, pseudo-inverse, or drop features.)*
3. Why prefer `lstsq`/QR/SVD over explicit inverse? *(Squaring the condition number in X^T X loses precision.)*
4. Derive the gradient of MSE. *(See above.)*
5. MSE vs MAE vs Huber? *(MSE = mean, outlier sensitive; MAE = median; Huber blends.)*
6. Is linear regression with polynomial features still "linear"? *(Yes, linear in parameters.)*
