# Day 02 - Logistic Regression, Cross-Entropy, Newton/IRLS, Softmax

## Model & loss
`p = sigma(w^T x)`, `sigma(z) = 1/(1+e^-z)`, `sigma' = sigma(1-sigma)`.
Bernoulli log-likelihood -> BCE: `J = -(1/n) sum[ y log p + (1-y) log(1-p) ]`.

## Gradient derivation
```
dJ/dp = -(y/p) + (1-y)/(1-p) = (p - y) / (p(1-p))
dp/dz = p(1-p)
dz/dw = x
=> dJ/dw = (1/n) X^T (p - y)         (same form as linear regression!)
Hessian H = (1/n) X^T S X,  S = diag(p_i (1-p_i))   -> PSD, J convex
Newton: w <- w - H^{-1} g   (== IRLS, weighted least squares each step)
```
Softmax: `p_k = e^{z_k} / sum_j e^{z_j}`, gradient `X^T (P - Y)`. Subtract max for stability.

## Pipeline
```
X -> scale -> z = Xw -> sigmoid/softmax -> BCE/CE -> {GD | Newton}
```

## Trade-offs
| Optimizer | Iterations | Per-iter cost | Note |
|---|---|---|---|
| GD | hundreds | O(nd) | needs lr |
| Newton/IRLS | ~5-15 | O(nd^2 + d^3) | quadratic convergence, d must be small |
| L-BFGS | tens | O(nd) | sklearn default |

## Interview Questions
1. Why cross-entropy instead of MSE for classification? *(MSE + sigmoid is non-convex and gradients vanish when confidently wrong.)*
2. What happens on perfectly separable data? *(Weights diverge; MLE does not exist; regularize.)*
3. Why is the decision boundary linear? *(p = 0.5 <=> w^T x = 0.)*
4. Interpret a coefficient. *(Change in log-odds per unit; exp(w) is the odds ratio.)*
5. Derive the gradient. *(Above.)*
6. Class imbalance handling? *(Class weights, threshold tuning, PR-AUC; see Day 09.)*
