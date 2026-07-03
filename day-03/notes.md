# Day 03 - Regularization: L1, L2, ElasticNet

## Objectives
```
Ridge:       (1/2n)||y - Xw||^2 + (a/2)||w||_2^2
Lasso:       (1/2n)||y - Xw||^2 + a||w||_1
ElasticNet:  (1/2n)||y - Xw||^2 + a[ r||w||_1 + (1-r)/2 ||w||_2^2 ]
```
## Derivations
**Ridge:** `grad = -(1/n)X^T(y-Xw) + a w = 0  ->  w = (X^T X + n a I)^{-1} X^T y`. Always invertible; shrinks along small-singular-value directions: `w_i = s_i/(s_i^2 + n a) * u_i^T y`.

**Lasso coordinate update:** with residual `r_j` excluding feature j and `rho_j = x_j^T r_j / n`:
`w_j = S(rho_j, a) / (x_j^T x_j / n)`, `S(z,g) = sign(z) max(|z|-g, 0)` (soft-threshold). The kink at 0 yields exact zeros.

**Bayesian view:** L2 = Gaussian prior; L1 = Laplace prior (MAP estimates).

**Why sparsity?** L1 ball has corners on axes; the loss ellipse typically touches a corner.

## Pipeline
```
X -> StandardScaler (mandatory: penalty is scale-dependent) -> {Ridge CF | Lasso CD | ENet CD} -> sparsity report
```
## Trade-offs
| | L2 | L1 | ElasticNet |
|---|---|---|---|
| Sparsity | no | yes | yes |
| Correlated features | spreads weight | picks one arbitrarily | groups them |
| Solver | closed form | CD / LARS | CD |
| d > n | fine | selects <= n | no limit |

## Interview Questions
1. Why standardize before regularizing? *(Penalty treats all weights equally; scale changes effective penalty.)*
2. Why doesn't L2 give exact zeros? *(Gradient of penalty vanishes at 0.)*
3. Lasso with two identical features? *(Non-unique; ElasticNet splits the weight.)*
4. Do you penalize the intercept? *(No.)*
5. How to choose alpha? *(CV along a regularization path, 1-SE rule.)*
6. Relation of L2 to weight decay in NNs? *(Equivalent for SGD; not for Adam, hence AdamW on Day 16.)*
