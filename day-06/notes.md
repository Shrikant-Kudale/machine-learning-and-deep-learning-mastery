# Day 06 - Support Vector Machines & Kernels

## Hard / soft margin
```
min_{w,b,xi}  (1/2)||w||^2 + C sum xi_i      s.t.  y_i (w.x_i + b) >= 1 - xi_i,  xi_i >= 0
Equivalent:   (lambda/2)||w||^2 + (1/n) sum max(0, 1 - y_i f(x_i))       (hinge loss)
```
Margin width = 2/||w||, so minimizing ||w|| maximizes the margin.

## Dual (derivation sketch)
Lagrangian stationarity: `w = sum a_i y_i x_i`, `sum a_i y_i = 0`, `0 <= a_i <= C`.
`max_a  sum a_i - (1/2) sum_ij a_i a_j y_i y_j K(x_i, x_j)`.
Only points with a_i > 0 (support vectors) matter. Predictions use kernels only: `f(x) = sum a_i y_i K(x_i, x) + b`.

## Kernels (Mercer: Gram matrix PSD)
`linear x.z`, `poly (x.z + c)^d`, `RBF exp(-gamma||x-z||^2)` (infinite-dim feature space).
gamma large -> wiggly boundary (overfit); C large -> less regularization.

## Architecture
```
X -> standardize -> Gram matrix K (n x n) -> solver (SMO / Pegasos) -> support vectors -> sign(f(x))
```
## Trade-offs
| | Linear SVM | Kernel SVM |
|---|---|---|
| Train | ~O(nd) (liblinear) | O(n^2)-O(n^3) |
| Predict | O(d) | O(n_sv d) |
| Use when | high-d sparse text | n <= ~50k, nonlinear |

## Interview Questions
1. Why is it called "support vector"? *(Only margin-violating/boundary points have nonzero alpha.)*
2. Kernel trick intuition? *(Inner products in feature space without computing features.)*
3. Effect of C and gamma? *(See above.)*
4. SVM vs logistic regression? *(Hinge vs log loss; SVM has no calibrated probabilities, sparse solution.)*
5. Why scale features? *(Distance-based kernels.)*
6. How to get probabilities from an SVM? *(Platt scaling / isotonic; Day 10.)*
