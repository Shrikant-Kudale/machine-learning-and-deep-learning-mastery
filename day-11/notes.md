# Day 11 - Bias-Variance Decomposition

## Proof (squared loss)
Let `y = f(x) + e`, `E[e] = 0`, `Var(e) = s^2`, and `f_hat` trained on random dataset D. Write `f_bar(x) = E_D[f_hat(x)]`.
```
E[(y - f_hat)^2] = E[(f + e - f_hat)^2]
 = E[e^2] + E[(f - f_hat)^2]                      (cross term 0 since e independent, mean 0)
 E[(f - f_hat)^2] = E[(f - f_bar + f_bar - f_hat)^2]
                  = (f - f_bar)^2 + E[(f_hat - f_bar)^2] + 2(f - f_bar) E[f_bar - f_hat]   (last term = 0)
=> Error = Bias^2 + Variance + Irreducible noise s^2
```
```
error
  |\                     total
  | \         ___---'''
  |  \  __--''   variance ->
  |   \'
  |   bias^2 ->  \___
  +-------------------- model complexity
```
## Levers
| To reduce | Do |
|---|---|
| Bias | more capacity, better features, less regularization, boosting |
| Variance | more data, regularization, bagging, early stopping, simpler model |
| Noise | cannot; better labels / features |

Modern caveat: **double descent** - past the interpolation threshold test error can fall again in over-parameterized nets.

## Interview Questions
1. Derive the decomposition. *(Above.)*
2. Does bagging reduce bias or variance? Boosting? *(Variance; bias.)*
3. How would you diagnose high bias vs high variance in practice? *(Learning curves: train/val gap vs both high.)*
4. kNN: effect of k? *(Small k -> low bias/high var.)*
5. Does the decomposition hold for 0-1 loss? *(Not additively; see Domingos.)*
6. What is double descent? *(Non-monotone risk curve around interpolation.)*
