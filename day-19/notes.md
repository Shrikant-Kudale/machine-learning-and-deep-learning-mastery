# Day 19 - Exploding & Vanishing Gradients

## Why it happens
`dL/dh_1 = dL/dh_L * prod_{l=1}^{L-1} J_l`, `J_l = diag(f'(z_l)) W_l`. If the spectral norm of the typical Jacobian is `< 1`, the product shrinks exponentially; if `> 1`, it explodes. Sigmoid (f' <= 0.25) and tanh saturation vanish; large weights or RNN recurrence (`W_h^T` powers) explode.

## Diagnostics
Per-layer gradient norm histograms, update-to-weight ratio (~1e-3 healthy), loss -> NaN/Inf, activations saturating.

## Fixes
| Technique | Mechanism |
|---|---|
| He/Xavier init | keep Var(J) ~ 1 (Day 18) |
| ReLU-family activations | f' = 1 on active units |
| Residual connections | `dh_{l+1}/dh_l = I + dF/dh_l`, gradient highway |
| Normalization (BN/LN/RMSNorm) | re-centres and re-scales activations each layer |
| Gradient clipping | `g <- g * min(1, c/||g||)`, preserves direction (exploding) |
| LSTM/GRU gates | additive cell-state path |
| Smaller LR / warmup | limits damage from large steps |

```
plain:     x -> F1 -> F2 -> ... -> FL        grad = prod J_l          (exponential)
residual:  x -> (+F1) -> (+F2) -> ...        grad = prod (I + J_l)    (identity path survives)
```
## Interview Questions
1. How would you detect vanishing gradients in a running job? *(Log per-layer grad norms, TensorBoard histograms.)*
2. Clip by value vs by norm? *(Norm keeps direction; value distorts it.)*
3. Why do residuals help? *(Identity term in Jacobian.)*
4. Why do RNNs suffer more? *(Same W multiplied T times; eigenvalue > or < 1.)*
5. Loss spikes in LLM training - causes/mitigations? *(Bad batches, Adam epsilon/beta2, LR too high; skip batch, lower beta2, clip, z-loss.)*
6. Pre-LN vs post-LN? *(Pre-LN trains stably without warmup sensitivity.)*
