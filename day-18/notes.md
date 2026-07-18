# Day 18 - Weight Initialization (Xavier, He)

## Variance derivation
For `y = sum_{i=1}^{n} w_i x_i` with independent zero-mean w and x:
`Var(y) = n Var(w) Var(x)`. To keep `Var(y) = Var(x)` in the forward pass: `Var(w) = 1/n_in`.
Backward pass wants `Var(w) = 1/n_out`. **Xavier/Glorot** compromises: `Var(w) = 2/(n_in + n_out)` (assumes linear/tanh near 0).
**ReLU** zeroes half the pre-activations: `E[relu(y)^2] = Var(y)/2`, so **He**: `Var(w) = 2/n_in`.
Uniform variants: `U(-a, a)` with `a = sqrt(3 Var)`.

```
depth ->   1    10    50
naive      .5   1e-3  1e-30     vanishing
xavier/relu  1   .3   1e-5      slow shrink (factor 1/sqrt2 per layer)
he/relu      1    1    1        stable
```
## Practical rules
| Activation | Init |
|---|---|
| ReLU / LeakyReLU | He (kaiming) |
| tanh / sigmoid | Xavier |
| SELU | LeCun normal |
| Transformers | N(0, 0.02), scale residual-branch outputs by 1/sqrt(2L) |
| Zero-init | last layer of residual branch (ReZero / "zero-gamma" BN) |

Never init all weights to the same constant: symmetry means all units in a layer receive identical gradients. Biases can be zero.

## Interview Questions
1. Derive He init. *(Above.)*
2. Why not initialize with zeros? *(Symmetry breaking.)*
3. Does BatchNorm remove the need for good init? *(Reduces sensitivity, not fully eliminates, especially first iterations.)*
4. Why scale residual branches in deep transformers? *(Variance grows with depth: sum of L branches.)*
5. Fan-in vs fan-out mode? *(Preserve forward vs backward variance.)*
6. Initialization for LSTMs? *(Orthogonal recurrent weights, forget-gate bias = 1.)*
