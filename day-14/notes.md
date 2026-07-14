# Day 14 - Neural Network Backpropagation

## Derivation (2-layer net, softmax CE)
```
z1 = X W1 + b1      a1 = tanh(z1)      z2 = a1 W2 + b2      P = softmax(z2)      L = -(1/n) sum log P[i, y_i]

dL/dz2 = (P - Y)/n                         (softmax + CE collapse)
dL/dW2 = a1^T dz2          dL/db2 = sum_rows dz2
dL/da1 = dz2 W2^T
dL/dz1 = dL/da1 * (1 - a1^2)               (elementwise, tanh' = 1 - tanh^2)
dL/dW1 = X^T dz1           dL/db1 = sum_rows dz1
```
General rule: `delta_l = (W_{l+1}^T delta_{l+1}) * f'(z_l)`; gradient of weights = `a_{l-1}^T delta_l`. Reverse-mode autodiff = chain rule evaluated from the output with cached forward activations: cost ~ 2-3x forward, memory O(depth) activations.

```
x --> [W1,b1] --> tanh --> [W2,b2] --> softmax --> loss
 ^dW1            ^dz1        ^dW2         ^dz2 = P - Y
 '------------ gradients flow right to left ------------'
```
## Gradient checking
Central difference `(L(w+e) - L(w-e))/2e`, e = 1e-5, in float64; relative error < 1e-7 means correct. Turn off dropout/BN-training during checks.

## Trade-offs
| | Manual backprop | Autograd |
|---|---|---|
| Speed to write | slow | instant |
| Memory control | full | checkpointing available |
| Bug risk | high | low |

## Interview Questions
1. Why reverse-mode instead of forward-mode? *(Many parameters, one scalar loss: one backward pass gives all gradients.)*
2. Why softmax+CE gradient is P - Y? *(Derivation above.)*
3. Memory cost of backprop? *(Stores activations; use gradient checkpointing to trade compute for memory.)*
4. Why is a network with no nonlinearity just linear regression? *(Product of matrices is a matrix.)*
5. How do you debug a net that won't learn? *(Overfit one batch, gradient check, check LR/init/data labels.)*
6. What is the Jacobian-vector product in autograd? *(Each backward op multiplies upstream gradient by the local Jacobian without materializing it.)*
