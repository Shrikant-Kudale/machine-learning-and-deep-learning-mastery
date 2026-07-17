# Day 17 - Activation Functions

## Catalogue
| f(x) | f'(x) | Notes |
|---|---|---|
| sigmoid `1/(1+e^-x)` | `s(1-s)` <= 0.25 | saturates, not zero-centered; output layer for probabilities |
| tanh | `1 - tanh^2` <= 1 | zero-centered, still saturates |
| ReLU `max(0,x)` | 0 or 1 | cheap, sparse; **dying ReLU** |
| LeakyReLU / PReLU | 1 or a | avoids dead units |
| GELU `x Phi(x)` | `Phi(x) + x phi(x)` | smooth; default in BERT/GPT |
| SiLU/Swish `x s(x)` | `s(1 + x(1-s))` | LLaMA (SwiGLU gating variant) |
| Mish `x tanh(softplus(x))` | smooth | vision detectors |

**Why vanishing gradients with sigmoid:** product of L factors each <= 0.25 gives <= 0.25^L.
**Why nonlinearity:** composition of affine maps is affine; universal approximation needs nonlinearity.
**SwiGLU:** `(xW) * silu(xV)` - gated linear unit, better quality per parameter in transformers.

```
 sigmoid: saturates both ends     ReLU: flat zero for x<0      GELU: smooth, small negative dip
   ___----                          /                              _.-
 _/                               _/                            _.'
```
## Trade-offs
| Choice | Use |
|---|---|
| ReLU | CNNs, MLPs, speed/edge (cheap, quantization friendly) |
| GELU / SiLU | Transformers |
| tanh | RNN/LSTM gates & state |
| sigmoid / softmax | output probabilities / gates |
| ReLU6 / hard-swish | mobile & INT8 (bounded range) |

## Interview Questions
1. Dying ReLU: cause and fix? *(Large negative bias/updates; use LeakyReLU, lower LR, good init, BatchNorm.)*
2. Why GELU over ReLU in transformers? *(Smoothness, better empirical optimization.)*
3. Why is tanh preferred over sigmoid in hidden layers? *(Zero-centered.)*
4. Can you use ReLU in the output layer? *(Only for non-negative regression targets.)*
5. Derivative of softmax? *(`diag(p) - p p^T`.)*
6. Why do bounded activations help quantization? *(Fixed dynamic range -> better INT8 scales.)*
