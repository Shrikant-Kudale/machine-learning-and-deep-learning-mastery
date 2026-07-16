# Day 16 - Modern Optimizers: AdamW, Lion, Cosine Annealing

## Update rules
```
SGD+momentum : m = b m + g ;                 w -= lr m
Adam         : m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2
               m^ = m/(1-b1^t) ; v^ = v/(1-b2^t) ; w -= lr m^ / (sqrt(v^) + eps)
AdamW        : same, plus  w -= lr * wd * w   (decoupled from the adaptive scaling)
Lion         : w -= lr * sign(b1 m + (1-b1) g) ; m = b2 m + (1-b2) g      (no v buffer)
```
**Why AdamW?** L2 regularization added to the gradient gets divided by sqrt(v), so decay strength varies per-parameter; decoupling gives uniform decay and better generalization.
**Bias correction:** m, v start at 0, so early estimates are biased toward 0; dividing by (1 - b^t) fixes it.
**Lion:** sign update gives uniform magnitude -> use ~3-10x smaller lr and larger weight decay than AdamW; saves 1 optimizer state (memory).

## Schedules
`lr_t = lr_min + 0.5 (lr_max - lr_min)(1 + cos(pi t/T))` after linear warmup. Warmup protects Adam while v is poorly estimated and early gradients are huge.
```
lr |    /\__
   |   /    '--.__
   |  /            '--.__
   +--warmup--cosine decay--> step
```
## Trade-offs
| Optimizer | State / param | Strength | Weakness |
|---|---|---|---|
| SGD+mom | 1 | best generalization on CNNs | needs tuning |
| AdamW | 2 | robust default, transformers | 2x memory |
| Lion | 1 | memory-light, strong on large models | lr sensitivity, batch-size dependent |
| Adafactor / 8-bit Adam | <1 | huge models | complexity |

## Interview Questions
1. Adam vs AdamW? *(Decoupled decay.)*
2. Why warmup? *(Unreliable second moments and large initial updates.)*
3. Why do we exclude bias/LayerNorm from weight decay? *(They are not scale-prone weights; decay hurts.)*
4. Large-batch training rules? *(Linear LR scaling + warmup, LARS/LAMB.)*
5. Why can Adam generalize worse than SGD? *(Sharper minima, different implicit bias.)*
6. How do you resume schedules? *(Save optimizer state and step counter in checkpoint.)*
