# Day 25 - Self-Attention From Scratch

## Mechanism
`Q = XW_Q, K = XW_K, V = XW_V`, `Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`. Each output is a data-dependent weighted average of values.
**Why 1/sqrt(d_k):** for unit-variance q,k: `Var(q.k) = d_k`; unscaled logits grow with d_k, saturating softmax (tiny gradients).
**Complexity:** time O(T^2 d), memory O(T^2) for the score matrix; FlashAttention fuses/tiles to avoid materializing it (same math, IO-aware).

```
X (T x d) --WQ--> Q \
          --WK--> K  >  scores (T x T) --/sqrt(dk)--> [mask] --softmax--> W (T x T) --@ V--> out (T x dk)
          --WV--> V /
causal mask (lower-triangular):     1 0 0 0
                                    1 1 0 0
                                    1 1 1 0
                                    1 1 1 1
```
**Properties:** permutation-equivariant (so add positional information: sinusoidal, learned, RoPE, ALiBi), global receptive field in one layer, no recurrence -> parallel training.
**Cross-attention:** Q from decoder, K/V from encoder.

## Trade-offs
| Variant | Idea | Benefit |
|---|---|---|
| Multi-head (Day 26) | h subspaces | diverse patterns |
| MQA / GQA | share K/V heads | smaller KV cache, faster decode |
| Sliding window / sparse | local + global tokens | O(T w) |
| Linear attention | kernel feature maps | O(T) but quality gap |
| FlashAttention | tiling in SRAM | 2-4x faster, exact |

## Interview Questions
1. Why divide by sqrt(d_k)? *(Variance argument.)*
2. Why is attention O(T^2) and how do you mitigate? *(Pairwise scores; windowing, flash, linear/state-space.)*
3. How does a causal mask work numerically? *(-inf before softmax -> zero weight.)*
4. Why do we need positional encodings? *(Permutation equivariance.)*
5. What is the KV cache? *(Reuse past keys/values during autoregressive decoding; memory grows with T.)*
6. Self- vs cross-attention? *(Same sequence vs two sequences.)*
