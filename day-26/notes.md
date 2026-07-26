# Day 26 - Multi-Head Attention & the Transformer Block

## Multi-head attention
`head_i = Attn(X W_Q^i, X W_K^i, X W_V^i)`, `MHA = Concat(head_1..head_h) W_O`, with `d_k = d_model / h`. Same FLOPs as single-head with full width, but h independent attention patterns (syntax, position, copy heads...). Implementation: one fused QKV matmul, reshape to (B, h, T, d_k), batched matmuls.

## Block (pre-LN, GPT-style)
```
x --> LN --> MHA(causal) --(+)--> LN --> MLP(d -> 4d -> d, GELU) --(+)--> out
|                          ^ |                                      ^
'--------- residual -------' '------------- residual ---------------'
```
Parameters per block `~ 12 d^2` (4d^2 attention + 8d^2 MLP). FLOPs/token `~ 24 d^2 + 4 T d` forward... attention term grows with context.
**Modern upgrades:** RMSNorm, SwiGLU MLP, RoPE, GQA/MQA, FlashAttention, KV cache, weight tying.

## Architecture families
| | Encoder-only (BERT) | Decoder-only (GPT) | Enc-Dec (T5) |
|---|---|---|---|
| Mask | bidirectional | causal | both + cross-attn |
| Use | embeddings, classification | generation, LLMs | translation, summarization |

## Inference economics
KV cache per token = `2 * layers * kv_heads * d_head * bytes`. Decode is **memory-bandwidth bound**; prefill is compute bound. Batching + quantization + speculative decoding raise throughput.

## Interview Questions
1. Why multiple heads? *(Different subspace relations at the same cost.)*
2. Parameter count of a transformer block? *(~12 d^2.)*
3. Pre-LN vs post-LN? *(Pre-LN stable gradients, no warmup fragility; post-LN slightly better final quality when it trains.)*
4. How does the KV cache change complexity? *(Per-step cost O(T d) instead of O(T^2 d) recomputation.)*
5. GQA/MQA? *(Share K/V across head groups -> smaller cache.)*
6. Why does the causal LM need only next-token loss? *(Chain rule factorization of sequence probability.)*
