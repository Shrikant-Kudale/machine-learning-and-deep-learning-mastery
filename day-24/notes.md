# Day 24 - RNNs, LSTMs & Sequence-to-Sequence

## Vanilla RNN & BPTT
`h_t = tanh(W_h h_{t-1} + W_x x_t + b)`. Gradient: `dL/dh_k = dL/dh_T prod_{t=k+1}^{T} diag(tanh') W_h` - powers of `W_h` vanish/explode (Day 19). Truncated BPTT limits the horizon; clipping handles explosion.

## LSTM
```
i = s(W_i [x,h] )   f = s(W_f [x,h])   g = tanh(W_g [x,h])   o = s(W_o [x,h])
c_t = f * c_{t-1} + i * g          (additive cell path = gradient highway)
h_t = o * tanh(c_t)
```
`dc_t/dc_{t-1} = f` -> gradient is scaled by the forget gate instead of repeated matrix products. Init forget bias ~1. GRU: 2 gates, merges cell/hidden, ~25% fewer params.

## Seq2seq
```
src tokens -> Embedding -> LSTM encoder -> (h,c) context --> LSTM decoder -> softmax per step
                                                             ^ previous token (teacher forcing in training)
```
**Bottleneck:** the fixed-size context is lossy for long inputs -> **attention** (Bahdanau/Luong) lets the decoder look at all encoder states (Day 25). **Exposure bias:** train with gold prefixes, test with own predictions; mitigate with scheduled sampling. Decoding: greedy, beam search (length normalization).

## Trade-offs
| | RNN/LSTM | Transformer |
|---|---|---|
| Parallel training | no (sequential) | yes |
| Inference memory | O(1) state per step | O(T) KV cache |
| Long-range | weak | strong |
| Edge/streaming | attractive | heavier |

## Interview Questions
1. Why does the LSTM mitigate vanishing gradients? *(Additive cell update with gating.)*
2. Teacher forcing: pros/cons? *(Fast, stable; exposure bias.)*
3. Beam search vs greedy? *(Better global likelihood; cost k x; can favour short outputs.)*
4. Bidirectional RNN restrictions? *(Needs full sequence; not for online generation.)*
5. How do you batch variable-length sequences? *(Padding + masking / pack_padded_sequence; bucket by length.)*
6. Why are RNNs still used on-device? *(Constant-memory streaming inference.)*
