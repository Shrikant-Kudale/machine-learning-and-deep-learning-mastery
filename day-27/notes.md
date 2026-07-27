# Day 27 - Quantization (INT8) & Pruning

## Affine quantization
`q = clamp(round(x/s) + z, q_min, q_max)`, `x_hat = s (q - z)`.
Symmetric (z = 0, `s = max|x|/127`) for weights; asymmetric for skewed activations. Per-channel scales on weights (one `s` per output row) lower error with negligible cost.
INT8 matmul: `y ~ s_x s_w * (X_q W_q^T)` accumulated in INT32, then requantized; bias in INT32.

```
FP32 model -> calibrate (100-500 batches: observe activation ranges) -> PTQ ----> INT8 model
                         |-> if accuracy drops: QAT (fake-quant nodes + straight-through estimator)
```
| Method | Needs training | Accuracy | Use |
|---|---|---|---|
| Dynamic PTQ | no | good for LSTM/Transformer linears | quick CPU wins |
| Static PTQ | calibration data | good for CNNs | edge default |
| QAT | fine-tune | best | when PTQ degrades (MobileNet, depthwise) |
| 4-bit weight-only (GPTQ/AWQ) | calibration | LLM-friendly | LLM serving |

## Pruning
- **Unstructured:** zero individual weights (magnitude); high sparsity but needs sparse kernels to speed up.
- **Structured:** remove neurons/channels/heads -> dense, smaller layers, real latency gain.
- Lottery ticket hypothesis; iterative prune + fine-tune beats one-shot. Combine: prune -> distill -> quantize.

## Deployment targets
ONNX Runtime (CPU/GPU/NPU EPs), TensorRT (NVIDIA; INT8 calibrators, layer fusion), TFLite/XNNPACK, Core ML, OpenVINO. Check **operator support** and fused patterns (Conv-BN-ReLU) before quantizing.

## Interview Questions
1. Why per-channel for weights? *(Row ranges differ; one scale wastes resolution.)*
2. Why is outlier handling critical in LLM quantization? *(A few huge activation channels dominate range; SmoothQuant/AWQ/rotations.)*
3. PTQ vs QAT? *(Cost vs accuracy.)*
4. Does 90% unstructured sparsity give 10x speed? *(No, not without sparse hardware kernels.)*
5. What layers are quantization-sensitive? *(First/last layers, depthwise convs, attention softmax, LayerNorm.)*
6. How do you validate a quantized model? *(Accuracy on holdout, per-layer SQNR, latency on target device, edge-case slices.)*
