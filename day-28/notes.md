# Day 28 - ONNX, ONNX Runtime & TensorRT

## Pipeline
```
PyTorch (eval mode, fixed preprocessing) --torch.onnx.export--> model.onnx (opset >= 17, dynamic batch)
   -> onnx.checker / onnxsim -> ORT (CPU/CUDA/TensorRT/OpenVINO/CoreML EPs) | trtexec -> TensorRT engine (.plan)
   -> parity test vs PyTorch (atol ~1e-4) -> latency p50/p99 on TARGET hardware -> package in container
```
**Export checklist:** `model.eval()`; trace with representative input; name inputs/outputs; mark dynamic axes (batch, sequence); avoid Python control flow on tensor values; custom ops need symbolic functions or plugins; verify numerically on several shapes.
**ORT optimizations:** constant folding, Conv-BN fusion, GELU/LayerNorm fusion, memory planning. Execution providers pick kernels per hardware.
**TensorRT:** builds an engine for a specific GPU/TensorRT version: layer/tensor fusion, kernel auto-tuning, FP16/INT8 with calibration, dynamic-shape optimization profiles (min/opt/max).
```
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 --minShapes=image:1x1x16x16 \
        --optShapes=image:8x1x16x16 --maxShapes=image:32x1x16x16
```
## Trade-offs
| Runtime | Portability | Peak perf | Notes |
|---|---|---|---|
| PyTorch eager | high | low | dev |
| TorchScript / torch.compile | medium | medium | stays in PyTorch |
| ONNX Runtime | very high | good | default portable serving |
| TensorRT | NVIDIA only | best | engines not portable across GPUs |
| TFLite / Core ML / ExecuTorch | mobile | good | NPU delegates |

## Interview Questions
1. Why export to ONNX? *(Framework-independent graph; hardware-specific optimizers.)*
2. What breaks export? *(Data-dependent control flow, unsupported ops, dynamic shapes mishandled.)*
3. How do you validate an exported model? *(Output diff on random + real inputs, across batch sizes, plus metric check.)*
4. Why is p99 more important than mean in serving? *(Tail latency drives SLOs.)*
5. TensorRT engine portability? *(Tied to GPU arch + TRT version; rebuild per target.)*
6. How do you handle preprocessing consistency? *(Bake into graph or share one implementation; test on raw inputs.)*
