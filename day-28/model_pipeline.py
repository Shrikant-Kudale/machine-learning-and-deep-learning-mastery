"""Day 28 - ONNX export, ONNX Runtime inference parity + latency benchmark, dynamic INT8 quantization (TensorRT notes in notes.md)."""
import argparse
import os
import tempfile
import time
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.f(x))


def export(model, path):
    model.eval()
    dummy = torch.randn(1, 1, 16, 16)
    kw = dict(input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
              opset_version=17)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)       # TorchScript-based exporter (stable)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kw)


def bench(fn, x, iters=100, warmup=10):
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(iters):
        s = time.perf_counter(); fn(x); t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return np.percentile(t, 50), np.percentile(t, 99)


def main(seed=0):
    torch.manual_seed(seed)
    model = Net().eval()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.onnx")
        export(model, path)
        try:
            import onnxruntime as ort
        except ImportError:
            print("pip install onnxruntime to run the parity/benchmark section"); return
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

        for bs in (1, 7, 32):                                           # dynamic batch axis works
            x = torch.randn(bs, 1, 16, 16)
            with torch.no_grad():
                ref = model(x).numpy()
            out = sess.run(None, {"image": x.numpy()})[0]
            assert np.allclose(ref, out, atol=1e-4), bs
        print("ONNX Runtime output matches PyTorch for batch sizes 1, 7, 32.")

        x = torch.randn(1, 1, 16, 16)
        with torch.inference_mode():
            p50_t, p99_t = bench(lambda a: model(a), x)
        p50_o, p99_o = bench(lambda a: sess.run(None, {"image": a}), x.numpy())
        print(f"batch=1 latency ms  PyTorch p50={p50_t:.3f} p99={p99_t:.3f} | ORT p50={p50_o:.3f} p99={p99_o:.3f}")
        print(f"model file size: {os.path.getsize(path) / 1024:.1f} KB")

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            qpath = os.path.join(d, "model.int8.onnx")
            quantize_dynamic(path, qpath, weight_type=QuantType.QInt8)
            print(f"dynamic INT8 file size: {os.path.getsize(qpath) / 1024:.1f} KB")
        except Exception as e:                                           # quantization tooling is optional
            print("dynamic quantization skipped:", type(e).__name__)
    print("Day 28 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
