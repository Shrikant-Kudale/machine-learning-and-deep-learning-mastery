"""Day 27 - Model compression: INT8 post-training quantization (per-tensor/per-channel, calibrated activations) and pruning."""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def quantize_symmetric(w, per_channel=False, bits=8):
    """q = clamp(round(w/s), -127, 127), s = max|w| / 127 (symmetric, zero-point 0)."""
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(dim=1, keepdim=True) if per_channel else w.abs().max()
    s = amax.clamp(min=1e-8) / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax).to(torch.int8), s


def quantize_affine(x, bits=8):
    """Asymmetric uint8 with zero-point for activations (e.g. post-ReLU / skewed ranges)."""
    qmin, qmax = 0, 2 ** bits - 1
    lo, hi = min(x.min().item(), 0.0), max(x.max().item(), 0.0)
    s = (hi - lo) / (qmax - qmin) or 1.0
    zp = int(round(qmin - lo / s))
    return torch.clamp(torch.round(x / s) + zp, qmin, qmax).to(torch.uint8), s, zp


class QLinear(nn.Module):
    """INT8 weights, calibrated static INT8 input activations; integer matmul emulated with int32 accumulation."""

    def __init__(self, lin, act_absmax, per_channel=True):
        super().__init__()
        self.wq, self.ws = quantize_symmetric(lin.weight.data, per_channel)
        self.b, self.s_in = lin.bias.data.clone(), act_absmax / 127

    def forward(self, x):
        xq = torch.clamp(torch.round(x / self.s_in), -127, 127)
        acc = xq.to(torch.int32) @ self.wq.to(torch.int32).t()                   # int32 accumulator
        return acc.float() * (self.s_in * self.ws.view(1, -1) if self.ws.dim() == 2 else self.s_in * self.ws) + self.b


def build_mlp():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


def accuracy(m, X, y):
    with torch.no_grad():
        return (m(X).argmax(1) == y).float().mean().item()


def magnitude_prune(model, sparsity):
    """Global unstructured magnitude pruning on weight matrices."""
    ws = torch.cat([m.weight.abs().flatten() for m in model if isinstance(m, nn.Linear)])
    thr = ws.kthvalue(int(sparsity * len(ws))).values
    for m in model:
        if isinstance(m, nn.Linear):
            m.weight.data *= (m.weight.abs() > thr)
    return model


def structured_prune_rows(lin, keep_ratio):
    """Remove whole output neurons (rows) with the smallest L2 norm -> real speedups on dense hardware."""
    k = max(1, int(keep_ratio * lin.out_features))
    idx = lin.weight.norm(dim=1).topk(k).indices.sort().values
    new = nn.Linear(lin.in_features, k)
    new.weight.data, new.bias.data = lin.weight.data[idx].clone(), lin.bias.data[idx].clone()
    return new, idx


def main(seed=0):
    torch.manual_seed(seed)
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split((X / 16.0).astype("float32"), y, test_size=0.25, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))
    net = build_mlp()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(40):
        for idx in torch.randperm(len(Xtr)).split(64):
            opt.zero_grad(); F.cross_entropy(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
    base = accuracy(net, Xte, yte)

    # weight-only error: per-tensor vs per-channel
    w = net[0].weight.data
    for pc in (False, True):
        q, s = quantize_symmetric(w, pc)
        print(f"weight RMSE per_channel={pc}: {((q.float() * s - w) ** 2).mean().sqrt():.5f}")

    # static PTQ with activation calibration
    calib, h, amax = Xtr[:256], Xtr[:256], []
    for m in net:
        if isinstance(m, nn.Linear):
            amax.append(h.abs().max().item())
        h = m(h)
    qnet, i = [], 0
    for m in net:
        if isinstance(m, nn.Linear):
            qnet.append(QLinear(m, amax[i])); i += 1
        else:
            qnet.append(m)
    qnet = nn.Sequential(*qnet)
    q_acc = accuracy(qnet, Xte, yte)
    fp32_kb = sum(p.numel() * 4 for p in net.parameters()) / 1024
    int8_kb = sum(m.wq.numel() + m.b.numel() * 4 for m in qnet if isinstance(m, QLinear)) / 1024
    print(f"FP32 acc={base:.3f} size={fp32_kb:.1f}KB | INT8 acc={q_acc:.3f} size={int8_kb:.1f}KB ({fp32_kb / int8_kb:.1f}x smaller)")

    q8, sc, zp = quantize_affine(torch.relu(torch.randn(1000)))
    assert q8.dtype == torch.uint8 and zp == 0                                  # post-ReLU range starts at 0
    for sp in (0.5, 0.8, 0.95):
        pruned = build_mlp()
        pruned.load_state_dict(net.state_dict())
        magnitude_prune(pruned, sp)
        zeros = sum((m.weight == 0).sum().item() for m in pruned if isinstance(m, nn.Linear)) / sum(m.weight.numel() for m in pruned if isinstance(m, nn.Linear))
        print(f"unstructured sparsity {sp:.0%}: actual={zeros:.2%} acc={accuracy(pruned, Xte, yte):.3f}")
    new, idx = structured_prune_rows(net[0], 0.5)
    print(f"structured: layer-1 neurons {net[0].out_features} -> {new.out_features}")
    assert q_acc > base - 0.03 and fp32_kb / int8_kb > 3.0
    print("Day 27 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
