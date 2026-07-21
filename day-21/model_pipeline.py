"""Day 21 - Convolutions from scratch (im2col), output-size math, and a small CNN trained on 8x8 digits."""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    """x: (N,C,H,W), w: (O,C,kh,kw). Convolution == one big matmul on unfolded patches."""
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    cols = F.unfold(x, (kh, kw), padding=padding, stride=stride)       # (N, C*kh*kw, L)
    out = w.view(O, -1) @ cols                                          # (N, O, L)
    if b is not None:
        out = out + b.view(1, O, 1)
    return out.view(N, O, out_size(H, kh, stride, padding), out_size(W, kw, stride, padding))


def conv2d_naive(x, w, b, stride=1, padding=0):
    """Reference loops (slow) to show what im2col computes."""
    x = F.pad(x, (padding,) * 4)
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    Ho, Wo = (H - kh) // stride + 1, (W - kw) // stride + 1
    out = torch.zeros(N, O, Ho, Wo)
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = torch.einsum("nchw,ochw->no", patch, w) + b
    return out


class SmallCNN(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, k)

    def forward(self, x):
        return self.head(self.f(x).flatten(1))


def receptive_field(layers):
    r, j = 1, 1                                  # layers: list of (kernel, stride)
    for k, s in layers:
        r += (k - 1) * j
        j *= s
    return r


def main(seed=0):
    torch.manual_seed(seed)
    x, w, b = torch.randn(2, 3, 9, 9), torch.randn(4, 3, 3, 3), torch.randn(4)
    for s, p in [(1, 0), (1, 1), (2, 1)]:
        ref = F.conv2d(x, w, b, stride=s, padding=p)
        assert torch.allclose(conv2d_im2col(x, w, b, s, p), ref, atol=1e-5)
        assert torch.allclose(conv2d_naive(x, w, b, s, p), ref, atol=1e-4)
        assert ref.shape[-1] == out_size(9, 3, s, p)
    print("im2col and naive convolution match F.conv2d; output-size formula verified.")
    assert receptive_field([(3, 1), (3, 1), (3, 1)]) == 7 and receptive_field([(3, 2), (3, 2)]) == 7
    conv_params, fc_params = 3 * 3 * 3 * 16 + 16, (3 * 8 * 8) * (16 * 8 * 8)
    print(f"parameter sharing: 3x3 conv (3->16) has {conv_params} params vs {fc_params:,} for equivalent dense layer")

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8).astype("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)
    net = SmallCNN()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3)
    for _ in range(15):
        net.train()
        for idx in torch.randperm(len(Xtr)).split(64):
            opt.zero_grad(); F.cross_entropy(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(Xte).argmax(1) == yte).float().mean().item()
        shifted = torch.roll(Xte, 1, dims=3)                             # translation robustness probe
        acc_shift = (net(shifted).argmax(1) == yte).float().mean().item()
    print(f"CNN test acc={acc:.3f}; acc after 1-pixel shift={acc_shift:.3f}")
    assert acc > 0.93
    print("Day 21 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
