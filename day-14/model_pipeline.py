"""Day 14 - Backpropagation from scratch (NumPy), verified by numerical and PyTorch autograd gradients."""
import argparse
import numpy as np
import torch


class MLP:
    """x -> Linear -> tanh -> Linear -> softmax-cross-entropy. Manual forward/backward."""

    def __init__(self, d_in, d_h, k, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(1 / d_in), (d_in, d_h)); self.b1 = np.zeros(d_h)
        self.W2 = r.normal(0, np.sqrt(1 / d_h), (d_h, k));     self.b2 = np.zeros(k)

    def forward(self, X, y):
        self.X, self.y = X, y
        self.a1 = np.tanh(X @ self.W1 + self.b1)
        z2 = self.a1 @ self.W2 + self.b2
        z2 = z2 - z2.max(1, keepdims=True)
        self.P = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
        return -np.mean(np.log(self.P[np.arange(len(y)), y] + 1e-12))

    def backward(self):
        n = len(self.y)
        dz2 = self.P.copy()
        dz2[np.arange(n), self.y] -= 1
        dz2 /= n                                     # dL/dz2 = (P - Y)/n
        g = {"W2": self.a1.T @ dz2, "b2": dz2.sum(0)}
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)               # tanh' = 1 - tanh^2
        g.update(W1=self.X.T @ dz1, b1=dz1.sum(0))
        return g


def numerical_grad(model, name, X, y, eps=1e-5):
    p = getattr(model, name)
    g = np.zeros_like(p)
    for idx in np.ndindex(p.shape):
        old = p[idx]
        p[idx] = old + eps; lp = model.forward(X, y)
        p[idx] = old - eps; lm = model.forward(X, y)
        p[idx] = old
        g[idx] = (lp - lm) / (2 * eps)
    return g


def main(seed=0):
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(32, 5)), rng.integers(0, 3, 32)
    m = MLP(5, 7, 3, seed)
    m.forward(X, y)
    grads = m.backward()

    for name, g in grads.items():
        num = numerical_grad(m, name, X, y)
        rel = np.linalg.norm(g - num) / (np.linalg.norm(g) + np.linalg.norm(num))
        print(f"grad check {name}: relative error {rel:.2e}")
        assert rel < 1e-7

    t = {k: torch.tensor(getattr(m, k), requires_grad=True) for k in ["W1", "b1", "W2", "b2"]}
    z = torch.tanh(torch.tensor(X) @ t["W1"] + t["b1"]) @ t["W2"] + t["b2"]
    torch.nn.functional.cross_entropy(z, torch.tensor(y)).backward()
    for k in t:
        assert np.allclose(t[k].grad.numpy(), grads[k], atol=1e-10)
    print("matches torch.autograd.")

    Xs = rng.normal(size=(600, 2)); ys = (np.sin(2 * Xs[:, 0]) + Xs[:, 1] > 0).astype(int)
    net = MLP(2, 32, 2, seed)
    first = net.forward(Xs, ys)
    for _ in range(800):
        net.forward(Xs, ys)
        for k, g in net.backward().items():
            setattr(net, k, getattr(net, k) - 0.5 * g)
    last = net.forward(Xs, ys)
    acc = (net.P.argmax(1) == ys).mean()
    print(f"loss {first:.3f} -> {last:.3f}  train acc {acc:.3f}")
    assert last < first and acc > 0.85
    print("Day 14 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
