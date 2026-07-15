"""Day 15 - Custom autograd Function, a micro scalar autograd engine, and nn.Module design patterns."""
import argparse
import math
import torch
import torch.nn as nn


# ---- 1. Custom autograd.Function with hand-written backward ------------------------------------
class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        return x @ w.t() + b

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return g @ w, g.t() @ x, g.sum(0)


class SwishFn(torch.autograd.Function):
    """x * sigmoid(x); saves only x and recomputes sigmoid in backward (memory vs compute)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return g * (s + x * s * (1 - s))


# ---- 2. Scalar reverse-mode autograd (micrograd style) -----------------------------------------
class Value:
    def __init__(self, data, parents=(), op=""):
        self.data, self.grad, self._parents, self._back = data, 0.0, parents, lambda: None

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def back():
            self.grad += out.grad; o.grad += out.grad
        out._back = back
        return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def back():
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._back = back
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def back():
            self.grad += (1 - t * t) * out.grad
        out._back = back
        return out

    def backward(self):
        order, seen = [], set()
        def topo(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    topo(p)
                order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order):
            v._back()


# ---- 3. nn.Module design: config-driven, parameters registered, no hidden state ------------------
class ResidualBlock(nn.Module):
    def __init__(self, d, expansion=2, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, expansion * d), nn.Linear(expansion * d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.fc2(nn.functional.gelu(self.fc1(self.norm(x)))))


class ResMLP(nn.Module):
    def __init__(self, d_in, d_model, n_blocks, n_out):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.Sequential(*[ResidualBlock(d_model) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_out))
        self.register_buffer("steps", torch.zeros(()))        # buffer: saved in state_dict, no gradient

    def forward(self, x):
        return self.head(self.blocks(self.inp(x)))


def main(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
    w = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    b = torch.randn(3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(LinearFn.apply, (x, w, b))
    assert torch.autograd.gradcheck(SwishFn.apply, (x,))
    print("gradcheck passed for LinearFn and SwishFn.")

    a, c = Value(0.7), Value(-1.3)
    (a * c + a).tanh().backward()
    ta, tc = torch.tensor(0.7, dtype=torch.double, requires_grad=True), torch.tensor(-1.3, dtype=torch.double, requires_grad=True)
    torch.tanh(ta * tc + ta).backward()
    assert abs(a.grad - ta.grad.item()) < 1e-9 and abs(c.grad - tc.grad.item()) < 1e-9
    print("scalar autograd engine matches torch.")

    net = ResMLP(8, 32, 3, 2)
    n_params = sum(p.numel() for p in net.parameters())
    sd = net.state_dict()
    assert "steps" in sd and "steps" not in dict(net.named_parameters())
    net2 = ResMLP(8, 32, 3, 2)
    net2.load_state_dict(sd)
    net.eval(); net2.eval()
    inp = torch.randn(5, 8)
    assert torch.allclose(net(inp), net2(inp))
    print(f"ResMLP params={n_params:,}; state_dict round-trip OK; eval() disables dropout deterministically.")
    print("Day 15 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
