"""Day 17 - Activation functions: implementations, derivatives, dead-ReLU analysis, and a training comparison."""
import argparse
import math
import torch
import torch.nn as nn

ACTS = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": lambda x: torch.clamp(x, min=0),
    "leaky_relu": lambda x: torch.where(x > 0, x, 0.01 * x),
    "gelu": lambda x: 0.5 * x * (1 + torch.erf(x / math.sqrt(2))),
    "gelu_tanh": lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))),
    "silu": lambda x: x * torch.sigmoid(x),
    "mish": lambda x: x * torch.tanh(nn.functional.softplus(x)),
}
REFERENCE = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu,
             "leaky_relu": lambda x: nn.functional.leaky_relu(x, 0.01), "gelu": nn.functional.gelu,
             "gelu_tanh": lambda x: nn.functional.gelu(x, approximate="tanh"),
             "silu": nn.functional.silu, "mish": nn.functional.mish}


class Act(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.f = ACTS[name]

    def forward(self, x):
        return self.f(x)


def mean_grad_magnitude(name, n=2001):
    x = torch.linspace(-6, 6, n, requires_grad=True)
    ACTS[name](x).sum().backward()
    return x.grad.abs().mean().item()


def train(name, seed=0, steps=300):
    torch.manual_seed(seed)
    X = torch.randn(1024, 2)
    y = ((X[:, 0] * X[:, 1] > 0) ^ (X.norm(dim=1) > 1.2)).long()      # nonlinear XOR-ish target
    net = nn.Sequential(nn.Linear(2, 32), Act(name), nn.Linear(32, 32), Act(name), nn.Linear(32, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(X), y)
        loss.backward()
        opt.step()
    return loss.item(), (net(X).argmax(1) == y).float().mean().item()


def dead_relu_fraction(bias=-3.0):
    """Fraction of hidden units that never activate on the data when pre-activations are shifted negative."""
    torch.manual_seed(0)
    lin = nn.Linear(10, 256)
    nn.init.constant_(lin.bias, bias)
    h = torch.relu(lin(torch.randn(2048, 10)))
    return ((h > 0).sum(0) == 0).float().mean().item()


def main():
    x = torch.linspace(-5, 5, 101)
    for k in ACTS:
        assert torch.allclose(ACTS[k](x), REFERENCE[k](x), atol=1e-5), k
    print("all activations match torch references.")
    print(f"{'act':<11s} {'mean|grad|':>10s} {'final loss':>10s} {'acc':>6s}")
    out = {}
    for k in ACTS:
        loss, acc = train(k)
        out[k] = acc
        print(f"{k:<11s} {mean_grad_magnitude(k):10.3f} {loss:10.3f} {acc:6.3f}")
    dead = dead_relu_fraction()
    print(f"dead ReLU units with bias=-3: {dead:.1%}")
    assert dead > 0.3 and max(out.values()) > 0.9 and mean_grad_magnitude("sigmoid") < mean_grad_magnitude("relu")
    print("Day 17 checks passed.")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
