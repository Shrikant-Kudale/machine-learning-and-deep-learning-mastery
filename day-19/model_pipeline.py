"""Day 19 - Vanishing / exploding gradients: diagnosis per layer, and fixes (init, residuals, norm, clipping)."""
import argparse
import torch
import torch.nn as nn


def make_net(kind, depth=30, width=64):
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        if kind == "sigmoid_naive":
            nn.init.normal_(lin.weight, 0, 0.1); layers += [lin, nn.Sigmoid()]
        elif kind == "relu_he":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu"); layers += [lin, nn.ReLU()]
        elif kind == "relu_exploding":
            nn.init.normal_(lin.weight, 0, 0.3); layers += [lin, nn.ReLU()]
        elif kind == "relu_he_ln":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu"); layers += [lin, nn.LayerNorm(width), nn.ReLU()]
    return nn.Sequential(*layers)


class ResNet1D(nn.Module):
    def __init__(self, depth=30, width=64):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
                                     for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)                                  # identity path: dL/dx gets "+1" term
        return x


def layer_grad_norms(net):
    x = torch.randn(256, 64)
    net(x).pow(2).mean().backward()
    return [p.grad.norm().item() for n, p in net.named_parameters() if n.endswith("weight") and p.dim() == 2]


def main():
    torch.manual_seed(0)
    res = {}
    for kind in ["sigmoid_naive", "relu_he", "relu_exploding", "relu_he_ln"]:
        g = layer_grad_norms(make_net(kind))
        res[kind] = g
        print(f"{kind:<15s} grad-norm first layer {g[0]:.2e} | last layer {g[-1]:.2e} | ratio {g[0] / (g[-1] + 1e-30):.2e}")
    g = layer_grad_norms(ResNet1D())
    res["residual_prenorm"] = g
    print(f"{'residual_prenorm':<15s} grad-norm first layer {g[0]:.2e} | last layer {g[-1]:.2e} | ratio {g[0] / g[-1]:.2e}")

    assert res["sigmoid_naive"][0] / res["sigmoid_naive"][-1] < 1e-6        # vanishing
    assert max(res["relu_exploding"]) > 1e6 > 1e3 * max(res["relu_he"])       # exploding: absolute norms blow up
    assert 1e-2 < res["residual_prenorm"][0] / res["residual_prenorm"][-1] < 1e2

    net = make_net("relu_exploding")
    x = torch.randn(256, 64)
    net(x).pow(2).mean().backward()
    before = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    after = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters())).item()
    print(f"clip_grad_norm_: global norm {before:.2e} -> {after:.2f}")
    assert abs(after - 1.0) < 1e-4
    print("Day 19 checks passed.")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
