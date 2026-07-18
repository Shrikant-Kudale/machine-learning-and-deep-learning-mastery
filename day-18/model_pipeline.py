"""Day 18 - Weight initialization: Xavier vs He, signal propagation through deep nets, and verification."""
import argparse
import math
import torch
import torch.nn as nn


def init_layer(layer, scheme, act):
    fan_in, fan_out = layer.in_features, layer.out_features
    with torch.no_grad():
        if scheme == "xavier":                       # Var(w) = 2/(fan_in+fan_out)
            layer.weight.normal_(0, math.sqrt(2.0 / (fan_in + fan_out)))
        elif scheme == "he":                         # Var(w) = 2/fan_in  (accounts for ReLU halving variance)
            layer.weight.normal_(0, math.sqrt(2.0 / fan_in))
        elif scheme == "naive":                      # N(0, 0.01^2) - the classic mistake
            layer.weight.normal_(0, 0.01)
        elif scheme == "lecun":
            layer.weight.normal_(0, math.sqrt(1.0 / fan_in))
        layer.bias.zero_()


def signal_stats(scheme, act, depth=50, width=512, seed=0):
    torch.manual_seed(seed)
    f = {"relu": torch.relu, "tanh": torch.tanh}[act]
    x = torch.randn(1024, width)
    stds = []
    for _ in range(depth):
        layer = nn.Linear(width, width)
        init_layer(layer, scheme, act)
        x = f(layer(x))
        stds.append(x.std().item())
    return stds


def main():
    print(f"{'scheme/act':<14s} {'std@L1':>9s} {'std@L10':>9s} {'std@L50':>9s}")
    res = {}
    for scheme, act in [("naive", "relu"), ("xavier", "relu"), ("he", "relu"),
                        ("naive", "tanh"), ("xavier", "tanh"), ("he", "tanh")]:
        s = signal_stats(scheme, act)
        res[(scheme, act)] = s
        print(f"{scheme + '/' + act:<14s} {s[0]:9.2e} {s[9]:9.2e} {s[-1]:9.2e}")
    assert res[("naive", "relu")][-1] < 1e-10                    # vanishes
    assert 0.1 < res[("he", "relu")][-1] < 10                    # stable
    assert res[("he", "relu")][-1] > 1e3 * res[("xavier", "relu")][-1] or res[("xavier", "relu")][-1] < 0.05
    assert 0.05 < res[("xavier", "tanh")][-1] < 2                # xavier fits tanh
    # parity with torch built-ins
    w = torch.empty(512, 512); nn.init.kaiming_normal_(w, nonlinearity="relu")
    assert abs(w.std().item() - math.sqrt(2 / 512)) < 5e-3
    print("He init keeps activation std ~O(1) through 50 ReLU layers; naive init collapses to ~0.")
    print("Day 18 checks passed.")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
