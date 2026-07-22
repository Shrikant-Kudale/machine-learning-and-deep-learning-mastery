"""Day 22 - ResNet skip connections: BasicBlock/Bottleneck, plain-vs-residual depth experiment, gradient flow."""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, residual=True):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.b2 = nn.BatchNorm2d(cout)
        self.residual = residual
        self.short = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(      # projection shortcut
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        if residual:
            nn.init.zeros_(self.b2.weight)        # zero-init last BN gamma: residual block starts as identity

    def forward(self, x):
        out = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(out + self.short(x)) if self.residual else F.relu(out)


class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand (x4): cheaper deep nets (ResNet-50+)."""
    expansion = 4

    def __init__(self, cin, mid, stride=1):
        super().__init__()
        cout = mid * 4
        self.f = nn.Sequential(nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(),
                               nn.Conv2d(mid, mid, 3, stride, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(),
                               nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.short = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.f(x) + self.short(x))


class Net(nn.Module):
    def __init__(self, n_blocks=8, width=16, residual=True, k=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[BasicBlock(width, width, residual=residual) for _ in range(n_blocks)])
        self.head = nn.Linear(width, k)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.blocks(self.stem(x)), 1).flatten(1))


def first_layer_grad(net, x, y):
    net.zero_grad()
    F.cross_entropy(net(x), y).backward()
    return net.stem[0].weight.grad.norm().item()


def train(net, data, epochs=20, seed=0):
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xtr)).split(64):
            opt.zero_grad(); F.cross_entropy(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return (net(Xte).argmax(1) == yte).float().mean().item()


def main(seed=0):
    X, y = load_digits(return_X_y=True)
    X = torch.tensor((X / 16.0).reshape(-1, 1, 8, 8), dtype=torch.float32)
    y = torch.tensor(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    data = (Xtr, ytr, Xte, yte)

    torch.manual_seed(seed)
    plain, res = Net(residual=False), Net(residual=True)
    g_plain, g_res = first_layer_grad(plain, Xtr[:128], ytr[:128]), first_layer_grad(res, Xtr[:128], ytr[:128])
    print(f"first-layer grad norm @ init, 8 blocks: plain={g_plain:.2e}  residual={g_res:.2e}")
    a_plain, a_res = train(plain, data), train(res, data)
    print(f"test acc: plain-8 blocks={a_plain:.3f}  ResNet-8 blocks={a_res:.3f}")

    b = Bottleneck(64, 16)
    p_bott = sum(p.numel() for p in b.parameters())
    p_basic = sum(p.numel() for p in BasicBlock(64, 64).parameters())
    print(f"params @64 channels: bottleneck(16->64)={p_bott:,}  basic(64)={p_basic:,}")
    assert BasicBlock(8, 8)(torch.randn(2, 8, 8, 8)).shape == (2, 8, 8, 8)
    assert Bottleneck(16, 8, stride=2)(torch.randn(2, 16, 8, 8)).shape == (2, 32, 4, 4)
    assert a_res > 0.9 and g_res > 0 and a_plain > 0.5
    print("Day 22 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
