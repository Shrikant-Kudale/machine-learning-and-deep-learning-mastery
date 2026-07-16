"""Day 16 - Optimizers from scratch: SGD+momentum, AdamW (decoupled decay), Lion, and warmup+cosine annealing."""
import argparse
import math
import torch
from torch.optim import Optimizer


class AdamWScratch(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]
                if not s:
                    s.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                s["t"] += 1
                s["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                s["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m_hat = s["m"] / (1 - b1 ** s["t"])                 # bias correction
                v_hat = s["v"] / (1 - b2 ** s["t"])
                p.mul_(1 - g["lr"] * g["weight_decay"])             # decoupled weight decay
                p.addcdiv_(m_hat, v_hat.sqrt().add_(g["eps"]), value=-g["lr"])


class Lion(Optimizer):
    """Chen et al. 2023: update = sign(beta1*m + (1-beta1)*g); only one momentum buffer (memory-light)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("m", torch.zeros_like(p))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def warmup_cosine(step, total, warmup, base_lr, min_lr=0.0):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def rosenbrock(p):
    x, y = p[0], p[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def run(opt_cls, steps=300, schedule=True, **kw):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([-1.0, 1.0]))
    opt = opt_cls([p], **kw)
    base = opt.param_groups[0]["lr"]
    for t in range(steps):
        if schedule:
            opt.param_groups[0]["lr"] = warmup_cosine(t, steps, 20, base)
        opt.zero_grad()
        rosenbrock(p).backward()
        opt.step()
    return rosenbrock(p).item()


def main():
    # 1) AdamW parity with torch.optim.AdamW on a small regression problem
    torch.manual_seed(0)
    X, y = torch.randn(64, 5), torch.randn(64, 1)
    a, b = torch.nn.Linear(5, 1), torch.nn.Linear(5, 1)
    b.load_state_dict(a.state_dict())
    oa, ob = AdamWScratch(a.parameters(), lr=1e-2), torch.optim.AdamW(b.parameters(), lr=1e-2)
    for _ in range(50):
        for m, o in [(a, oa), (b, ob)]:
            o.zero_grad()
            torch.nn.functional.mse_loss(m(X), y).backward()
            o.step()
    assert torch.allclose(a.weight, b.weight, atol=1e-6)
    print("AdamW from scratch matches torch.optim.AdamW.")

    # 2) schedule sanity
    lrs = [warmup_cosine(t, 100, 10, 1.0) for t in range(100)]
    assert lrs[0] < lrs[9] <= 1.0 and lrs[-1] < 0.01 and max(lrs) <= 1.0
    # 3) optimizer comparison on Rosenbrock
    res = {"AdamW": run(AdamWScratch, lr=0.05, weight_decay=0.0),
           "Lion": run(Lion, lr=0.01, weight_decay=0.0),
           "SGD+momentum": run(lambda ps, lr: torch.optim.SGD(ps, lr=lr, momentum=0.9), lr=1e-3)}
    for k, v in res.items():
        print(f"{k:>13s}: final Rosenbrock loss = {v:.4f}")
    assert res["AdamW"] < 1.0 and res["Lion"] < 1.0
    print("Day 16 checks passed.")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
