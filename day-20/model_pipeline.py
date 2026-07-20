"""Day 20 - Capstone: production-style PyTorch training harness (AdamW + warmup-cosine, clipping, AMP-ready,
early stopping, checkpoint/resume, determinism, throughput benchmark)."""
import argparse
import copy
import math
import os
import tempfile
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def make_model(d_in=64, k=10):
    m = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, k))
    for l in m:
        if isinstance(l, nn.Linear):
            nn.init.kaiming_normal_(l.weight, nonlinearity="relu"); nn.init.zeros_(l.bias)
    return m


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        return nn.functional.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()


def fit(model, data, epochs=30, bs=64, lr=3e-3, wd=1e-2, clip=1.0, patience=6, use_amp=False,
        ckpt=None, start_epoch=0, state=None, stop_after=None):
    Xtr, ytr, Xva, yva = data
    decay = [p for n, p in model.named_parameters() if p.dim() > 1]            # no decay on biases
    no_decay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    steps_per_epoch = math.ceil(len(Xtr) / bs)
    total = epochs * steps_per_epoch
    best, bad, best_state = float("inf"), 0, None
    if state:
        opt.load_state_dict(state["opt"]); best, bad, best_state = state["best"], state["bad"], state["best_state"]
        torch.set_rng_state(state["rng"])
    scaler = torch.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    for ep in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(ep))
        for i in range(steps_per_epoch):
            step = ep * steps_per_epoch + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, steps_per_epoch, lr)
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev_type, enabled=use_amp):
                loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update()
        vl, va = evaluate(model, Xva, yva)
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best, "bad": bad,
                        "best_state": best_state, "rng": torch.get_rng_state()}, ckpt)
        if bad >= patience or (stop_after is not None and ep + 1 >= stop_after):
            break
    if best_state and stop_after is None:
        model.load_state_dict(best_state)
    return model


def main(seed=0):
    seed_all(seed)
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    T = lambda a, dt=None: torch.tensor(a, dtype=dt)
    data = (T(Xtr), T(ytr, torch.long), T(Xva), T(yva, torch.long))
    Xte_t, yte_t = T(Xte), T(yte, torch.long)

    seed_all(seed); m1 = fit(make_model(), data)
    seed_all(seed); m2 = fit(make_model(), data)
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values())), "determinism"
    _, acc = evaluate(m1, Xte_t, yte_t)
    print(f"test accuracy = {acc:.3f} (deterministic across two runs)")
    assert acc > 0.95

    with tempfile.TemporaryDirectory() as d:                                    # interrupted + resumed == uninterrupted
        path = os.path.join(d, "ck.pt")
        seed_all(seed); a = make_model(); fit(a, data, epochs=8, ckpt=path, patience=99, stop_after=4)   # 'crash' after 4 of 8 epochs
        ck = torch.load(path, weights_only=False)
        b = make_model(); b.load_state_dict(ck["model"])
        fit(b, data, epochs=8, patience=99, start_epoch=ck["epoch"], state=ck)
        seed_all(seed); c = make_model(); fit(c, data, epochs=8, patience=99)
        same = all(torch.allclose(p, q, atol=1e-6) for p, q in zip(b.parameters(), c.parameters()))
        print("resume matches uninterrupted run:", same)
        assert same

    m1.eval(); xb = Xte_t[:128]
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(200):
            m1(xb)
        dt = time.perf_counter() - t0
    print(f"inference throughput: {200 * 128 / dt:,.0f} samples/s, {1000 * dt / 200:.3f} ms per batch of 128")
    print("Day 20 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
