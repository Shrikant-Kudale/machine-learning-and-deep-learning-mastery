"""Day 30 - ML system design prototypes: (A) two-tower real-time retrieval with ANN (IVF) + rerank, (B) visual anomaly detection service."""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------- A. Two-tower recommender ----------------
class Tower(nn.Module):
    def __init__(self, n_ids, d_feat, d=32):
        super().__init__()
        self.emb, self.mlp = nn.Embedding(n_ids, d), nn.Sequential(nn.Linear(d_feat, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, ids, feats):
        return F.normalize(self.emb(ids) + self.mlp(feats), dim=-1)


def synthetic_interactions(n_users=800, n_items=1200, d=8, n=40000, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, I = torch.randn(n_users, d, generator=g), torch.randn(n_items, d, generator=g)
    users = torch.randint(0, n_users, (n,), generator=g)
    logits = (U[users] @ I.T) * 2.0
    items = torch.multinomial(F.softmax(logits, -1), 1, generator=g).squeeze(1)
    return U, I, users, items


class IVF:
    """Inverted-file ANN: k-means coarse quantizer; search only `nprobe` closest lists."""

    def __init__(self, X, n_lists=32, iters=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X, self.C = X, X[torch.randperm(len(X), generator=g)[:n_lists]].clone()
        for _ in range(iters):
            a = (X @ self.C.T).argmax(1)
            for k in range(n_lists):
                if (a == k).any():
                    self.C[k] = F.normalize(X[a == k].mean(0), dim=0)
        self.lists = [torch.nonzero(a == k).squeeze(1) for k in range(n_lists)]

    def search(self, q, k=10, nprobe=4):
        probe = (q @ self.C.T).topk(nprobe, dim=1).indices
        out = []
        for qi, pr in zip(q, probe):
            cand = torch.cat([self.lists[p] for p in pr])
            out.append(cand[(self.X[cand] @ qi).topk(min(k, len(cand))).indices])
        return out


def part_a(seed=0):
    U, I, users, items = synthetic_interactions(seed=seed)
    torch.manual_seed(seed)
    ut, it = Tower(len(U), U.shape[1]), Tower(len(I), I.shape[1])
    opt = torch.optim.AdamW(list(ut.parameters()) + list(it.parameters()), lr=5e-3)
    n_train = int(0.8 * len(users))
    for ep in range(15):
        for idx in torch.randperm(n_train).split(512):
            u, i = users[idx], items[idx]
            zu, zi = ut(u, U[u]), it(i, I[i])
            logits = zu @ zi.T / 0.1                                    # in-batch softmax; logQ correction omitted
            loss = F.cross_entropy(logits, torch.arange(len(u)))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        item_vecs = it(torch.arange(len(I)), I)
        ann = IVF(item_vecs)
        tu, ti = users[n_train:][:500], items[n_train:][:500]
        zq = ut(tu, U[tu])
        exact = (zq @ item_vecs.T).topk(20, dim=1).indices
        t0 = time.perf_counter(); approx = ann.search(zq, k=20, nprobe=6); t_ann = time.perf_counter() - t0
    hit_exact = np.mean([ti[j].item() in exact[j].tolist() for j in range(len(ti))])
    hit_ann = np.mean([ti[j].item() in approx[j].tolist() for j in range(len(ti))])
    rand = 20 / len(I)
    overlap = np.mean([len(set(exact[j].tolist()) & set(approx[j].tolist())) / 20 for j in range(len(ti))])
    print(f"[A] recall@20 held-out: exact={hit_exact:.3f} IVF={hit_ann:.3f} random={rand:.3f} | ANN/exact overlap={overlap:.2f} | ANN {t_ann / len(ti) * 1000:.2f} ms/query")
    assert hit_exact > 5 * rand and overlap > 0.5


# ---------------- B. Visual anomaly detection ----------------
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(32 * 2 * 2, 8))
        self.dec = nn.Sequential(nn.Linear(8, 32 * 2 * 2), nn.ReLU(), nn.Unflatten(1, (32, 2, 2)),
                                 nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def part_b(seed=0):
    torch.manual_seed(seed)
    X, y = load_digits(return_X_y=True)
    X = torch.tensor((X / 16.0).reshape(-1, 1, 8, 8), dtype=torch.float32)
    normal = (y == 0) | (y == 1) | (y == 2)                               # "normal" product classes; others = defects
    Xn_tr, Xn_te = train_test_split(X[normal], test_size=0.3, random_state=seed)
    Xa = X[~torch.tensor(normal)]
    ae = ConvAE()
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    for _ in range(80):
        for idx in torch.randperm(len(Xn_tr)).split(64):
            opt.zero_grad(); F.mse_loss(ae(Xn_tr[idx]), Xn_tr[idx]).backward(); opt.step()
    ae.eval()
    score = lambda a: ((ae(a) - a) ** 2).flatten(1).mean(1)
    with torch.no_grad():
        sn, sa, str_ = score(Xn_te), score(Xa), score(Xn_tr)
    auc = roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn.numpy(), sa.numpy()])
    thr = np.percentile(str_.numpy(), 99)                                  # threshold from normal training scores
    print(f"[B] anomaly AUROC={auc:.3f} | thr@p99(train)={thr:.4f} | FPR on normal test={float((sn > thr).float().mean()):.3f} | TPR={float((sa > thr).float().mean()):.3f}")
    assert auc > 0.8


def main(seed=0):
    part_a(seed); part_b(seed)
    print("\nLatency budget (real-time recsys, 100 ms p99): features 15 | retrieval ANN 10 | ranking 40 | re-rank/business rules 10 | network 15 | headroom 10")
    print("Day 30 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
