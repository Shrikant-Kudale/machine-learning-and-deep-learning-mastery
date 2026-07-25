"""Day 25 - Self-attention from scratch: scaled dot-product, masks, positional encoding; invariants verified."""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    """softmax(QK^T / sqrt(d_k)) V.  mask: bool, True = keep. Returns (out, weights)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ v, w


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None):
        super().__init__()
        d_k = d_k or d_model
        self.q, self.k, self.v = nn.Linear(d_model, d_k, bias=False), nn.Linear(d_model, d_k, bias=False), nn.Linear(d_model, d_k, bias=False)

    def forward(self, x, causal=False, pad_mask=None):
        T = x.size(1)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        if causal:
            mask = torch.tril(mask)
        if pad_mask is not None:                                   # (B,T) True = real token
            mask = mask[None] & pad_mask[:, None, :]
        return attention(self.q(x), self.k(x), self.v(x), mask)


def sinusoidal_pe(T, d):
    pos, i = torch.arange(T)[:, None], torch.arange(0, d, 2)[None]
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / 10000 ** (i / d)), torch.cos(pos / 10000 ** (i / d))
    return pe


def main(seed=0):
    torch.manual_seed(seed)
    B, T, d = 2, 7, 16
    x = torch.randn(B, T, d)
    sa = SelfAttention(d)
    out, w = sa(x)
    ref = F.scaled_dot_product_attention(sa.q(x), sa.k(x), sa.v(x))
    assert torch.allclose(out, ref, atol=1e-5) and torch.allclose(w.sum(-1), torch.ones(B, T))
    print("matches F.scaled_dot_product_attention; rows of attention sum to 1.")

    _, wc = sa(x, causal=True)
    assert torch.allclose(wc.triu(1), torch.zeros_like(wc))                       # no peeking at future
    xp = x.clone(); xp[:, -1] += 5.0                                              # changing last token...
    assert torch.allclose(sa(x, causal=True)[0][:, :-1], sa(xp, causal=True)[0][:, :-1], atol=1e-6)   # ...leaves earlier outputs unchanged
    print("causal mask verified: output at t independent of tokens > t.")

    perm = torch.randperm(T)
    assert torch.allclose(sa(x[:, perm])[0], sa(x)[0][:, perm], atol=1e-5)        # permutation equivariance
    pe = sinusoidal_pe(T, d)
    shuffled_with_pe = sa(x[:, perm] + pe)[0]
    permuted_original = sa(x + pe)[0][:, perm]
    assert not torch.allclose(shuffled_with_pe, permuted_original, atol=1e-3)     # PE breaks the symmetry
    print("self-attention is permutation-equivariant -> needs positional information (sinusoidal PE).")

    pad = torch.tensor([[True] * 5 + [False] * 2] * B)
    _, wp = sa(x, pad_mask=pad)
    assert wp[..., 5:].abs().max() == 0
    qq, kk = torch.randn(1, 4, 256), torch.randn(1, 4, 256)                      # unit-variance q, k: Var(q.k) = d_k
    s = (qq @ kk.transpose(-1, -2)).softmax(-1).max(-1).values.mean().item()
    s_scaled = (qq @ kk.transpose(-1, -2) / 16).softmax(-1).max(-1).values.mean().item()
    print(f"softmax peakedness without / with 1/sqrt(d) scaling: {s:.3f} / {s_scaled:.3f}")
    assert s > s_scaled + 0.1
    print("Day 25 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
