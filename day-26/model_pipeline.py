"""Day 26 - Multi-head attention and a pre-LN Transformer block in PyTorch; tiny causal LM on a modular-arithmetic sequence."""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)            # fused projection
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal=True, return_weights=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # each (B,h,T,dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        if causal:
            scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)), float("-inf"))
        w = self.drop(scores.softmax(-1))
        out = (w @ v).transpose(1, 2).reshape(B, T, D)                                # concat heads
        out = self.proj(out)
        return (out, w) if return_weights else out


class TransformerBlock(nn.Module):
    """Pre-LN: x + MHA(LN(x)); x + MLP(LN(x)). MLP expansion 4x with GELU."""

    def __init__(self, d, h, p=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, p)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, d=64, h=4, layers=2, max_len=64):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.Sequential(*[TransformerBlock(d, h) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight                          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        return self.head(self.ln(self.blocks(x)))


def make_seqs(n, T, vocab, gen):
    """x_{t+1} = (x_t + x_{t-1}) mod vocab - needs attention to the previous two tokens."""
    s = torch.zeros(n, T, dtype=torch.long)
    s[:, :2] = torch.randint(0, vocab, (n, 2), generator=gen)
    for t in range(2, T):
        s[:, t] = (s[:, t - 1] + s[:, t - 2]) % vocab
    return s


def main(seed=0):
    torch.manual_seed(seed)
    d, h = 32, 4
    mine = MultiHeadAttention(d, h)
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(mine.qkv.weight); ref.in_proj_bias.copy_(mine.qkv.bias)
        ref.out_proj.weight.copy_(mine.proj.weight); ref.out_proj.bias.copy_(mine.proj.bias)
    x = torch.randn(3, 9, d)
    mask = torch.triu(torch.ones(9, 9, dtype=torch.bool), 1)
    assert torch.allclose(mine(x), ref(x, x, x, attn_mask=mask, need_weights=False)[0], atol=1e-5)
    print("MHA matches nn.MultiheadAttention (causal).")
    blk = TransformerBlock(d, h)
    xp = x.clone(); xp[:, -1] += 3
    assert torch.allclose(blk(x)[:, :-1], blk(xp)[:, :-1], atol=1e-5)
    print("Transformer block respects causality.")

    vocab, T = 11, 16
    gen = torch.Generator().manual_seed(seed)
    model = TinyLM(vocab, d=64, h=4, layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    for step in range(600):
        s = make_seqs(128, T, vocab, gen)
        loss = F.cross_entropy(model(s[:, :-1]).reshape(-1, vocab), s[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    s = make_seqs(512, T, vocab, torch.Generator().manual_seed(99))
    with torch.no_grad():
        pred = model(s[:, :-1]).argmax(-1)
    acc = (pred[:, 2:] == s[:, 3:]).float().mean().item()          # positions that are fully determined
    print(f"params={n_params:,} loss={loss.item():.3f} next-token accuracy on determined positions={acc:.3f}")
    assert acc > 0.9
    print("Day 26 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
