"""Day 24 - LSTM cell from scratch (parity with nn.LSTM) and an LSTM encoder-decoder seq2seq trained to reverse sequences."""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCellScratch(nn.Module):
    """i,f,g,o = split(W [x,h] + b); c' = f*c + i*g; h' = o*tanh(c')   (PyTorch gate order: i,f,g,o)."""

    def __init__(self, d_in, d_h):
        super().__init__()
        self.W_ih = nn.Parameter(torch.empty(4 * d_h, d_in)); self.W_hh = nn.Parameter(torch.empty(4 * d_h, d_h))
        self.b_ih = nn.Parameter(torch.zeros(4 * d_h));       self.b_hh = nn.Parameter(torch.zeros(4 * d_h))
        k = d_h ** -0.5
        for p in (self.W_ih, self.W_hh):
            nn.init.uniform_(p, -k, k)
        self.d_h = d_h

    def forward(self, x, state):
        h, c = state
        z = x @ self.W_ih.t() + self.b_ih + h @ self.W_hh.t() + self.b_hh
        i, f, g, o = z.chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Seq2Seq(nn.Module):
    def __init__(self, vocab, d=64, pad=0, bos=1):
        super().__init__()
        self.emb, self.bos, self.vocab = nn.Embedding(vocab, d, padding_idx=pad), bos, vocab
        self.enc, self.dec = LSTMCellScratch(d, d), LSTMCellScratch(d, d)
        self.out = nn.Linear(d, vocab)
        self.d = d

    def encode(self, src):
        h = c = torch.zeros(src.shape[0], self.d)
        for t in range(src.shape[1]):
            h, c = self.enc(self.emb(src[:, t]), (h, c))
        return h, c

    def forward(self, src, tgt, teacher_forcing=1.0):
        state, logits = self.encode(src), []
        inp = torch.full((src.shape[0],), self.bos, dtype=torch.long)
        for t in range(tgt.shape[1]):
            h, c = self.dec(self.emb(inp), state)
            state = (h, c)
            logits.append(self.out(h))
            use_gt = torch.rand(()) < teacher_forcing
            inp = tgt[:, t] if use_gt else logits[-1].argmax(-1)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src, length):
        state = self.encode(src)
        inp, outs = torch.full((src.shape[0],), self.bos, dtype=torch.long), []
        for _ in range(length):
            state = self.dec(self.emb(inp), state)
            inp = self.out(state[0]).argmax(-1)
            outs.append(inp)
        return torch.stack(outs, 1)


def batch(n, L, vocab, gen):
    src = torch.randint(2, vocab, (n, L), generator=gen)
    return src, src.flip(1)


def main(seed=0):
    torch.manual_seed(seed)
    mine, ref = LSTMCellScratch(5, 7), nn.LSTMCell(5, 7)
    ref.load_state_dict({"weight_ih": mine.W_ih, "weight_hh": mine.W_hh, "bias_ih": mine.b_ih, "bias_hh": mine.b_hh})
    x, h, c = torch.randn(3, 5), torch.randn(3, 7), torch.randn(3, 7)
    for a, b in zip(mine(x, (h, c)), ref(x, (h, c))):
        assert torch.allclose(a, b, atol=1e-6)
    print("LSTM cell matches nn.LSTMCell.")

    vocab, L = 12, 6
    gen = torch.Generator().manual_seed(seed)
    model = Seq2Seq(vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(700):
        src, tgt = batch(64, L, vocab, gen)
        loss = F.cross_entropy(model(src, tgt, teacher_forcing=max(0.5, 1 - step / 1000)).reshape(-1, vocab), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    src, tgt = batch(500, L, vocab, torch.Generator().manual_seed(123))
    pred = model.greedy(src, L)
    tok_acc, seq_acc = (pred == tgt).float().mean().item(), (pred == tgt).all(1).float().mean().item()
    print(f"final loss={loss.item():.3f} token acc={tok_acc:.3f} exact-sequence acc={seq_acc:.3f}")
    assert tok_acc > 0.8
    print("Day 24 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
