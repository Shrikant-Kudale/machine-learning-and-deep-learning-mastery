"""Day 08 - Cross-validation for leaky / time-dependent data: stratified, group, time-series, purged+embargo."""
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


class PurgedKFold:
    """K contiguous folds; training samples within `purge` steps before or `embargo` steps after the test fold are dropped.
    Needed when labels look `horizon` steps into the future (overlapping labels)."""

    def __init__(self, n_splits=5, purge=10, embargo=10):
        self.n_splits, self.purge, self.embargo = n_splits, purge, embargo

    def split(self, X):
        n = len(X)
        bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)
        for a, b in zip(bounds[:-1], bounds[1:]):
            test = np.arange(a, b)
            train = np.r_[np.arange(0, max(0, a - self.purge)), np.arange(min(n, b + self.embargo), n)]
            yield train.astype(int), test


def ar1(n, phi, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def make_leaky_series(n=3000, phi=0.98, seed=0):
    """Slow autocorrelated features and an independent slow label process.
    Features carry NO real signal, but neighbouring rows look alike, so shuffled CV memorizes neighbours."""
    rng = np.random.default_rng(seed)
    X = np.c_[[ar1(n, phi, rng) for _ in range(3)]].T
    y = (ar1(n, phi, rng) > 0).astype(int)
    return X, y


def score(X, y, splits):
    s = []
    for tr, te in splits:
        m = RandomForestClassifier(60, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
        s.append(m.score(X[te], y[te]))
    return float(np.mean(s))


def main(seed=42):
    X, y = make_leaky_series(seed=seed)
    shuffled = score(X, y, KFold(5, shuffle=True, random_state=seed).split(X))
    ts = score(X, y, TimeSeriesSplit(5).split(X))
    purged = score(X, y, PurgedKFold(5, purge=100, embargo=100).split(X))
    print(f"shuffled KFold (leaky)  : {shuffled:.3f}")
    print(f"TimeSeriesSplit          : {ts:.3f}")
    print(f"PurgedKFold + embargo    : {purged:.3f}   (true skill ~0.5: features carry no signal)")

    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(100), 10)                    # 100 users x 10 rows
    Xg = 0.3 * rng.normal(size=(1000, 5)) + rng.normal(size=(100, 5))[groups]   # user fingerprint
    yg = (rng.random(100) > 0.5).astype(int)[groups]           # label depends only on user
    leaky = score(Xg, yg, KFold(5, shuffle=True, random_state=seed).split(Xg))
    safe = score(Xg, yg, GroupKFold(5).split(Xg, yg, groups))
    print(f"user-level label: KFold={leaky:.3f} GroupKFold={safe:.3f}")
    for tr, te in StratifiedKFold(5).split(Xg, yg):
        assert abs(yg[tr].mean() - yg.mean()) < 0.1
    assert shuffled > 0.7 and shuffled > purged + 0.15 and leaky > safe + 0.1
    print("Day 08 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
