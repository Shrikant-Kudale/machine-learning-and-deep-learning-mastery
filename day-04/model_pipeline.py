"""Day 04 - Decision tree (Gini/CART) and Random Forest from scratch."""
import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def gini(y, k):
    p = np.bincount(y, minlength=k) / len(y)
    return 1.0 - np.sum(p ** 2)


class Tree:
    def __init__(self, max_depth=8, min_samples=5, max_features=None, rng=None):
        self.max_depth, self.min_samples = max_depth, min_samples
        self.max_features, self.rng = max_features, rng or np.random.default_rng(0)

    def fit(self, X, y, k=None):
        self.k = k or int(y.max()) + 1
        self.root = self._grow(X, y, 0)
        return self

    def _best_split(self, X, y):
        n, d = X.shape
        feats = self.rng.choice(d, self.max_features or d, replace=False)
        best = (None, None, gini(y, self.k))
        for f in feats:
            order = np.argsort(X[:, f])
            xs, ys = X[order, f], y[order]
            left = np.zeros(self.k)
            right = np.bincount(ys, minlength=self.k).astype(float)
            for i in range(1, n):                       # O(n) sweep per feature after sort
                c = ys[i - 1]
                left[c] += 1
                right[c] -= 1
                if xs[i] == xs[i - 1]:
                    continue
                g = (i * (1 - np.sum((left / i) ** 2)) +
                     (n - i) * (1 - np.sum((right / (n - i)) ** 2))) / n
                if g < best[2]:
                    best = (f, (xs[i] + xs[i - 1]) / 2, g)
        return best

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or gini(y, self.k) == 0:
            return ("leaf", np.bincount(y, minlength=self.k) / len(y))
        f, t, _ = self._best_split(X, y)
        if f is None:
            return ("leaf", np.bincount(y, minlength=self.k) / len(y))
        m = X[:, f] <= t
        return ("node", f, t, self._grow(X[m], y[m], depth + 1), self._grow(X[~m], y[~m], depth + 1))

    def _one(self, x, node):
        while node[0] == "node":
            node = node[3] if x[node[1]] <= node[2] else node[4]
        return node[1]

    def predict_proba(self, X):
        return np.array([self._one(x, self.root) for x in X])


class Forest:
    def __init__(self, n_trees=50, seed=0, **kw):
        self.n_trees, self.kw, self.rng = n_trees, kw, np.random.default_rng(seed)

    def fit(self, X, y):
        n, d = X.shape
        self.k = int(y.max()) + 1
        self.trees = []
        for _ in range(self.n_trees):
            b = self.rng.integers(0, n, n)              # bootstrap
            t = Tree(max_features=max(1, int(np.sqrt(d))), rng=self.rng, **self.kw)
            self.trees.append(t.fit(X[b], y[b], self.k))
        return self

    def predict(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0).argmax(1)


def main(seed=42):
    X, y = make_classification(1200, 12, n_informative=6, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    tree = Tree(max_depth=6, rng=np.random.default_rng(seed)).fit(Xtr, ytr)
    a_tree = accuracy_score(yte, tree.predict_proba(Xte).argmax(1))
    a_rf = accuracy_score(yte, Forest(n_trees=30, seed=seed, max_depth=8).fit(Xtr, ytr).predict(Xte))
    a_ref = RandomForestClassifier(100, random_state=seed).fit(Xtr, ytr).score(Xte, yte)
    print(f"tree={a_tree:.3f}  forest(ours)={a_rf:.3f}  forest(sklearn)={a_ref:.3f}")
    assert a_rf >= a_tree - 0.02 and a_rf > 0.8
    print("Day 04 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
