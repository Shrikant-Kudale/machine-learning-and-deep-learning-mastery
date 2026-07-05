"""Day 05 - Gradient boosting from scratch (squared loss + logistic loss) vs XGBoost/LightGBM if installed."""
import argparse
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class GBM:
    """F_m = F_{m-1} + lr * h_m, h_m fit to the negative gradient (pseudo-residuals)."""

    def __init__(self, loss="mse", n_estimators=150, lr=0.1, max_depth=3, subsample=1.0, seed=0):
        self.loss, self.n, self.lr, self.depth = loss, n_estimators, lr, max_depth
        self.subsample, self.rng = subsample, np.random.default_rng(seed)

    def _grad(self, y, F):
        if self.loss == "mse":
            return y - F                                  # -dL/dF for 1/2 (y-F)^2
        return y - 1 / (1 + np.exp(-F))                   # -dL/dF for logistic loss

    def fit(self, X, y):
        if self.loss == "mse":
            self.f0 = y.mean()
        else:
            p = y.mean()
            self.f0 = np.log(p / (1 - p))
        F = np.full(len(y), self.f0)
        self.trees = []
        for _ in range(self.n):
            r = self._grad(y, F)
            idx = self.rng.random(len(y)) < self.subsample
            t = DecisionTreeRegressor(max_depth=self.depth).fit(X[idx], r[idx])
            F += self.lr * t.predict(X)
            self.trees.append(t)
        return self

    def decision(self, X):
        return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)

    def predict(self, X):
        z = self.decision(X)
        return z if self.loss == "mse" else (z > 0).astype(int)


def maybe_libs(Xtr, ytr, Xte, yte):
    out = {}
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1).fit(Xtr, ytr).score(Xte, yte)
    except ImportError:
        out["xgboost"] = "not installed"
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, verbose=-1).fit(Xtr, ytr).score(Xte, yte)
    except ImportError:
        out["lightgbm"] = "not installed"
    return out


def main(seed=42):
    X, y = make_regression(1500, 8, noise=15, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    ours = GBM("mse", seed=seed).fit(Xtr, ytr)
    ref = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed).fit(Xtr, ytr)
    m_o, m_r = mean_squared_error(yte, ours.predict(Xte)), mean_squared_error(yte, ref.predict(Xte))
    print(f"regression MSE ours={m_o:.1f} sklearn={m_r:.1f}")
    assert m_o < 1.25 * m_r

    X, y = make_classification(2000, 15, n_informative=8, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    clf = GBM("logistic", seed=seed).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    ref_acc = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed).fit(Xtr, ytr).score(Xte, yte)
    print(f"classification acc ours={acc:.3f} sklearn={ref_acc:.3f}", maybe_libs(Xtr, ytr, Xte, yte))
    assert acc > ref_acc - 0.03
    print("Day 05 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
