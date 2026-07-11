"""Day 11 - Bias-variance decomposition via bootstrap on a known function (squared loss)."""
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

f = lambda x: np.sin(2 * np.pi * x)
SIGMA = 0.3


def decompose(make_model, n_train=60, n_rounds=300, seed=0):
    """Expected test error = bias^2 + variance + sigma^2, averaged over test x."""
    rng = np.random.default_rng(seed)
    xt = np.linspace(0, 1, 200)[:, None]
    preds = np.empty((n_rounds, len(xt)))
    for r in range(n_rounds):
        x = rng.uniform(0, 1, (n_train, 1))
        y = f(x[:, 0]) + rng.normal(0, SIGMA, n_train)
        preds[r] = make_model().fit(x, y).predict(xt)
    mean = preds.mean(0)
    bias2 = np.mean((mean - f(xt[:, 0])) ** 2)
    var = np.mean(preds.var(0))
    return bias2, var, bias2 + var + SIGMA ** 2


def main(seed=0):
    models = {
        "poly deg 1": lambda: make_pipeline(PolynomialFeatures(1), Ridge(1e-6)),
        "poly deg 3": lambda: make_pipeline(PolynomialFeatures(3), Ridge(1e-6)),
        "poly deg 12": lambda: make_pipeline(PolynomialFeatures(12), Ridge(1e-6)),
        "tree depth 2": lambda: DecisionTreeRegressor(max_depth=2),
        "tree depth None": lambda: DecisionTreeRegressor(),
        "kNN k=1": lambda: KNeighborsRegressor(1),
        "kNN k=10": lambda: KNeighborsRegressor(10),
    }
    print(f"{'model':<16s} {'bias^2':>8s} {'var':>8s} {'total':>8s}   (noise floor {SIGMA ** 2:.3f})")
    out = {}
    for name, mk in models.items():
        out[name] = decompose(mk, seed=seed)
        print(f"{name:<16s} {out[name][0]:8.4f} {out[name][1]:8.4f} {out[name][2]:8.4f}")
    assert out["poly deg 1"][0] > out["poly deg 3"][0]            # underfit: high bias
    assert out["poly deg 12"][1] > out["poly deg 3"][1]           # overfit: high variance
    assert out["tree depth None"][1] > out["tree depth 2"][1]
    assert out["kNN k=1"][1] > out["kNN k=10"][1]
    print("Day 11 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
