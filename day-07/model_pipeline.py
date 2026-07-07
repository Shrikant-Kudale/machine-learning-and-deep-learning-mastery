"""Day 07 - Hyperparameter tuning: grid, random, successive halving, and Bayesian optimization (GP + EI)."""
import argparse
import itertools
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score, train_test_split

SPACE = {"lr": (-3.0, 0.0), "depth": (1, 6)}          # lr is log10-scaled


def build(cfg, n_estimators=60, seed=0):
    return GradientBoostingClassifier(learning_rate=10 ** cfg[0], max_depth=int(round(cfg[1])),
                                      n_estimators=n_estimators, random_state=seed)


def objective(cfg, X, y, n_estimators=60):
    return cross_val_score(build(cfg, n_estimators), X, y, cv=3).mean()


def grid_search(X, y, k=4):
    grid = itertools.product(np.linspace(*SPACE["lr"], k), np.linspace(*SPACE["depth"], k))
    return max(((objective(c, X, y), c) for c in grid), key=lambda t: t[0])


def random_search(X, y, n, rng):
    cands = [(rng.uniform(*SPACE["lr"]), rng.uniform(*SPACE["depth"])) for _ in range(n)]
    return max(((objective(c, X, y), c) for c in cands), key=lambda t: t[0])


def successive_halving(X, y, n, rng, eta=3):
    """Start with n configs on a small budget (few trees); keep top 1/eta each rung with eta x budget."""
    cands = [(rng.uniform(*SPACE["lr"]), rng.uniform(*SPACE["depth"])) for _ in range(n)]
    budget = 10
    while len(cands) > 1:
        scores = [objective(c, X, y, budget) for c in cands]
        keep = max(1, len(cands) // eta)
        cands = [cands[i] for i in np.argsort(scores)[::-1][:keep]]
        budget *= eta
    return objective(cands[0], X, y, budget), cands[0]


def bayes_opt(X, y, n_init, n_iter, rng):
    """GP surrogate with Expected Improvement: EI = (mu - f*) Phi(z) + sigma phi(z)."""
    lo, hi = np.array([SPACE["lr"][0], SPACE["depth"][0]]), np.array([SPACE["lr"][1], SPACE["depth"][1]])
    pts = [rng.uniform(lo, hi) for _ in range(n_init)]
    vals = [objective(p, X, y) for p in pts]
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True, alpha=1e-4).fit(np.array(pts), vals)
        cand = rng.uniform(lo, hi, size=(500, 2))
        mu, sd = gp.predict(cand, return_std=True)
        z = (mu - max(vals)) / (sd + 1e-9)
        nxt = cand[np.argmax((mu - max(vals)) * norm.cdf(z) + sd * norm.pdf(z))]
        pts.append(nxt)
        vals.append(objective(nxt, X, y))
    i = int(np.argmax(vals))
    return vals[i], pts[i]


def main(seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_classification(800, 20, n_informative=8, flip_y=0.05, random_state=seed)
    Xs, Xh, ys, yh = train_test_split(X, y, test_size=0.3, random_state=seed)
    results = {
        "grid(16 evals)": grid_search(Xs, ys, 4),
        "random(16)": random_search(Xs, ys, 16, rng),
        "succ.halving(27)": successive_halving(Xs, ys, 27, rng),
        "bayes(6+10)": bayes_opt(Xs, ys, 6, 10, rng),
    }
    for name, (cv, cfg) in results.items():
        hold = build(cfg, 60, seed).fit(Xs, ys).score(Xh, yh)
        print(f"{name:>18s}: cv={cv:.3f} holdout={hold:.3f} lr={10 ** cfg[0]:.3f} depth={int(round(cfg[1]))}")
        assert hold > 0.7
    print("Day 07 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
