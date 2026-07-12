"""Day 12 - Explainability: exact Shapley values, permutation importance, and LIME from scratch."""
import argparse
import itertools
import math
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def exact_shapley(predict, x, background):
    """phi_i = sum_{S not containing i} |S|!(d-|S|-1)!/d! [v(S+i) - v(S)], v(S) = E_bg[f(x_S, X_bg,~S)]."""
    d = len(x)

    def v(S):
        Z = background.copy()
        Z[:, list(S)] = x[list(S)]
        return predict(Z).mean()

    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for k in range(d):
            w = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            for S in itertools.combinations(others, k):
                phi[i] += w * (v(S + (i,)) - v(S))
    return phi, v(()), v(tuple(range(d)))


def permutation_importance(predict, X, y, rng, repeats=5):
    base = np.mean((predict(X) - y) ** 2)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[j] += (np.mean((predict(Xp) - y) ** 2) - base) / repeats
    return imp


def lime(predict, x, scale, rng, n=3000, kernel_width=None):
    """Local linear surrogate: sample around x, weight by proximity, fit weighted ridge."""
    d = len(x)
    kernel_width = kernel_width or 0.75 * np.sqrt(d)
    Z = x + rng.normal(size=(n, d)) * scale
    dist = np.linalg.norm((Z - x) / scale, axis=1)
    w = np.exp(-dist ** 2 / kernel_width ** 2)
    return Ridge(alpha=1e-3).fit((Z - x) / scale, predict(Z), sample_weight=w).coef_ / scale


def main(seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_regression(1500, 6, n_informative=3, noise=5, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=seed).fit(Xtr, ytr)
    x0, bg = Xte[0], Xtr[rng.choice(len(Xtr), 40, replace=False)]

    phi, base, fx = exact_shapley(model.predict, x0, bg)
    print("Shapley phi      :", phi.round(2))
    print(f"efficiency check : base {base:.2f} + sum(phi) {phi.sum():.2f} = {base + phi.sum():.2f}  f(x)~{fx:.2f}")
    assert abs(base + phi.sum() - fx) < 1e-6, "Shapley efficiency axiom"

    imp = permutation_importance(model.predict, Xte, yte, rng)
    print("Permutation imp. :", imp.round(1))
    print("LIME local slope :", lime(model.predict, x0, Xtr.std(0), rng).round(2))
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(x0[None])[0]
        print("shap TreeExplainer:", np.round(sv, 2))
    except ImportError:
        print("(pip install shap to compare against TreeSHAP)")
    top = set(np.argsort(-imp)[:3])
    assert top == set(np.argsort(-np.abs(phi))[:3]) or len(top & set(np.argsort(-np.abs(phi))[:3])) >= 2
    print("Day 12 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
