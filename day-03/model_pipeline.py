"""Day 03 - Regularization from scratch: Ridge (closed form), Lasso (coordinate descent), ElasticNet."""
import argparse
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler


def soft_threshold(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def ridge(X, y, alpha):
    """Closed form on centered data: w = (X^T X + n*alpha I)^-1 X^T y (sklearn convention)."""
    n, d = X.shape
    return np.linalg.solve(X.T @ X + n * alpha * np.eye(d), X.T @ y)


def elastic_net_cd(X, y, alpha, l1_ratio=1.0, iters=500, tol=1e-8):
    """Objective: 1/(2n)||y-Xw||^2 + alpha*l1_ratio*||w||_1 + 0.5*alpha*(1-l1_ratio)*||w||^2."""
    n, d = X.shape
    w = np.zeros(d)
    r = y - X @ w
    col_sq = (X ** 2).sum(0) / n
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    for _ in range(iters):
        w_old = w.copy()
        for j in range(d):
            r += X[:, j] * w[j]                     # remove feature j from residual
            rho = X[:, j] @ r / n
            w[j] = soft_threshold(rho, l1) / (col_sq[j] + l2)
            r -= X[:, j] * w[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def main(seed=42):
    X, y, coef = make_regression(500, 30, n_informative=5, noise=5.0, coef=True, random_state=seed)
    X = StandardScaler().fit_transform(X)
    y = y - y.mean()

    for a in [0.1, 1.0, 10.0]:
        w = ridge(X, y, a)
        ref = Ridge(alpha=a * len(X), fit_intercept=False).fit(X, y).coef_
        assert np.allclose(w, ref, atol=1e-6)
    print("Ridge closed form matches sklearn.")

    w_l = elastic_net_cd(X, y, alpha=1.0, l1_ratio=1.0)
    ref_l = Lasso(alpha=1.0, fit_intercept=False, tol=1e-10, max_iter=10000).fit(X, y).coef_
    assert np.allclose(w_l, ref_l, atol=1e-4)
    w_e = elastic_net_cd(X, y, alpha=1.0, l1_ratio=0.5)
    ref_e = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False, tol=1e-10, max_iter=10000).fit(X, y).coef_
    assert np.allclose(w_e, ref_e, atol=1e-4)
    print("Lasso / ElasticNet coordinate descent match sklearn.")

    print(f"non-zeros: true={int((coef != 0).sum())} lasso={int((np.abs(w_l) > 1e-8).sum())} "
          f"enet={int((np.abs(w_e) > 1e-8).sum())} ridge={int((np.abs(ridge(X, y, 1.0)) > 1e-8).sum())}")
    assert (np.abs(w_l) > 1e-8).sum() < 30, "L1 must induce sparsity"
    print("Day 03 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
