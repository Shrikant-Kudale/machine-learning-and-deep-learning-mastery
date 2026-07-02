"""Day 02 - Logistic & softmax regression from scratch (GD and Newton/IRLS)."""
import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))  # stable


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_gd(X, y, lr=0.5, epochs=500):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        w -= lr * Xb.T @ (sigmoid(Xb @ w) - y) / len(y)  # gradient = X^T (p - y) / n
    return w


def fit_newton(X, y, iters=15, l2=1e-6):
    """IRLS: w <- w - H^-1 g, H = X^T S X, S = diag(p(1-p))."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        g = Xb.T @ (p - y) / len(y)
        H = (Xb.T * (p * (1 - p))) @ Xb / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_softmax(X, y, k, lr=0.5, epochs=400):
    Xb = np.c_[np.ones(len(X)), X]
    W = np.zeros((Xb.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(epochs):
        W -= lr * Xb.T @ (softmax(Xb @ W) - Y) / len(y)
    return W


def main(seed=42):
    X, y = make_classification(2000, 10, n_informative=6, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xb = lambda A: np.c_[np.ones(len(A)), A]

    w_gd, w_nt = fit_gd(Xtr, ytr), fit_newton(Xtr, ytr)
    ref = LogisticRegression(C=1e6, max_iter=1000).fit(Xtr, ytr)
    for name, w in [("GD", w_gd), ("Newton", w_nt)]:
        p = sigmoid(Xb(Xte) @ w)
        print(f"{name:>7s}: acc={accuracy_score(yte, p > .5):.3f} bce={bce(yte, p):.4f}")
    print(f"sklearn: acc={ref.score(Xte, yte):.3f} bce={log_loss(yte, ref.predict_proba(Xte)):.4f}")
    assert np.allclose(w_nt[1:], ref.coef_[0], atol=5e-2)

    Xm, ym = make_classification(1500, 8, n_informative=6, n_classes=3, random_state=seed)
    Xm = StandardScaler().fit_transform(Xm)
    W = fit_softmax(Xm, ym, 3)
    acc = accuracy_score(ym, softmax(Xb(Xm) @ W).argmax(1))
    ref_acc = LogisticRegression(C=1e6, max_iter=2000).fit(Xm, ym).score(Xm, ym)
    print(f"softmax 3-class train acc ours={acc:.3f} sklearn={ref_acc:.3f}")
    assert abs(acc - ref_acc) < 0.02
    print("Day 02 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
