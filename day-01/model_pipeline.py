"""Day 01 - Linear regression from scratch: normal equation, batch GD, mini-batch SGD."""
import argparse
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def normal_equation(X, y):
    """theta = (X^T X)^-1 X^T y, solved via lstsq for numerical stability."""
    Xb = np.c_[np.ones(len(X)), X]
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta


def gradient_descent(X, y, lr=0.1, epochs=500, batch_size=None, seed=0):
    rng = np.random.default_rng(seed)
    Xb = np.c_[np.ones(len(X)), X]
    n, d = Xb.shape
    theta = np.zeros(d)
    history = []
    for _ in range(epochs):
        idx = rng.permutation(n) if batch_size else np.arange(n)
        step = batch_size or n
        for s in range(0, n, step):
            b = idx[s:s + step]
            grad = (2.0 / len(b)) * Xb[b].T @ (Xb[b] @ theta - y[b])  # dJ/dtheta
            theta -= lr * grad
        history.append(mse(y, Xb @ theta))
    return theta, history


def predict(theta, X):
    return np.c_[np.ones(len(X)), X] @ theta


def main(seed=42):
    X, y = make_regression(n_samples=2000, n_features=10, noise=10.0, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    th_ne = normal_equation(Xtr, ytr)
    th_gd, hist = gradient_descent(Xtr, ytr, lr=0.1, epochs=300)
    th_sgd, _ = gradient_descent(Xtr, ytr, lr=0.02, epochs=60, batch_size=64, seed=seed)
    ref = LinearRegression().fit(Xtr, ytr)

    res = {
        "normal_eq": mse(yte, predict(th_ne, Xte)),
        "batch_gd": mse(yte, predict(th_gd, Xte)),
        "minibatch_sgd": mse(yte, predict(th_sgd, Xte)),
        "sklearn": mse(yte, ref.predict(Xte)),
    }
    for k, v in res.items():
        print(f"{k:>14s} test MSE = {v:9.3f}")
    assert np.allclose(th_ne[1:], ref.coef_, atol=1e-6), "normal eq must match sklearn"
    assert hist[-1] <= hist[0], "GD loss must decrease"
    assert np.allclose(th_gd, th_ne, atol=1e-2), "GD must converge to the closed form"
    print("Day 01 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
