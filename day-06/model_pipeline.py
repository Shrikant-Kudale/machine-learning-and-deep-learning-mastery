"""Day 06 - SVMs: kernel functions, Pegasos primal solver with kernels, and sklearn comparison."""
import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def rbf(A, B, gamma):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gamma * d)


def poly(A, B, degree=3, c=1.0):
    return (A @ B.T + c) ** degree


def linear(A, B):
    return A @ B.T


class KernelPegasos:
    """Kernelized Pegasos: stochastic sub-gradient descent on hinge loss + (lambda/2)||w||^2."""

    def __init__(self, kernel, lam=0.01, epochs=15, seed=0):
        self.k, self.lam, self.epochs, self.rng = kernel, lam, epochs, np.random.default_rng(seed)

    def fit(self, X, y):
        y = np.where(y > 0, 1, -1)
        n = len(y)
        K = self.k(X, X)
        alpha = np.zeros(n)
        t = 0
        for _ in range(self.epochs):
            for i in self.rng.permutation(n):
                t += 1
                margin = y[i] * (1 / (self.lam * t)) * np.sum(alpha * y * K[:, i])
                if margin < 1:
                    alpha[i] += 1
        self.alpha, self.y, self.X, self.t = alpha, y, X, t
        return self

    def decision(self, Xnew):
        return (1 / (self.lam * self.t)) * (self.alpha * self.y) @ self.k(self.X, Xnew)

    def predict(self, Xnew):
        return (self.decision(Xnew) > 0).astype(int)


def main(seed=42):
    X, y = make_moons(800, noise=0.25, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    gamma = 2.0
    lin = KernelPegasos(linear, epochs=10, seed=seed).fit(Xtr, ytr)
    rb = KernelPegasos(lambda A, B: rbf(A, B, gamma), epochs=10, seed=seed).fit(Xtr, ytr)
    ref = SVC(kernel="rbf", gamma=gamma, C=10).fit(Xtr, ytr)
    a_lin, a_rb = accuracy_score(yte, lin.predict(Xte)), accuracy_score(yte, rb.predict(Xte))
    a_ref = ref.score(Xte, yte)
    print(f"linear={a_lin:.3f}  rbf(ours)={a_rb:.3f}  rbf(sklearn SMO)={a_ref:.3f}  n_SV={ref.n_support_.sum()}")
    assert a_rb > a_lin and a_rb > 0.85
    print("Day 06 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
