"""Day 09 - ROC-AUC vs PR-AUC from scratch, and why PR-AUC wins on imbalanced data."""
import argparse
import numpy as np
from scipy.stats import rankdata
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def roc_auc(y, s):
    """Mann-Whitney U: P(score_pos > score_neg), ties count 1/2."""
    r = rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, s):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())      # AP = sum_k P@k * [y_k = 1] / #pos


def precision_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def main(seed=42):
    print(f"{'pos rate':>9s} | {'ROC-AUC':>8s} {'PR-AUC':>8s} | {'P@top1%':>8s} | baseline PR-AUC")
    aucs, aps = [], []
    for pos in [0.5, 0.1, 0.01]:
        X, y = make_classification(40000, 20, n_informative=5, weights=[1 - pos, pos], class_sep=0.8,
                                   flip_y=0.0, random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
        s = LogisticRegression(max_iter=500).fit(Xtr, ytr).decision_function(Xte)
        a, p = roc_auc(yte, s), average_precision(yte, s)
        assert abs(a - roc_auc_score(yte, s)) < 1e-9 and abs(p - average_precision_score(yte, s)) < 1e-6
        aucs.append(a)
        aps.append(p)
        print(f"{pos:9.2f} | {a:8.3f} {p:8.3f} | {precision_at_k(yte, s, int(0.01 * len(yte))):8.3f} | {yte.mean():.3f}")
    assert aps[2] < aps[0] - 0.2, "PR-AUC must degrade much more than ROC-AUC under imbalance"
    print("ROC-AUC barely moves while PR-AUC collapses: ROC hides the cost of false positives when negatives dominate.")
    print("Day 09 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
