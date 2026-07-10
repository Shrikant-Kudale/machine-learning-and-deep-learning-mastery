"""Day 10 - Probability calibration: reliability curve, ECE, Brier, Platt scaling, isotonic regression."""
import argparse
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def ece(y, p, bins=10):
    """Expected Calibration Error: sum_b (n_b/n) |acc_b - conf_b|."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def reliability_table(y, p, bins=10):
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return list(zip(mean.round(2), frac.round(2)))


def main(seed=42):
    X, y = make_classification(6000, 20, n_informative=6, n_redundant=6, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed)
    Xfit, Xcal, yfit, ycal = train_test_split(Xtr, ytr, test_size=0.4, random_state=seed)

    base = GaussianNB().fit(Xfit, yfit)                      # overconfident (independence assumption)
    p_raw = base.predict_proba(Xte)[:, 1]

    platt = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5).fit(Xtr, ytr)
    iso = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(Xtr, ytr)
    rf = RandomForestClassifier(200, random_state=seed).fit(Xfit, yfit)

    rows = {"NaiveBayes raw": p_raw,
            "  + Platt (sigmoid)": platt.predict_proba(Xte)[:, 1],
            "  + Isotonic": iso.predict_proba(Xte)[:, 1],
            "RandomForest raw": rf.predict_proba(Xte)[:, 1]}
    print(f"{'model':<22s} {'ECE':>7s} {'Brier':>7s}")
    for k, p in rows.items():
        print(f"{k:<22s} {ece(yte, p):7.4f} {brier_score_loss(yte, p):7.4f}")
    print("reliability (pred, observed) NB raw:", reliability_table(yte, p_raw)[:5], "...")
    assert ece(yte, rows["  + Isotonic"]) < ece(yte, p_raw)
    assert ece(yte, rows["  + Platt (sigmoid)"]) < ece(yte, p_raw)
    print("Day 10 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
