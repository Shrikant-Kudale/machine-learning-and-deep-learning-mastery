"""Day 13 - Scikit-Learn custom transformers + leakage-safe pipelines (+ persistence)."""
import argparse
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numerics at learned quantiles (statistics fitted on train only)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanquantile(X, self.lower, 0), np.nanquantile(X, self.upper, 0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return names


class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding. fit_transform uses out-of-fold means to avoid target leakage."""

    def __init__(self, smoothing=10.0, n_splits=5):
        self.smoothing, self.n_splits = smoothing, n_splits

    def _stats(self, x, y):
        df = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["sum", "count"])
        return df

    def _enc(self, x, df, prior):
        s = pd.Series(x).map(df["sum"]).fillna(0).values
        c = pd.Series(x).map(df["count"]).fillna(0).values
        return (s + self.smoothing * prior) / (c + self.smoothing)

    def fit(self, X, y):
        x, y = np.asarray(X).ravel(), np.asarray(y)
        self.prior_, self.stats_ = y.mean(), self._stats(x, y)
        return self

    def transform(self, X):
        return self._enc(np.asarray(X).ravel(), self.stats_, self.prior_)[:, None]

    def fit_transform(self, X, y=None):
        x, y = np.asarray(X).ravel(), np.asarray(y)
        out = np.zeros(len(x))
        for tr, va in KFold(self.n_splits, shuffle=True, random_state=0).split(x):
            out[va] = self._enc(x[va], self._stats(x[tr], y[tr]), y[tr].mean())
        self.fit(X, y)
        return out[:, None]

    def get_feature_names_out(self, names=None):
        return names


def make_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    city = rng.choice([f"c{i}" for i in range(60)], n)
    eff = {c: rng.normal(0, 1) for c in set(city)}
    df = pd.DataFrame({"age": rng.normal(40, 12, n), "income": rng.lognormal(10, 1, n),
                       "channel": rng.choice(["web", "app", "store"], n), "city": city})
    df.loc[rng.random(n) < 0.1, "age"] = np.nan
    z = 0.03 * (df.age.fillna(40) - 40) + 0.4 * np.log(df.income / 22000) + df.city.map(eff) + rng.normal(0, 1, n)
    return df, (z > 0).astype(int)


def build_pipeline():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", QuantileClipper()), ("sc", StandardScaler())])
    pre = ColumnTransformer([("num", num, ["age", "income"]),
                             ("ohe", OneHotEncoder(handle_unknown="ignore"), ["channel"]),
                             ("te", OOFTargetEncoder(), ["city"])])
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_iter=100, random_state=0))])


def main(seed=42):
    df, y = make_data(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.25, random_state=seed)
    gs = GridSearchCV(build_pipeline(), {"clf__learning_rate": [0.05, 0.1], "pre__te__smoothing": [1.0, 20.0]}, cv=3)
    gs.fit(Xtr, ytr)
    acc = gs.score(Xte, yte)
    print(f"best params={gs.best_params_}  cv={gs.best_score_:.3f}  holdout acc={acc:.3f}")

    with tempfile.NamedTemporaryFile(suffix=".joblib") as f:     # serialize whole pipeline = train/serve parity
        joblib.dump(gs.best_estimator_, f.name)
        loaded = joblib.load(f.name)
    assert np.array_equal(loaded.predict(Xte), gs.predict(Xte))
    unseen = Xte.copy()
    unseen["city"], unseen["channel"] = "never_seen", "kiosk"      # unseen categories must not crash
    assert len(loaded.predict(unseen)) == len(unseen)
    assert acc > 0.65
    print("Day 13 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
