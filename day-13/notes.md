# Day 13 - Scikit-Learn Custom Transformers & Production Pipelines

## Contract
`fit(X, y)` learns state (stored with trailing underscore attrs) and returns `self`; `transform(X)` is pure/stateless; inherit `BaseEstimator` (get/set_params, clone) + `TransformerMixin` (fit_transform). `__init__` only stores params.

## Architecture
```
raw DataFrame
   |-- num cols -> Imputer -> QuantileClipper -> Scaler ----\
   |-- low-card cat -> OneHot(handle_unknown=ignore) --------+--> HistGBM -> predict_proba
   '-- high-card cat -> OOF TargetEncoder (leak-safe) -------/
   whole Pipeline is one artifact: joblib.dump -> serve (train/serve parity)
```
## Key decisions
- Fit preprocessing inside `Pipeline` so CV refits it per fold (no leakage).
- Target encoding: out-of-fold in `fit_transform`, full-train stats in `transform`, smoothing toward the prior.
- Unseen categories: `handle_unknown="ignore"` / prior fallback.
- Version-pin sklearn at serve time; pickles are not portable across versions. Consider ONNX (Day 28) for portability.

## Trade-offs
| Encoding | Cardinality | Leakage risk | Note |
|---|---|---|---|
| One-hot | low | none | sparse, wide |
| Ordinal | any (trees) | none | imposes order |
| Target (OOF) | high | managed | needs smoothing |
| Hashing | very high | none | collisions |
| Embeddings | very high | none | needs NN |

## Interview Questions
1. Why must scalers be fit only on training data? *(Test statistics leak distribution info.)*
2. Why OOF target encoding? *(In-sample means encode the row's own label.)*
3. How do you guarantee train/serve parity? *(Single serialized pipeline, shared feature code, schema validation.)*
4. What does `clone` require of a transformer? *(Constructor params stored verbatim, no logic in `__init__`.)*
5. How do you handle drift in a pipeline? *(Monitor feature stats, PSI, retrain triggers.)*
6. `ColumnTransformer` remainder behaviour? *(Dropped by default; `passthrough` to keep.)*
