# Day 08 - Cross-Validation for Leaky & Time-Dependent Data

## Schemes
```
KFold / Stratified : i.i.d. rows; stratified keeps class ratios
GroupKFold         : all rows of an entity (user, patient, device) stay in one fold
TimeSeriesSplit    : train on [0,t), test on [t,t+h) - expanding window
Purged K-Fold      : drop train rows whose label window overlaps test; add embargo after test
Nested CV          : inner loop tunes, outer loop estimates generalization
```
```
time ------------------------------------------------------------>
|  train  |purge| TEST |embargo|        train         |
```
## Leakage taxonomy
Target leakage (feature derived from label), train/test contamination (scaler fit on all data), group leakage, temporal leakage (future info; overlapping label horizons), duplicate rows.

## Rule
Anything with `.fit()` (imputer, scaler, encoder, selector) goes **inside** the CV loop (use `Pipeline`).

## Trade-offs
| Scheme | Bias of estimate | Variance | Use |
|---|---|---|---|
| shuffled KFold on time data | optimistic | low | never |
| TimeSeriesSplit | realistic | high (few folds) | forecasting |
| Purged + embargo | realistic | medium | finance, overlapping labels |
| GroupKFold | realistic | medium | user/patient data |

## Interview Questions
1. Why is random KFold wrong for time series? *(Autocorrelation leaks neighbours across the split.)*
2. Online score is much lower than CV; list causes. *(Leakage, drift, selection bias, train/serve skew.)*
3. When do you need an embargo? *(Labels computed from future windows.)*
4. How do you CV with heavy class imbalance? *(Stratify; evaluate with PR-AUC; repeated CV.)*
5. Why nested CV? *(Tuning on the same folds inflates the estimate.)*
6. Leave-one-group-out vs GroupKFold? *(LOGO has more variance and cost; fine for few groups.)*
