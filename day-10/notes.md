# Day 10 - Calibration Curves

## Definition
A model is calibrated if `P(y=1 | p_hat = p) = p` for all p. Accuracy/AUC say nothing about this (AUC is invariant to monotone transforms of the score).

## Measures
```
Reliability diagram: bin predictions, plot mean prediction vs observed frequency (diagonal = perfect)
ECE   = sum_b (n_b/n) |acc_b - conf_b|
Brier = mean (p - y)^2 = calibration + refinement (Murphy decomposition)
Log loss penalizes confident errors heavily
```
```
observed
 1 |            .*   over-confident (S-shaped below diagonal at high end)
   |        . '  /
   |     .*    /  <- ideal
   |  .'     /
 0 +----------- predicted
```
## Fixes (post-hoc, fit on held-out data)
- **Platt:** `p = sigma(a s + b)`, 2 params, good for small data / sigmoid distortion (SVM, boosting).
- **Isotonic:** monotone step function, flexible, needs ~1000+ calibration points, can overfit.
- **Temperature scaling:** `softmax(z/T)`, single scalar, standard for neural nets.

## Trade-offs
| Model | Typical miscalibration |
|---|---|
| Naive Bayes | extreme (over-confident) |
| Random Forest / SVM | pushed away from 0 and 1 (under-confident at extremes) |
| Boosting | sigmoid-shaped distortion |
| Logistic regression | usually well calibrated |
| Deep nets | over-confident |

## Interview Questions
1. Why does calibration matter if AUC is high? *(Thresholding on cost, expected-value decisions, risk scores, ensembling.)*
2. Platt vs isotonic? *(Parametric vs non-parametric; data size.)*
3. Why calibrate on separate data? *(Fitting on train scores reuses already-overfit outputs.)*
4. Does calibration change AUC? *(Monotone maps keep ranking; AUC unchanged.)*
5. Calibration under class re-weighting or subsampling? *(Prior shift; correct via logit adjustment: add log(pi_true/pi_train).)*
6. Multiclass calibration? *(Temperature scaling, Dirichlet calibration, one-vs-rest.)*
