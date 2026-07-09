# Day 09 - ROC-AUC vs PR-AUC

## Definitions
```
TPR = TP/(TP+FN)  (recall)     FPR = FP/(FP+TN)     Precision = TP/(TP+FP)
ROC: TPR vs FPR over thresholds     PR: precision vs recall over thresholds
ROC-AUC = P(s(x+) > s(x-))     (Mann-Whitney U / rank statistic)
AP      = sum_n (R_n - R_{n-1}) P_n
```
## Why they diverge
FPR divides FP by the (huge) number of negatives, so 1000 false positives barely move FPR on 1M negatives, but precision (TP/(TP+FP)) drops sharply. Random classifier: ROC-AUC = 0.5 always; PR-AUC = prevalence.

```
precision
 1 |*.
   |  '*.            good model
   |      '--.__
 pi|- - - - - - '--*---- random (= prevalence pi)
   +---------------------- recall
```
## Choosing
| Situation | Metric |
|---|---|
| Balanced, care about ranking overall | ROC-AUC |
| Rare positives, FP costly (fraud, defects) | PR-AUC, precision@k, recall@fixed-precision |
| Need a decision | pick threshold by cost matrix / F-beta |
| Need probabilities | log loss, Brier, calibration (Day 10) |

## Interview Questions
1. Why can ROC-AUC be misleading with 0.1% positives? *(Large TN pool deflates FPR.)*
2. What is the interpretation of ROC-AUC? *(Probability a random positive outranks a random negative.)*
3. How do you pick an operating threshold? *(Cost-sensitive analysis or constraint, e.g. precision >= 0.9.)*
4. Is AUC threshold-independent? Is that always good? *(Yes; but production uses one threshold, so evaluate there too.)*
5. Do metrics change if you subsample negatives? *(PR-AUC does; ROC-AUC does not.)*
6. Macro vs micro averaging for multi-class? *(Macro treats classes equally, micro weights by frequency.)*
