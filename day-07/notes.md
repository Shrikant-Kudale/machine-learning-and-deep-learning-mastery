# Day 07 - Hyperparameter Tuning Algorithms

## Methods
```
Grid search      : exhaustive Cartesian product, cost k^d
Random search    : sample uniformly (Bergstra & Bengio 2012) - better coverage of important dims
Successive halving / Hyperband : many configs, tiny budget -> promote top 1/eta with eta x budget
Bayesian opt     : surrogate p(f|D) (GP or TPE) + acquisition function
```
**Expected Improvement:** `EI(x) = (mu - f*) Phi(z) + sigma phi(z)`, `z = (mu - f*)/sigma`; balances exploitation (high mu) and exploration (high sigma).

**Why random beats grid:** if only 1 of d dims matters, n random trials give n distinct values on it; grid gives n^(1/d).

## Pipeline
```
search space (log-scaled lr, int depth) -> sampler -> CV objective -> [prune | update surrogate] -> best cfg -> refit -> untouched holdout
```
## Trade-offs
| Method | Parallel | Sample-efficient | Notes |
|---|---|---|---|
| Grid | yes | no | only for <=3 params |
| Random | yes | low | strong baseline |
| Halving/Hyperband | yes | medium | needs monotone budget (epochs/trees/data) |
| Bayesian (GP) | limited | high | GP scales O(n^3); TPE (Optuna) for many/cat params |

## Interview Questions
1. Why search learning rate on a log scale? *(Effect is multiplicative.)*
2. How do you avoid overfitting the validation set during tuning? *(Nested CV or a final untouched test set.)*
3. When does Bayesian optimization fail? *(High dims, noisy objective, conditional spaces with GP.)*
4. Hyperband vs BO? *(Budget allocation vs smart sampling; BOHB combines them.)*
5. Early stopping as tuning? *(Treat n_estimators as a free parameter via validation curve.)*
6. How would you tune at scale? *(Ray Tune / Optuna distributed, ASHA scheduler.)*
