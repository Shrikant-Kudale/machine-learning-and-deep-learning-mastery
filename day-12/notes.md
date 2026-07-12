# Day 12 - Model Explainability: SHAP, LIME, Permutation Importance

## Shapley values
```
phi_i = sum_{S subset F\{i}} |S|!(d-|S|-1)!/d!  [ v(S u {i}) - v(S) ]
```
Axioms: **efficiency** (`f(x) = phi_0 + sum phi_i`), symmetry, dummy, additivity. Exact cost is O(2^d), so use TreeSHAP (poly time for trees), KernelSHAP (sampling), DeepSHAP/Integrated Gradients (nets).
Value function choice matters: *interventional* (break correlations, uses background) vs *conditional*.

## LIME
Sample perturbations z around x, weight by proximity kernel pi_x(z), fit sparse linear model g minimizing `sum pi_x(z)(f(z)-g(z))^2 + Omega(g)`. Local only; unstable w.r.t. kernel width and sampling.

## Permutation importance
Increase in loss after shuffling one feature on held-out data. Model-agnostic; biased with correlated features (shuffling creates unrealistic rows).

```
black-box f ->  global: permutation imp., SHAP summary, PDP/ICE
            ->  local : SHAP force/waterfall, LIME, counterfactuals
```
## Trade-offs
| Method | Scope | Consistent | Cost | Caveat |
|---|---|---|---|---|
| Impurity importance | global | no | free | high-cardinality bias |
| Permutation | global | yes-ish | n_feat x predict | correlated features |
| TreeSHAP | local+global | yes | fast | trees only |
| KernelSHAP | local | yes | slow | background choice |
| LIME | local | no | medium | unstable |

## Interview Questions
1. State the Shapley axioms. *(Efficiency, symmetry, dummy, additivity.)*
2. Why is SHAP not causal? *(It explains the model, not the world; correlated features share credit.)*
3. SHAP vs LIME? *(Axiomatic and consistent vs heuristic local surrogate.)*
4. Explanations for a regulated credit model? *(Monotonic constraints, reason codes via SHAP, stability audits.)*
5. How do you validate an explanation? *(Deletion/insertion curves, sanity checks, randomization tests.)*
6. Permutation importance with correlated features? *(Underestimates; use grouped permutation or conditional methods.)*
