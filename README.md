# machine-learning-and-deep-learning-mastery

A 30-day, from-scratch curriculum for **Machine Learning Engineer** and **AI Research Engineer** interview preparation, taking you from the mathematics of linear regression to Transformers, quantization, edge deployment and ML system design.

Every day ships two artifacts:

| File | Purpose |
|---|---|
| `day-XX/notes.md` | derivations/proofs, ASCII architecture diagrams, trade-off matrices, interview questions with answer hints |
| `day-XX/model_pipeline.py` | runnable PyTorch / scikit-learn / NumPy code that builds, trains, benchmarks and **asserts** correctness (against sklearn/PyTorch references, gradient checks, invariants) |

Each pipeline is self-contained, runs on CPU in seconds to a couple of minutes, takes `--seed` where randomness matters, and exits non-zero if a check fails.

**Interview importance tags:** `CRITICAL` = asked constantly, must be able to derive/implement; `HIGH` = frequently asked; `MEDIUM` = role/team dependent.

## 30-Day Syllabus

### Phase 1 - Classical ML Mechanics & Math from Scratch (Days 01-07)

| Day | Topic | Notes | Code | Interview importance |
|---|---|---|---|---|
| 01 | Linear regression: normal equation and gradient descent from scratch | [notes](day-01/notes.md) | [code](day-01/model_pipeline.py) | `CRITICAL` |
| 02 | Logistic and softmax regression with cross-entropy and Newton/IRLS | [notes](day-02/notes.md) | [code](day-02/model_pipeline.py) | `CRITICAL` |
| 03 | L1/L2/ElasticNet regularization with coordinate descent | [notes](day-03/notes.md) | [code](day-03/model_pipeline.py) | `CRITICAL` |
| 04 | Decision trees (CART) and random forests from scratch | [notes](day-04/notes.md) | [code](day-04/model_pipeline.py) | `CRITICAL` |
| 05 | Gradient boosting from scratch and XGBoost/LightGBM internals | [notes](day-05/notes.md) | [code](day-05/model_pipeline.py) | `CRITICAL` |
| 06 | SVMs with kernels: hinge loss, Pegasos, RBF and polynomial kernels | [notes](day-06/notes.md) | [code](day-06/model_pipeline.py) | `MEDIUM` |
| 07 | Hyperparameter tuning: grid, random, successive halving, Bayesian optimization | [notes](day-07/notes.md) | [code](day-07/model_pipeline.py) | `HIGH` |

### Phase 2 - Evaluation, Validation & Classical Production (Days 08-13)

| Day | Topic | Notes | Code | Interview importance |
|---|---|---|---|---|
| 08 | Cross-validation for leaky and time-dependent data (purged, grouped) | [notes](day-08/notes.md) | [code](day-08/model_pipeline.py) | `CRITICAL` |
| 09 | ROC-AUC vs PR-AUC for imbalanced classification | [notes](day-09/notes.md) | [code](day-09/model_pipeline.py) | `CRITICAL` |
| 10 | Probability calibration: reliability curves, Platt scaling, isotonic regression | [notes](day-10/notes.md) | [code](day-10/model_pipeline.py) | `HIGH` |
| 11 | Bias-variance decomposition with bootstrap experiments | [notes](day-11/notes.md) | [code](day-11/model_pipeline.py) | `CRITICAL` |
| 12 | Model explainability: Shapley values, LIME and permutation importance | [notes](day-12/notes.md) | [code](day-12/model_pipeline.py) | `HIGH` |
| 13 | Scikit-Learn custom transformers and leak-safe pipelines | [notes](day-13/notes.md) | [code](day-13/model_pipeline.py) | `HIGH` |

### Phase 3 - Deep Learning Foundations & PyTorch Mechanics (Days 14-20)

| Day | Topic | Notes | Code | Interview importance |
|---|---|---|---|---|
| 14 | Neural network backpropagation derivation with gradient checking | [notes](day-14/notes.md) | [code](day-14/model_pipeline.py) | `CRITICAL` |
| 15 | Custom PyTorch autograd Functions and nn.Module design | [notes](day-15/notes.md) | [code](day-15/model_pipeline.py) | `HIGH` |
| 16 | Optimizers from scratch: AdamW, Lion and warmup-cosine schedules | [notes](day-16/notes.md) | [code](day-16/model_pipeline.py) | `CRITICAL` |
| 17 | Activation functions and dead-ReLU analysis | [notes](day-17/notes.md) | [code](day-17/model_pipeline.py) | `HIGH` |
| 18 | Weight initialization: Xavier and He | [notes](day-18/notes.md) | [code](day-18/model_pipeline.py) | `HIGH` |
| 19 | Exploding and vanishing gradients: diagnosis and fixes | [notes](day-19/notes.md) | [code](day-19/model_pipeline.py) | `CRITICAL` |
| 20 | Capstone: production-grade PyTorch training harness | [notes](day-20/notes.md) | [code](day-20/model_pipeline.py) | `HIGH` |

### Phase 4 - Computer Vision & Sequential Architectures (Days 21-25)

| Day | Topic | Notes | Code | Interview importance |
|---|---|---|---|---|
| 21 | Convolutions from scratch (im2col) and CNNs | [notes](day-21/notes.md) | [code](day-21/model_pipeline.py) | `HIGH` |
| 22 | ResNet skip connections and bottleneck blocks | [notes](day-22/notes.md) | [code](day-22/model_pipeline.py) | `HIGH` |
| 23 | Object detection: IoU, NMS and a YOLO-style grid head | [notes](day-23/notes.md) | [code](day-23/model_pipeline.py) | `MEDIUM` |
| 24 | LSTM from scratch and sequence-to-sequence modeling | [notes](day-24/notes.md) | [code](day-24/model_pipeline.py) | `MEDIUM` |
| 25 | Self-attention from scratch | [notes](day-25/notes.md) | [code](day-25/model_pipeline.py) | `CRITICAL` |

### Phase 5 - Transformers, Edge ML & System Design (Days 26-30)

| Day | Topic | Notes | Code | Interview importance |
|---|---|---|---|---|
| 26 | Multi-head attention and the Transformer block | [notes](day-26/notes.md) | [code](day-26/model_pipeline.py) | `CRITICAL` |
| 27 | INT8 quantization and pruning | [notes](day-27/notes.md) | [code](day-27/model_pipeline.py) | `HIGH` |
| 28 | ONNX export and ONNX Runtime benchmarking | [notes](day-28/notes.md) | [code](day-28/model_pipeline.py) | `HIGH` |
| 29 | Edge inference: dynamic batching and TorchScript | [notes](day-29/notes.md) | [code](day-29/model_pipeline.py) | `HIGH` |
| 30 | ML system design: two-tower retrieval and visual anomaly detection | [notes](day-30/notes.md) | [code](day-30/model_pipeline.py) | `CRITICAL` |

## Environment Setup

**Python 3.10-3.12** is recommended. Create an isolated environment first:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # installs the default PyTorch build for your OS
```

**GPU (optional - nothing here requires one).** Pick the PyTorch wheel that matches your driver's CUDA runtime ([pytorch.org/get-started](https://pytorch.org/get-started/locally/) is authoritative):

```bash
# CUDA 12.4 example (driver >= 550)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1 example
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only (smallest download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Guidance: PyTorch >= 2.2 for all days; Day 28 uses `onnx` + `onnxruntime` (for NVIDIA GPUs use `onnxruntime-gpu` instead; TensorRT commands in the Day 28 notes need a TensorRT install). Optional extras that are picked up automatically: `xgboost`, `lightgbm` (Day 05) and `shap` (Day 12).

All pipelines were executed on CPU (Python 3.12, PyTorch 2.14, scikit-learn 1.9). The GPU/AMP code path on Day 20 and the optional xgboost/lightgbm/shap comparisons were not exercised.

## Running

```bash
python day-01/model_pipeline.py            # a single day
python run_all.py                          # all 30 days with pass/fail summary
python run_all.py --days 14 15 16          # a subset
```

## Repository Layout

```
machine-learning-and-deep-learning-mastery/
|-- README.md  requirements.txt  run_all.py  orchestrator.py  .gitignore
|-- day-01/ ... day-30/
|   |-- notes.md
|   `-- model_pipeline.py
```

## ML Engineering Interview Cheatsheet (high priority)

**Math you must be able to write on a whiteboard**
- Linear regression: `theta = (X^T X)^-1 X^T y`, gradient `(2/n) X^T (X theta - y)`. Logistic/softmax gradient: `X^T (p - y) / n`.
- Ridge: `(X^T X + lambda I)^-1 X^T y`; L1 gives sparsity (Laplace prior), L2 shrinks (Gaussian prior).
- Bias-variance: `E[(y - f_hat)^2] = Bias^2 + Variance + sigma^2`.
- Backprop: `delta_l = (W_{l+1}^T delta_{l+1}) * f'(z_l)`, `dW_l = a_{l-1}^T delta_l`; softmax+CE gives `P - Y`.
- Attention: `softmax(QK^T / sqrt(d_k)) V`; the scale keeps logit variance ~1.
- Adam: `m/(1-b1^t) / (sqrt(v/(1-b2^t)) + eps)`; AdamW decouples weight decay.
- He init `Var(w) = 2/n_in` (ReLU); Xavier `2/(n_in+n_out)` (tanh/linear).

**Evaluation reflexes**
- Imbalanced data -> PR-AUC / precision@k, not accuracy or ROC-AUC alone; random PR-AUC = prevalence.
- Time or grouped data -> never shuffled KFold; use time-series / group / purged splits; fit every transformer inside the CV fold.
- ROC-AUC measures ranking, not calibration: calibrate (Platt / isotonic / temperature) when probabilities drive decisions.
- Tuning on the test set or reusing CV folds for tuning *and* reporting inflates results; use nested CV or a final holdout.

**Model selection heuristics**
- Tabular -> gradient-boosted trees first (XGBoost/LightGBM), strong baselines before deep nets.
- Small data + interpretability -> regularized linear models. Images -> pretrained CNN/ViT fine-tuning. Sequences/text -> Transformers.
- Edge latency/memory -> distill, prune structurally, INT8 (PTQ then QAT), export ONNX/TFLite/TensorRT.

**Debugging a net that will not train**
1. Overfit a single batch. 2. Check loss at init (`ln K`). 3. Verify labels/shuffling/normalization. 4. Per-layer gradient norms. 5. Lower LR / add warmup / clip. 6. Check train vs eval mode.

**Serving & systems**
- Report p50/p95/p99, not means. Dynamic batching trades tail latency for throughput. Prevent train/serve skew with one serialized pipeline and point-in-time features.
- System-design template: clarify -> metrics -> data/labels -> model & serving -> evaluation/rollout -> monitoring/drift -> trade-offs (see Day 30).

**Rapid-fire facts**
- Bagging reduces variance, boosting reduces bias. OOB fraction ~ 36.8%.
- Dropout/BN behave differently in train vs eval. Residuals give an identity gradient path.
- KV cache: decode cost per token O(T d); memory grows with context. GQA/MQA shrink the cache.
- Post-hoc monotone calibration does not change AUC. Permutation importance understates correlated features.
