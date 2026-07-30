# Day 30 - ML System Design Interviews: Real-Time Recommendation & Visual Anomaly Detection

## Interview framework (use every time)
1. **Clarify:** objective, users, scale (QPS, catalog size, image rate), latency SLO, freshness, constraints (edge/cloud, privacy, cost).
2. **Define metrics:** business (CTR, watch time, defect escape rate) -> offline (recall@k, NDCG, PR-AUC) -> online (A/B, guardrails).
3. **Data & labels:** logging, implicit feedback bias, negative sampling, label delay, train/serve skew.
4. **Model & serving architecture**, 5. **Evaluation & rollout**, 6. **Monitoring, drift, retraining**, 7. **Failure modes & trade-offs.**

## A. Real-time recommendation system
```
events (clicks, views) -> Kafka -> stream features (Flink) -> online feature store (Redis)   \
                                 -> batch lake -> offline features / training (daily) ------> model registry
request(user) -> [candidate generation: two-tower + ANN (IVF/HNSW), 500-1000 items]
              -> [ranking: GBDT / deep ranker w/ rich features, 100-200 items]
              -> [re-rank: diversity, freshness, business rules, exploration (bandit)] -> top-k
```
- **Two-tower:** user and item encoders trained with in-batch sampled softmax, `loss = -log exp(s(u,i+)/t) / sum_j exp(s(u,j)/t)`; item embeddings precomputed, indexed in ANN; apply **logQ correction** for popularity bias in sampled negatives.
- **ANN:** IVF (k-means lists, `nprobe`), HNSW (graph, high recall), PQ (compression). Recall/latency/memory trade-off.
- **Cold start:** content features, popularity priors, exploration (epsilon-greedy / Thompson).
- **Budget (p99 100 ms):** features 15 | retrieval 10 | ranking 40 | re-rank 10 | network 15 | headroom 10.
- **Offline vs online mismatch:** counterfactual bias, position bias -> use IPS, randomized slices, interleaving, A/B.

## B. Visual anomaly detection service (manufacturing / edge)
```
camera -> preprocess (crop, normalize) -> edge model (INT8, ONNX/TensorRT) -> anomaly score + heatmap
       -> threshold (calibrated on normal data) -> {pass | reject | human review queue}
       -> uploads flagged crops + sampled normals -> labeling -> periodic retrain / threshold update
```
- Few/no defect labels: **one-class** modelling - autoencoder reconstruction error (this repo), PatchCore / PaDiM (patch-feature memory bank), student-teacher, normalizing flows.
- Threshold from normal-score percentile or from a small labeled set targeting recall at fixed false-reject rate; calibrate per line/camera.
- Drift: lighting, new part variants -> monitor score distribution (PSI/KS), trigger review.
- Edge constraints: latency per frame, fixed input size, INT8, thermal limits, offline operation, signed model updates.

## Trade-offs
| Decision | Option A | Option B |
|---|---|---|
| Retrieval | exact search (small catalog) | ANN (large, approximate) |
| Ranking | GBDT (fast, tabular) | deep net (sequence/multimodal features) |
| Anomaly | reconstruction AE (simple, can reconstruct defects too well) | patch-feature methods (stronger, heavier memory) |
| Deployment | cloud GPU (flexible) | edge (latency, privacy, bandwidth) |
| Retraining | scheduled | drift-triggered / online |

## Interview Questions
1. Design a real-time recommender for 100M users. Walk through candidate generation vs ranking. *(Funnel; two-tower + ANN; GBDT/DNN ranker; feature store.)*
2. How do you handle position bias and feedback loops? *(Randomization, IPS, exploration, holdouts.)*
3. How would you evaluate offline vs online and what guardrails do you set? *(Recall/NDCG offline; A/B on CTR with latency, diversity, revenue guardrails.)*
4. Training-serving skew: causes and prevention. *(Different code paths, time-travel leakage; shared feature definitions, point-in-time joins.)*
5. Design visual anomaly detection with only normal images. *(One-class methods, threshold selection, human-in-loop.)*
6. Your model's online metric dropped 3% after deploy. Debug. *(Data/feature pipeline diffs, shadow comparison, segment analysis, rollback criteria.)*
7. How do you serve embeddings fresh when items change? *(Streaming re-embed + incremental index updates; periodic full rebuild.)*
