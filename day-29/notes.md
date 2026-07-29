# Day 29 - Edge Inference & Microservice Serving

## Latency model
`service(b) = a + c*b` (fixed launch/overhead + per-sample cost). Throughput at batch b: `b / (a + c b)` -> approaches `1/c`. Batching amortizes `a`; it adds queueing delay up to `max_wait`. Choose `(max_batch, max_wait)` so that `p99 = max_wait + service(max_batch) + queue` meets the SLO.

```
clients -> LB -> [ API gateway: auth, validation ]
                      -> [ dynamic batcher: queue, max_batch, max_wait ]
                            -> [ worker: warm model (ORT/TensorRT), pinned threads, fp16/int8 ]
                                 -> [ postprocess / NMS ] -> response
metrics: p50/p95/p99, queue depth, batch size histogram, GPU util, error rate, drift
```
## Edge / embedded checklist
| Concern | Tactic |
|---|---|
| Memory (MBs) | INT8/INT4, pruning, distillation, shared weights |
| Latency | operator fusion, depthwise convs, avoid dynamic shapes, warm-up |
| Power | NPU/DSP delegates, lower resolution, duty cycling |
| Updates | signed model artifacts, A/B + rollback, delta updates |
| Runtime | TFLite/ExecuTorch/ORT-mobile/TensorRT/OpenVINO; pin threads, fixed input sizes |
| Streaming | stateful RNN/SSM models, ring buffers, frame skipping |
| Offline | on-device logging, privacy-preserving telemetry |

**Serving options:** Triton Inference Server (dynamic batching, ensembles, multi-framework), TorchServe, KServe/Ray Serve (autoscale), FastAPI + ORT for small services. **Cold-start:** pre-load + warm-up requests; keep min replicas.

## Interview Questions
1. How does dynamic batching work, and what are its knobs? *(max batch, max queue delay; SLO-driven.)*
2. Throughput vs latency trade-off? *(Larger batches raise throughput, increase tail latency.)*
3. How would you deploy a vision model on a 512 MB device? *(Quantize + prune/distill, ORT-mobile/TFLite, fixed shapes, profile on device.)*
4. How do you monitor an edge fleet? *(Heartbeat + sampled inputs/outputs, on-device metrics, shadow mode, staged rollout.)*
5. Why pin threads / set `torch.set_num_threads`? *(Avoid oversubscription across workers.)*
6. Diagnose p99 spikes. *(GC, batch timeouts, cold caches, noisy neighbours, large inputs, NMS.)*
