# Day 20 - Capstone: A Production-Grade Training Loop

## Checklist implemented
```
data split (stratified train/val/test) -> model (He init) -> AdamW (no decay on bias/norm)
 -> warmup + cosine LR per step -> forward under autocast -> GradScaler -> unscale -> clip_grad_norm_
 -> optimizer step -> validate each epoch -> early stopping on val loss -> restore best weights
 -> checkpoint {model, optimizer, epoch, best, patience} every epoch -> resume
```
## Engineering decisions
| Decision | Reason |
|---|---|
| Seeds + seeded shuffles | reproducibility; debugging |
| Unscale before clipping | clip true gradient magnitudes under AMP |
| Checkpoint optimizer + scheduler state | exact resume (Adam moments) |
| `model.eval()` + `inference_mode()` | correct dropout/norm, no autograd overhead |
| Early stopping on val, not test | avoid test leakage |
| `zero_grad(set_to_none=True)` | faster, less memory |

**Throughput levers:** larger batch, `DataLoader(num_workers, pin_memory, persistent_workers)`, bf16 autocast, `torch.compile`, fused AdamW, channels_last (CNNs).
**Scaling beyond one GPU:** DDP (data parallel, all-reduce grads), FSDP/ZeRO (shard params/grads/optimizer), gradient accumulation to simulate large batches.

## Sanity-test ladder
1. Overfit 1 batch to ~0 loss. 2. Loss at init ~ `ln(K)`. 3. Gradients non-zero, finite. 4. Val metric tracks train initially. 5. Resume == uninterrupted.

## Interview Questions
1. Loss at initialization for K-class CE? *(ln K.)*
2. Gradient accumulation with BatchNorm? *(BN stats per micro-batch; use GroupNorm/LN or SyncBN.)*
3. bf16 vs fp16 AMP? *(bf16 has fp32 range, no loss scaling; fp16 needs GradScaler.)*
4. How to make training deterministic on GPU? *(Seeds, `torch.use_deterministic_algorithms`, fixed cuDNN settings; some speed cost.)*
5. DDP vs FSDP? *(Replicated vs sharded states; FSDP for models that don't fit.)*
6. Training is slower than expected - where do you look? *(Data loader stalls, small batches, CPU-GPU syncs, profile with torch.profiler.)*
