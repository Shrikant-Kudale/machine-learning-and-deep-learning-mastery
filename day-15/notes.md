# Day 15 - Custom Autograd & nn.Module Design

## How autograd works
Every tensor op with `requires_grad` records a node (`grad_fn`) in a DAG. `loss.backward()` topologically sorts the graph and calls each node's `backward` with the upstream gradient (vector-Jacobian product). Gradients **accumulate** into `.grad`, hence `optimizer.zero_grad()`.

```
x --\
     (*) --> (+) --> tanh --> loss        backward: topo-order reversed,
a --/         ^                            grad[parent] += local_jacobian * grad[child]
b ------------'
```
## Custom Function rules
- `forward(ctx, ...)` / `backward(ctx, grad_out)`; stash with `ctx.save_for_backward`; return one gradient per input (`None` for non-differentiable).
- Validate with `torch.autograd.gradcheck` in float64.
- Use when: numerically stable fused ops, non-differentiable ops with surrogate gradients (straight-through estimators), memory-saving recompute.

## nn.Module design
| Practice | Why |
|---|---|
| Sub-modules as attributes / `ModuleList` | registered -> `.parameters()`, `.to(device)`, state_dict |
| `register_buffer` for non-trainable state | moves with device, saved, no grad |
| Config in `__init__`, pure `forward` | testable, scriptable, exportable |
| Pre-norm residual blocks | stable gradients in deep nets |
| `model.eval()` + `torch.no_grad()` at inference | dropout/BN behave correctly, no graph memory |

## Interview Questions
1. Why does PyTorch accumulate gradients? *(Supports multi-loss and gradient accumulation for large effective batches.)*
2. `detach()` vs `no_grad()`? *(Detach cuts one tensor from the graph; no_grad disables graph recording in a scope.)*
3. Parameter vs buffer? *(Learnable vs persistent non-learnable state.)*
4. In-place ops and autograd? *(Can corrupt saved tensors -> version-counter error.)*
5. Implement a straight-through estimator. *(Forward: round; backward: identity.)*
6. How does gradient checkpointing work? *(Drop activations, recompute in backward: ~33% more compute, big memory saving.)*
