# Day 22 - ResNet & Skip Connections

## Idea
Learn a residual `F(x) = H(x) - x`: `y = x + F(x)`. If identity is optimal, push F -> 0 (easier than fitting identity with stacked nonlinear layers). He et al. observed the **degradation problem**: deeper *plain* nets have higher *training* error - an optimization, not overfitting, issue.

## Gradient proof
`x_{l+1} = x_l + F(x_l)`  =>  `x_L = x_l + sum_{i=l}^{L-1} F(x_i)`
`dL/dx_l = dL/dx_L (1 + d/dx_l sum F(x_i))` - the `1` provides a direct path for gradients; vanishing is avoided.

```
        +----------------------+
 x ---> | conv-BN-ReLU-conv-BN | ---(+)---> ReLU ---> out        BasicBlock
 |      +----------------------+     ^
 '--------- identity / 1x1 proj -----'

 Bottleneck: 1x1 (C->C/4) -> 3x3 -> 1x1 (C/4->C): ~8x fewer FLOPs than two 3x3 at width C
```
## Design details
- Projection shortcut (1x1 conv, stride) when shape changes; identity otherwise.
- Zero-init last BN gamma in each block -> network starts as shallow.
- Pre-activation ResNet (BN-ReLU-conv) gives cleaner identity path; ResNeXt (groups), Wide-ResNet (width > depth), DenseNet (concat), SE (channel attention).

## Trade-offs
| Model | Params | Top-1 (ImageNet) | Note |
|---|---|---|---|
| ResNet-18 | 11.7M | ~70% | edge-friendly |
| ResNet-50 | 25.6M | ~76-80% | strong baseline backbone |
| MobileNetV3 / EfficientNet | 5-20M | 75-80% | depthwise convs, NAS |

## Interview Questions
1. Why do residuals ease optimization? *(Identity gradient path; loss landscape smoother.)*
2. What happens if input/output channels differ? *(1x1 projection shortcut.)*
3. Bottleneck blocks - why? *(Reduce FLOPs while keeping depth/width.)*
4. Relationship to LSTM gates / highway nets? *(Gated vs ungated skip.)*
5. Why do transformers use residuals around attention/MLP? *(Same reasons; plus pre-LN stability.)*
6. How do you fine-tune a ResNet on a small dataset? *(Freeze early layers, low LR for backbone, BN in eval or frozen.)*
