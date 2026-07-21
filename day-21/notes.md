# Day 21 - Convolutional Networks & Spatial Convolutions

## Math
Output size: `H_out = floor((H + 2p - d(k-1) - 1)/s) + 1`.
Cross-correlation: `y[o,i,j] = b_o + sum_{c,u,v} W[o,c,u,v] x[c, si+u, sj+v]`.
**im2col:** unfold each receptive field into a column; conv = `W.view(O, C*k*k) @ cols` - one GEMM (how cuDNN-style implementations are reasoned about).
Params: `O*C*k*k + O`, independent of image size (**weight sharing**); FLOPs `~ O*C*k*k*H_out*W_out`.
Receptive field: `r_l = r_{l-1} + (k_l - 1) * prod_{i<l} s_i`.
Backward: grad wrt input is a "full" convolution with the flipped kernel; grad wrt weights is a correlation of input with upstream grad.

```
input CxHxW --[3x3 conv, pad1]--> OxHxW --BN--ReLU--[maxpool2]--> Ox(H/2)x(W/2) --> GAP --> FC
   local connectivity + shared weights + translation equivariance
```
## Variants
| Op | Params | Use |
|---|---|---|
| Depthwise separable | `C k^2 + C O` | MobileNet, edge |
| Dilated | same, larger RF | segmentation |
| 1x1 conv | `C O` | channel mixing, bottlenecks |
| Strided conv vs pooling | learnable vs fixed downsampling | |
| Transposed conv | | upsampling (checkerboard risk; prefer resize+conv) |

## Interview Questions
1. Conv layer parameters and FLOPs? *(Formulas above.)*
2. Equivariance vs invariance? *(Conv: equivariant to translation; pooling/GAP adds approximate invariance.)*
3. Why 3x3 stacks instead of 7x7? *(Same RF, fewer params, more nonlinearity.)*
4. BatchNorm in train vs eval? *(Batch stats vs running stats; common source of train/serve skew.)*
5. How does a depthwise-separable conv save compute? *(~k^2 fold reduction for large O.)*
6. Why can CNNs fail under 1-pixel shifts? *(Aliasing from strided ops.)*
