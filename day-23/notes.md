# Day 23 - Object Detection Paradigms & YOLO

## Paradigms
| Family | Examples | Idea | Trade-off |
|---|---|---|---|
| Two-stage | R-CNN, Fast/Faster R-CNN, Mask R-CNN | region proposals (RPN) then classify/refine | accurate, slower |
| One-stage anchor | SSD, RetinaNet, YOLOv2-v5 | dense predictions on anchors | fast; class imbalance -> focal loss |
| Anchor-free | FCOS, CenterNet, YOLOX/v8 | predict center/ltrb per location | simpler, fewer hyperparameters |
| Transformer | DETR, RT-DETR | set prediction + Hungarian matching, no NMS | slower convergence (fixed in variants) |

## YOLO architecture
```
image -> backbone (CSP/ResNet-like) -> neck (FPN+PAN, multi-scale) -> heads at strides 8/16/32
each cell: [objectness, box (dx,dy,w,h), class scores]
inference: decode -> score filter -> NMS -> boxes
```
**Box encoding:** `cx = (gx + sigma(tx)) * stride`, `w = anchor_w * exp(tw)` (or direct anchor-free).
**Loss:** `L = l_obj (BCE) + l_cls (BCE/focal) + l_box (IoU-family: IoU, GIoU, DIoU, CIoU)`; label assignment (SimOTA / TAL) is a big accuracy lever.
**IoU** = `|A n B| / |A u B|`; **GIoU** = `IoU - |C \ (A u B)|/|C|` gives gradient even when boxes do not overlap.
**NMS:** sort by score, keep best, drop boxes with IoU > thr; Soft-NMS decays scores instead.
**mAP:** per-class AP = area under PR curve at IoU thresholds (COCO: mean over 0.5:0.95).

## Edge notes
Latency budget = preprocess + inference + NMS (NMS is CPU-heavy: export with fused/NMS-free heads); INT8 quantization of backbone (Day 27); input size trades small-object recall for speed.

## Interview Questions
1. Why is IoU loss better than L2 on coordinates? *(Scale-invariant, matches metric.)*
2. Why focal loss in dense detectors? *(~100k easy negatives dominate gradients.)*
3. How is NMS a problem in crowded scenes? *(Suppresses true neighbours; use Soft-NMS/DIoU-NMS or end-to-end detectors.)*
4. How do you detect small objects? *(Higher resolution, FPN/P2 head, tiling, more stride-8 anchors.)*
5. Anchors: how chosen? *(k-means on training boxes with IoU distance, or go anchor-free.)*
6. Explain mAP. *(Above.)*
