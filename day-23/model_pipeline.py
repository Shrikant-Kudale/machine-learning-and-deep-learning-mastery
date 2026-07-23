"""Day 23 - Object detection paradigms: IoU/GIoU, NMS, anchor-free YOLO-style grid head trained on synthetic boxes."""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG, S = 32, 4                                      # image size, S x S grid (stride 8)


def box_iou(a, b):
    """a: (N,4), b: (M,4) in xyxy -> (N,M)."""
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda t: (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def giou(a, b):
    """Generalized IoU for aligned pairs (N,4),(N,4): IoU - (C - U)/C."""
    lt, rb = torch.max(a[:, :2], b[:, :2]), torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    aa, ab = (a[:, 2:] - a[:, :2]).prod(-1), (b[:, 2:] - b[:, :2]).prod(-1)
    union = aa + ab - inter
    c = (torch.max(a[:, 2:], b[:, 2:]) - torch.min(a[:, :2], b[:, :2])).prod(-1)
    return inter / union - (c - union) / c


def nms(boxes, scores, thr=0.5):
    order, keep = scores.argsort(descending=True), []
    while len(order):
        i = order[0].item(); keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][box_iou(boxes[i:i + 1], boxes[order[1:]])[0] <= thr]
    return keep


def make_batch(n, gen):
    """One bright square per image at random position/size on noise."""
    imgs = torch.rand(n, 1, IMG, IMG, generator=gen) * 0.2
    wh = torch.randint(8, 16, (n, 2), generator=gen).float()
    xy = torch.stack([torch.randint(0, IMG - int(w), (1,), generator=gen).float()[0] for w in wh[:, 0]]), \
         torch.stack([torch.randint(0, IMG - int(h), (1,), generator=gen).float()[0] for h in wh[:, 1]])
    boxes = torch.stack([xy[0], xy[1], xy[0] + wh[:, 0], xy[1] + wh[:, 1]], 1)
    for i, b in enumerate(boxes.int()):
        imgs[i, 0, b[1]:b[3], b[0]:b[2]] += 0.8
    return imgs, boxes


class TinyYOLO(nn.Module):
    """Backbone -> S x S x 5 grid: (objectness, dx, dy, log w, log h). Anchor-free, center-cell assignment."""

    def __init__(self):
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.net = nn.Sequential(*c(1, 16), *c(16, 32), *c(32, 64), nn.Conv2d(64, 5, 1))

    def forward(self, x):
        return self.net(x)                                       # (N,5,S,S)


def encode(boxes):
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    cell = IMG / S
    gx, gy = (cx // cell).long().clamp(0, S - 1), (cy // cell).long().clamp(0, S - 1)
    t = torch.stack([cx / cell - gx, cy / cell - gy, torch.log((boxes[:, 2] - boxes[:, 0]) / IMG),
                     torch.log((boxes[:, 3] - boxes[:, 1]) / IMG)], 1)
    return gx, gy, t


def yolo_loss(out, boxes):
    n = len(boxes)
    gx, gy, t = encode(boxes)
    obj_t = torch.zeros(n, S, S)
    obj_t[torch.arange(n), gy, gx] = 1
    l_obj = F.binary_cross_entropy_with_logits(out[:, 0], obj_t, pos_weight=torch.tensor(float(S * S) / 2))
    pred = out[torch.arange(n), :, gy, gx]                       # (n,5)
    l_box = F.mse_loss(torch.sigmoid(pred[:, 1:3]), t[:, :2]) + F.mse_loss(pred[:, 3:5], t[:, 2:])
    return l_obj + 5 * l_box


def decode(out):
    n = out.shape[0]
    obj = torch.sigmoid(out[:, 0]).view(n, -1)
    idx = obj.argmax(1)
    gy, gx = idx // S, idx % S
    p = out[torch.arange(n), :, gy, gx]
    cell = IMG / S
    cx, cy = (gx + torch.sigmoid(p[:, 1])) * cell, (gy + torch.sigmoid(p[:, 2])) * cell
    w, h = torch.exp(p[:, 3]) * IMG, torch.exp(p[:, 4]) * IMG
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def main(seed=0):
    torch.manual_seed(seed)
    a, b = torch.tensor([[0., 0, 10, 10]]), torch.tensor([[5., 5, 15, 15]])
    assert abs(box_iou(a, b).item() - 25 / 175) < 1e-6
    assert giou(a, b).item() < box_iou(a, b).item()                     # GIoU penalizes enclosing-box slack
    bx = torch.tensor([[0., 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    assert nms(bx, torch.tensor([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    print("IoU / GIoU / NMS unit checks passed.")

    gen = torch.Generator().manual_seed(seed)
    Xtr, Btr = make_batch(1500, gen)
    Xte, Bte = make_batch(300, gen)
    net = TinyYOLO()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for ep in range(25):
        net.train()
        for idx in torch.randperm(len(Xtr)).split(64):
            opt.zero_grad(); yolo_loss(net(Xtr[idx]), Btr[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = decode(net(Xte))
    ious = torch.stack([box_iou(pred[i:i + 1], Bte[i:i + 1])[0, 0] for i in range(len(pred))])
    print(f"mean IoU={ious.mean():.3f}  AP@0.5-style hit rate={(ious > 0.5).float().mean():.3f}")
    assert (ious > 0.5).float().mean() > 0.7
    print("Day 23 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
