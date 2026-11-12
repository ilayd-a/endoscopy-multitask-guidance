"""
gradient_features.py
Per-frame loss gradients of the pretrained Kvasir UNet, for train and eval frames.

Each frame's gradient of the fine-tuning loss (Dice + BCE, BatchNorm in eval mode)
is stored twice: exactly for the last two decoder blocks plus the segmentation
head ("head", ~53k parameters), and as a 8192-dim count sketch of the full
24M-parameter gradient ("sketch"; inner products are preserved in expectation).

Output: results/grad_features_256.npz with
    head_train (800, P), head_eval (200, P), sketch_train (800, 8192), sketch_eval (200, 8192),
    loss_train (800,), loss_eval (200,)

Usage (from repo root):
    ./venv/bin/python quantum-medical-research/frame_selection/gradient_features.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_selection_qubo import OUT_DIR, SIZE, get_device, load_features, load_model  # noqa: E402

SKETCH_DIM = 8192


def per_frame_gradients(model, images, masks, device, head_params, sketch):
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters()]
    bucket, sign = sketch
    head, sk, losses = [], [], []
    model.eval()
    for i in range(len(images)):
        x = torch.from_numpy(images[i:i + 1]).to(device)
        y = torch.from_numpy(masks[i:i + 1]).to(device)
        out = model(x)
        loss = dice_loss(out, y) + bce(out, y)
        grads = torch.autograd.grad(loss, params)
        flat = torch.cat([g.reshape(-1) for g in grads])
        sk.append(torch.zeros(SKETCH_DIM, device=device).index_add_(0, bucket, flat * sign).cpu().numpy())
        head.append(torch.cat([g.reshape(-1) for g, p in zip(grads, params) if id(p) in head_params]).cpu().numpy())
        losses.append(float(loss))
    return np.stack(head), np.stack(sk), np.array(losses)


def main():
    device = get_device()
    _, data = load_features(device, need_eval_data=True)
    model = load_model(device)
    head_params = {id(p) for mod in (model.decoder.blocks[3], model.decoder.blocks[4], model.segmentation_head)
                   for p in mod.parameters()}
    n_par = sum(p.numel() for p in model.parameters())
    g = torch.Generator().manual_seed(0)
    sketch = (torch.randint(0, SKETCH_DIM, (n_par,), generator=g).to(device),
              (torch.randint(0, 2, (n_par,), generator=g) * 2 - 1).float().to(device))
    res = {}
    for split, img, msk in (("train", data["tr_img"], data["tr_mask"]), ("eval", data["ev_img"], data["ev_mask"])):
        t = time.time()
        res[f"head_{split}"], res[f"sketch_{split}"], res[f"loss_{split}"] = per_frame_gradients(
            model, img, msk, device, head_params, sketch)
        print(f"{split}: {len(img)} frames, head dim {res[f'head_{split}'].shape[1]}, {time.time() - t:.0f}s")
    np.savez(OUT_DIR / f"grad_features_{SIZE}.npz", **res)


if __name__ == "__main__":
    main()
