import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing

# --- Import your modules ---
from model import FINNCompatibleGHM_MultiOutput
from dataset import GraspNetHeatmapDataset

# --- Configuration ---
GRASPNET_ROOT = "data/graspnet"
CAMERA = 'kinect'
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_TRAIN_IMAGES = 10000  # Set to -1 to use full dataset
SAVE_PATH = "trained_model_hggd.pth"

# ==============================================================================
# 1. HGGD Loss Function (Paper Section IV-D)
# ==============================================================================
def hggd_loss(preds, targets, loc_a=1.0, reg_b=5.0, cls_c=1.0):
    pred_loc, pred_cls, pred_theta, pred_width, pred_depth = preds
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets
    device, eps = pred_loc.device, 1e-6

    # --- A. Location Loss (Penalty-Reduced Focal Loss) ---
    pred_loc = torch.clamp(torch.sigmoid(pred_loc), eps, 1 - eps)

    # [cite_start]Critical: Only exact centers (1.0) are positive [cite: 10]
    pos_inds = gt_loc.eq(1).float()
    neg_inds = gt_loc.lt(1).float()
    neg_weights = torch.pow(1 - gt_loc, 4)

    loss_pos = torch.log(pred_loc) * torch.pow(1 - pred_loc, 2) * pos_inds
    loss_neg = torch.log(1 - pred_loc) * torch.pow(pred_loc, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    loss_loc = -(loss_pos.sum() + loss_neg.sum()) / (num_pos if num_pos > 0 else 1)

    # --- B. Regression Loss (Masked Smooth L1) ---
    mask = pos_inds.expand_as(pred_theta)
    loss_reg = torch.tensor(0.0, device=device)

    if mask.sum() > 0:
        l_theta = F.smooth_l1_loss(pred_theta[mask>0], gt_theta[mask>0])
        l_width = F.smooth_l1_loss(pred_width[mask>0], gt_width[mask>0])
        l_depth = F.smooth_l1_loss(pred_depth[mask>0], gt_depth[mask>0])
        loss_reg = l_theta + l_width + l_depth

    # --- C. Anchor Classification Loss (Focal Loss) ---
    pred_cls = torch.clamp(torch.sigmoid(pred_cls), eps, 1 - eps)
    alpha, gamma = 0.25, 2.0

    bce = -(gt_cls * torch.log(pred_cls) + (1 - gt_cls) * torch.log(1 - pred_cls))
    focal_weight = alpha * torch.pow(torch.abs(gt_cls - pred_cls), gamma)
    loss_cls = (focal_weight * bce).mean()

    return (loc_a * loss_loc) + (cls_c * loss_cls) + (reg_b * loss_reg), loss_loc, loss_cls, loss_reg

# ==============================================================================
# 2. Bias Init & Viz
# ==============================================================================
def init_heatmap_bias(model):
    """Init bias to -4.59 (prob=0.01) to prevent loss explosion in epoch 1."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            # Heuristic: Detect final heads by kernel size 1 or name
            if m.kernel_size == (1, 1):
                nn.init.constant_(m.bias, -4.59)

# ==============================================================================
# 3. Main Loop
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")

    # Data & Model
    full_ds = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)

    # Use NUM_TRAIN_IMAGES or full dataset
    limit = NUM_TRAIN_IMAGES if NUM_TRAIN_IMAGES > 0 else len(indices)
    train_ds = Subset(full_ds, indices[:limit])

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=min(8, os.cpu_count()-1), pin_memory=True)

    model = FINNCompatibleGHM_MultiOutput(in_channels=4, num_angles=6).to(device)
    init_heatmap_bias(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        ep_loss = 0
        for i, (x, targets) in enumerate(loader):
            x = x.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            preds = model(x)
            loss, l_loc, l_cls, l_reg = hggd_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            ep_loss += loss.item()
            if i % 50 == 0:
                print(f"Ep {epoch+1} | B {i} | L: {loss.item():.4f} (Loc:{l_loc:.3f} Cls:{l_cls:.3f} Reg:{l_reg:.3f})")

        print(f"--- Epoch {epoch+1} Avg Loss: {ep_loss/len(loader):.4f} ---")
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), SAVE_PATH)

    torch.save(model.state_dict(), SAVE_PATH)
    print("✅ Done.")

if __name__ == "__main__":
    try: multiprocessing.set_start_method('fork', force=True)
    except: pass
    main()