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
EPOCHS = 10
NUM_TRAIN_IMAGES = 500
SAVE_PATH = "trained_model_1k.pth"
OUTPUT_IMAGE = "train_result_dashboard.png"

# ==============================================================================
# 1. HGGD Loss Function
# ==============================================================================
def hggd_loss(preds, targets, loc_a=1, reg_b=5, cls_c=1):
    pred_loc, pred_cls, pred_theta, pred_width, pred_depth = preds
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets

    device = pred_loc.device
    eps = 1e-6

    # --- A. Location Map Loss (Focal Loss) ---
    pos_inds = gt_loc.ge(0.99).float()
    neg_inds = gt_loc.lt(0.99).float()

    pred_loc = torch.sigmoid(pred_loc)
    pred_loc = torch.clamp(pred_loc, eps, 1 - eps)

    alpha, gamma = 0.5, 2
    pos_loss = torch.log(pred_loc) * torch.pow(1 - pred_loc, gamma) * pos_inds
    neg_loss = torch.log(1 - pred_loc) * torch.pow(pred_loc, gamma) * neg_inds * torch.pow(1 - gt_loc, 4)

    num_pos = pos_inds.sum()
    loc_loss = -(alpha * pos_loss.sum() + (1 - alpha) * neg_loss.sum())
    if num_pos > 0:
        loc_loss /= num_pos

    # --- B. Classification Loss (Focal Loss on Anchors) ---
    pred_cls = torch.sigmoid(pred_cls)
    pred_cls = torch.clamp(pred_cls, eps, 1 - eps)

    cls_pos_inds = gt_cls.ge(0.5).float()
    cls_neg_inds = gt_cls.lt(0.5).float()

    cls_pos_loss = 0.25 * torch.log(pred_cls) * torch.pow(1 - pred_cls, gamma) * cls_pos_inds
    cls_neg_loss = 0.75 * torch.log(1 - pred_cls) * torch.pow(pred_cls, gamma) * cls_neg_inds

    num_cls_pos = cls_pos_inds.sum()
    cls_loss = -(cls_pos_loss.sum() + cls_neg_loss.sum())
    if num_cls_pos > 0:
        cls_loss /= num_cls_pos

    # --- C. Regression Loss (Smooth L1) ---
    reg_mask = (gt_cls > 0).float()
    num_reg = reg_mask.sum() + eps

    l1 = F.smooth_l1_loss(pred_theta * reg_mask, gt_theta * reg_mask, reduction='sum')
    l2 = F.smooth_l1_loss(pred_width * reg_mask, gt_width * reg_mask, reduction='sum')
    l3 = F.smooth_l1_loss(pred_depth * reg_mask, gt_depth * reg_mask, reduction='sum')

    reg_loss = (l1 + l2 + l3) / num_reg

    total_loss = (loc_a * loc_loss) + (cls_c * cls_loss) + (reg_b * reg_loss)
    return total_loss, loc_loss, cls_loss, reg_loss

# ==============================================================================
# 2. Visualization Helper
# ==============================================================================
def save_dashboard(model, dataset, idx, device):
    print(f"\nGenerating test output for image index {idx}...")
    model.eval()

    # Load data
    rgb_path, _, _ = dataset.dataset.file_list[idx]
    x_image, targets = dataset.dataset[idx]
    gt_loc, gt_cls, _, _, _ = targets

    # Run Inference
    with torch.no_grad():
        inp = x_image.unsqueeze(0).to(device)
        p_loc, p_cls, _, _, _ = model(inp)

    # Prepare for plotting
    rgb = np.array(Image.open(rgb_path).resize((640, 360)))

    # Post-process predictions
    pred_conf = torch.sigmoid(p_loc).squeeze().cpu().numpy()
    pred_cls_map = torch.sigmoid(p_cls).max(dim=1)[0].squeeze().cpu().numpy()

    # Ground Truths
    gt_conf = gt_loc.squeeze().cpu().numpy()
    gt_cls_map = gt_cls.sum(dim=0).cpu().numpy()

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("HGGD Training Check (Parallelized)", fontsize=16)

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Input RGB")
    ax[0, 0].axis('off')

    ax[0, 1].imshow(pred_conf, cmap='jet', vmin=0, vmax=1)
    ax[0, 1].set_title(f"Pred Loc Map (Max: {pred_conf.max():.3f})")
    ax[0, 1].axis('off')

    ax[1, 0].imshow(gt_conf, cmap='jet', vmin=0, vmax=1)
    ax[1, 0].set_title("GT Loc Map")
    ax[1, 0].axis('off')

    ax[1, 1].imshow(np.zeros_like(pred_cls_map), cmap='gray')
    ax[1, 1].imshow(gt_cls_map, cmap='Reds', alpha=0.5)
    ax[1, 1].imshow(pred_cls_map > 0.1, cmap='Cyan', alpha=0.3)
    ax[1, 1].set_title("Anchors: GT (Red) vs Pred (Cyan)")
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"✅ Dashboard saved to {OUTPUT_IMAGE}")
    model.train()

# ==============================================================================
# 3. Main Training Loop
# ==============================================================================
def main():
    # --- PARALLEL CONFIGURATION ---
    # Use all CPU cores for data loading
    num_cores = os.cpu_count()
    # Determine safe number of workers (leave 1-2 cores for system/training)
    num_workers = max(1, num_cores - 1)

    # Force PyTorch to use all cores for Matrix Multiplications
    torch.set_num_threads(num_cores)

    device = torch.device("cpu")
    print(f"🚀 Training on CPU with {num_workers} data loader workers.")
    print(f"   (Using {num_cores} threads for model computation)")

    # 1. Load Data
    print("Initializing Dataset...")
    try:
        full_dataset = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    if len(full_dataset) == 0:
        print("❌ Dataset is empty. Check path.")
        return

    # 2. Random Subset
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:NUM_TRAIN_IMAGES]
    train_dataset = Subset(full_dataset, subset_indices)
    test_idx = subset_indices[0]

    # --- PARALLEL DATA LOADER ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,   # Parallelize data generation
        persistent_workers=True,   # Keep workers alive between epochs
        prefetch_factor=2,         # Load 2 batches ahead per worker
        pin_memory=False           # False because we are on CPU anyway
    )
    print(f"Selected {len(train_dataset)} random images for training.")

    # 3. Initialize Model
    model = FINNCompatibleGHM_MultiOutput(in_channels=4, num_angles=6).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train Loop
    print("\n--- Starting Training ---")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for i, (x, targets) in enumerate(train_loader):
            x = x.to(device)
            targets = [t.to(device) for t in targets]

            preds = model(x)
            loss, l_loc, l_cls, l_reg = hggd_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} "
                      f"(Loc:{l_loc.item():.3f} Cls:{l_cls.item():.3f} Reg:{l_reg.item():.3f})")

        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} ---")

    # 5. Save Model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")

    # 6. Visualize Output
    save_dashboard(model, train_dataset, test_idx, device)

if __name__ == "__main__":
    # Fix for multiprocessing on MacOS/Linux
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    main()