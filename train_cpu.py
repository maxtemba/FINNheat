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
# Ensure these files are in your python path
from model import FINNCompatibleGHM_MultiOutput
from dataset import GraspNetHeatmapDataset

# --- Configuration ---
GRASPNET_ROOT = "data/graspnet"
CAMERA = 'kinect'
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_TRAIN_IMAGES = 1000
SAVE_PATH = "trained_model_hggd.pth"
OUTPUT_IMAGE = "train_dashboard.png"

# ==============================================================================
# 1. HGGD Loss Function (Paper Compliant)
# ==============================================================================
def hggd_loss(preds, targets, loc_a=1.0, reg_b=5.0, cls_c=1.0):
    """
    Fixed HGGD Loss Function

    Components:
    - Location heatmap loss (focal loss variant)
    - Anchor classification loss (multi-label focal loss)
    - Regression loss (smooth L1 for offset predictions)
    """
    pred_loc, pred_cls, pred_theta, pred_width, pred_depth = preds
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets

    device = pred_loc.device
    eps = 1e-6

    # ==========================================================================
    # PART A: Location Heatmap Loss (Penalty-Reduced Focal Loss)
    # ==========================================================================

    pred_loc = torch.sigmoid(pred_loc)
    pred_loc = torch.clamp(pred_loc, eps, 1 - eps)

    # ✅ FIX #1: Use threshold instead of exact equality
    # Gaussian peaks typically have values 0.7-1.0 at the center region
    pos_inds = (gt_loc >= 0.7).float()  # Positive = peak region
    neg_inds = (gt_loc < 0.7).float()   # Negative = background

    # Weight negative samples: pixels closer to center get less penalty
    # This prevents punishing predictions near the actual grasp location
    neg_weights = torch.pow(1 - gt_loc, 4)

    # Focal loss components
    # For positive pixels: encourage high confidence
    loss_pos = torch.log(pred_loc) * torch.pow(1 - pred_loc, 2) * pos_inds

    # For negative pixels: penalize false positives, but less near centers
    loss_neg = torch.log(1 - pred_loc) * torch.pow(pred_loc, 2) * neg_weights * neg_inds

    # ✅ FIX #2: Safer normalization
    num_pos = pos_inds.sum().clamp(min=1.0)  # Prevent division by zero

    # Normalize by number of positive pixels
    loss_loc = -(loss_pos.sum() + loss_neg.sum()) / num_pos

    # ==========================================================================
    # PART B: Regression Loss (Only at Object Centers)
    # ==========================================================================

    # Use the same positive mask
    mask = pos_inds
    num_mask = mask.sum()

    if num_mask > 0:
        # ✅ FIX #3: Only compute loss where mask is positive
        # Expand mask to match multi-channel outputs (6 angles)
        mask_expanded = mask.expand_as(pred_theta)  # [B, 6, H, W]

        # Extract only the masked values
        pred_theta_masked = pred_theta[mask_expanded > 0]
        gt_theta_masked = gt_theta[mask_expanded > 0]

        pred_width_masked = pred_width[mask_expanded > 0]
        gt_width_masked = gt_width[mask_expanded > 0]

        pred_depth_masked = pred_depth[mask_expanded > 0]
        gt_depth_masked = gt_depth[mask_expanded > 0]

        # Compute smooth L1 loss only on valid locations
        l_theta = F.smooth_l1_loss(pred_theta_masked, gt_theta_masked, reduction='mean')
        l_width = F.smooth_l1_loss(pred_width_masked, gt_width_masked, reduction='mean')
        l_depth = F.smooth_l1_loss(pred_depth_masked, gt_depth_masked, reduction='mean')

        loss_reg = l_theta + l_width + l_depth
    else:
        loss_reg = torch.tensor(0.0, device=device)

    # ==========================================================================
    # PART C: Anchor Classification Loss (Focal Loss)
    # ==========================================================================

    pred_cls = torch.sigmoid(pred_cls)
    pred_cls = torch.clamp(pred_cls, eps, 1 - eps)

    # Standard focal loss for multi-label classification
    alpha, gamma = 0.25, 2.0

    # Binary cross-entropy for each anchor channel
    bce = -(gt_cls * torch.log(pred_cls) + (1 - gt_cls) * torch.log(1 - pred_cls))

    # Focal term: down-weight easy examples
    pt = torch.where(gt_cls == 1, pred_cls, 1 - pred_cls)
    focal_weight = alpha * torch.pow(1 - pt, gamma)

    loss_cls = (focal_weight * bce).mean()

    # ==========================================================================
    # PART D: Combine All Losses
    # ==========================================================================

    total_loss = (loc_a * loss_loc) + (cls_c * loss_cls) + (reg_b * loss_reg)

    return total_loss, loss_loc, loss_cls, loss_reg


# ==============================================================================
# 2. Initialization Helper (CRITICAL FIX)
# ==============================================================================
def init_heatmap_bias(model):
    """
    Initializes the bias of the final heatmap convolution to -4.59.
    This corresponds to a prior probability of 0.01.
    Prevents the 'loss goes to 0' collapse in the first epoch.
    """
    print("🔧 Initializing Heatmap Bias...")
    initialized = False
    for name, m in model.named_modules():
        # Look for the specific output heads.
        # Adjust 'hm' or 'loc' to match the variable names in your FINNCompatibleGHM model
        if isinstance(m, nn.Conv2d) and ('loc' in name or 'heatmap' in name or 'hm' in name):
            if m.bias is not None:
                nn.init.constant_(m.bias, -4.59) # -log((1-0.01)/0.01)
                print(f"   -> Applied bias init to layer: {name}")
                initialized = True

    if not initialized:
        print("⚠️ WARNING: Could not find a layer named 'loc' or 'heatmap'. "
              "Please check your model definition and rename the final layer or update this function.")

# ==============================================================================
# 3. Visualization Helper
# ==============================================================================
def save_dashboard(model, dataset, idx, device):
    print(f"\nGenerating dashboard for image index {idx}...")
    model.eval()

    # Load data
    rgb_path, _, _ = dataset.dataset.file_list[idx]
    x_image, targets = dataset.dataset[idx]
    gt_loc = targets[0] # targets is a list: [loc, cls, theta, width, depth]

    # Run Inference
    with torch.no_grad():
        inp = x_image.unsqueeze(0).to(device)
        preds = model(inp)
        p_loc = preds[0]

    # Prepare for plotting
    try:
        rgb = np.array(Image.open(rgb_path).resize((640, 360)))
    except:
        rgb = np.zeros((360, 640, 3)) # Fallback if path fails

    # Post-process predictions
    pred_conf = torch.sigmoid(p_loc).squeeze().cpu().numpy()
    gt_conf = gt_loc.squeeze().cpu().numpy()

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("HGGD Training Dashboard", fontsize=16)

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Input RGB")
    ax[0, 0].axis('off')

    # Use fixed scales to see if the model is confident
    ax[0, 1].imshow(pred_conf, cmap='turbo', vmin=0, vmax=1)
    ax[0, 1].set_title(f"Pred Heatmap (Max: {pred_conf.max():.4f})")
    ax[0, 1].axis('off')

    ax[1, 0].imshow(gt_conf, cmap='turbo', vmin=0, vmax=1)
    ax[1, 0].set_title("GT Heatmap (Gaussian Encoded)")
    ax[1, 0].axis('off')

    # Difference map
    diff = np.abs(gt_conf - pred_conf)
    ax[1, 1].imshow(diff, cmap='inferno')
    ax[1, 1].set_title("Error Map (L1 Diff)")
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"✅ Dashboard saved to {OUTPUT_IMAGE}")
    model.train()

# ==============================================================================
# 4. Main Training Loop
# ==============================================================================
def main():
    # --- Parallel Setup ---
    num_cores = os.cpu_count()
    num_workers = max(1, num_cores - 2)
    torch.set_num_threads(num_cores)

    # Use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device} with {num_workers} workers.")

    # 1. Load Data
    print("Initializing Dataset...")
    try:
        # Assuming dataset outputs: image, [loc, cls, theta, width, depth]
        full_dataset = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    if len(full_dataset) == 0:
        print("❌ Dataset is empty.")
        return

    # Random Subset
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:NUM_TRAIN_IMAGES]
    train_dataset = Subset(full_dataset, subset_indices)

    # Keep a fixed test index for visualization
    test_idx = subset_indices[0]

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 2. Initialize Model
    model = FINNCompatibleGHM_MultiOutput(in_channels=4, num_angles=6).to(device)

    # --- APPLY BIAS INIT ---
    init_heatmap_bias(model)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train Loop
    print("\n--- Starting Training ---")
    model.train()

    for epoch in range(EPOCHS):
        total_loss_epoch = 0

        for i, (x, targets) in enumerate(train_loader):
            x = x.to(device)
            # Unpack targets to device
            targets = [t.to(device) for t in targets]

            # Forward
            preds = model(x)

            # Loss Calculation
            loss, l_loc, l_cls, l_reg = hggd_loss(preds, targets)

            # Check for NaN
            if torch.isnan(loss):
                print("❌ Loss is NaN! Stopping.")
                return

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Optional but recommended for Heatmaps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            total_loss_epoch += loss.item()

            if i % 10 == 0:
                print(f"Ep {epoch+1} | B {i} | Loss: {loss.item():.4f} "
                      f"(Loc:{l_loc.item():.3f} Cls:{l_cls.item():.3f} Reg:{l_reg.item():.3f})")

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} ---")

        # Save check point every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_ep{epoch+1}.pth")

    # 4. Final Save & Viz
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")
    save_dashboard(model, full_dataset, test_idx, device)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()