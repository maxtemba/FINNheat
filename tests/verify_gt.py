import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Fix Imports: Add parent directory to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import GraspNetHeatmapDataset

# --- CONFIGURATION ---
IMG_INDEX = 3000
CAMERA = 'kinect'
# Use ".." to go up one level from 'test/' to the project root
GRASPNET_ROOT = "../data/graspnet"
OUTPUT_FILENAME = f"gt_raw_{IMG_INDEX}.png"  # Name of saved file

def verify_gt():
    # 1. Load Data
    try:
        ds = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   (Checked path: {os.path.abspath(GRASPNET_ROOT)})")
        return

    if IMG_INDEX >= len(ds):
        print(f"❌ Index {IMG_INDEX} out of range (Dataset size: {len(ds)})")
        return

    print(f"🔎 Viewing Ground Truth for Image {IMG_INDEX}...")

    # 2. Get GT Data
    x_tensor, targets = ds[IMG_INDEX]
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets

    # 3. Process for Vis
    rgb = x_tensor[:3].permute(1, 2, 0).numpy()

    # Confidence Map
    loc = gt_loc.squeeze().numpy()

    # Anchor Class (Max across channels)
    cls = gt_cls.max(dim=0)[0].numpy()

    # Regression Maps (RAW - No Masking)
    # We visualize Channel 0 (corresponding to Anchor 0)
    # Note: In GT, background pixels are exactly 0.0
    theta = gt_theta.numpy()[0]
    width = gt_width.numpy()[0]
    depth = gt_depth.numpy()[0]

    # 4. Plot (Paper Style)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    fig.suptitle(f"Ground Truth (RAW): Image {IMG_INDEX}", fontsize=16)

    # Inputs
    axs[0,0].imshow(rgb)
    axs[0,0].set_title("Input RGB")

    # Heatmaps
    im1 = axs[0,1].imshow(loc, cmap='jet', vmin=0, vmax=1)
    axs[0,1].set_title("GT Confidence (c)")
    fig.colorbar(im1, ax=axs[0,1])

    im2 = axs[0,2].imshow(cls, cmap='viridis', vmin=0, vmax=1)
    axs[0,2].set_title("GT Anchor Class (θ bin)")
    fig.colorbar(im2, ax=axs[0,2])

    # Regression (Unmasked)
    im3 = axs[1,0].imshow(theta, cmap='twilight')
    axs[1,0].set_title("GT Theta Offset (Channel 0)")
    fig.colorbar(im3, ax=axs[1,0])

    im4 = axs[1,1].imshow(width, cmap='magma')
    axs[1,1].set_title("GT Width Offset (Channel 0)")
    fig.colorbar(im4, ax=axs[1,1])

    im5 = axs[1,2].imshow(depth, cmap='coolwarm')
    axs[1,2].set_title("GT Depth Offset (Channel 0)")
    fig.colorbar(im5, ax=axs[1,2])

    for ax in axs.flat: ax.axis('off')
    plt.tight_layout()

    # --- SAVE TO PNG ---
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"✅ Saved ground truth visualization to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    verify_gt()