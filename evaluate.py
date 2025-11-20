import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- Import your specific modules ---
from model import FINNCompatibleGHM_MultiOutput
from dataset import GraspNetHeatmapDataset

# --- Config ---
GRASPNET_ROOT = "data/graspnet"
CAMERA = 'kinect'
MODEL_PATH = "trained_model_1k.pth" # Uses the model you just trained
TEST_IMAGE_IDX = 0 # Change this to see different images
OUTPUT_IMAGE_NAME = "evaluation_dashboard.png"
# ----------------

def evaluate():
    print("Loading model and data for evaluation...")
    device = torch.device("cpu") # Evaluate on CPU for simplicity

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return

    # 1. Load Model (Updated architecture with 5 heads)
    model = FINNCompatibleGHM_MultiOutput(in_channels=4, num_angles=6).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Make sure 'model.py' defines the exact architecture used during training.")
        return
    model.eval()

    # 2. Load Data
    try:
        test_dataset = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        return

    if len(test_dataset) == 0:
        print("Dataset empty.")
        return

    print(f"Testing image index: {TEST_IMAGE_IDX}")

    # Unpack the 5 targets from the dataset
    x_image, targets = test_dataset[TEST_IMAGE_IDX]
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets

    # Get file path for visualization
    rgb_path, _, label_path = test_dataset.file_list[TEST_IMAGE_IDX]
    original_rgb_img = Image.open(rgb_path)

    # Add batch dimension
    x_image = x_image.unsqueeze(0).to(device)

    # 3. Run Inference
    with torch.no_grad():
        # Model now returns 5 tensors
        p_loc, p_cls, p_theta, p_width, p_depth = model(x_image)

    # 4. Generate High-Res Ground Truth for Visualization
    # We manually trigger the generator to get the full 640x360 Confidence Map
    # (The dataset returns a downsampled 80x45 map for training)
    try:
        # Re-load raw data logic from dataset (simplified)
        npz_data = np.load(label_path)
        centers = npz_data['centers_2d'].astype(np.float32)
        # We only need centers for the Loc Map
        # Create dummy array for the rest to satisfy generator signature
        dummy_zeros = np.zeros_like(centers[:, 0:1])
        projected_grasps = np.hstack([centers, dummy_zeros, dummy_zeros, dummy_zeros])

        # Scale centers if image was resized (assuming 720p -> 360p standard)
        orig_h = np.array(original_rgb_img).shape[0]
        if orig_h != 360:
            scale = 360 / orig_h
            projected_grasps[:, :2] *= scale

        gt_loc_full, _, _, _, _ = test_dataset.generator.generate_ground_truth(projected_grasps)
        gt_loc_full_np = gt_loc_full.squeeze().numpy()

    except Exception as e:
        print(f"Warning: Could not generate high-res GT: {e}")
        gt_loc_full_np = np.zeros((360, 640))

    # 5. Process Predictions for Display
    # Loc: Sigmoid -> [0,1]
    pred_loc_map = torch.sigmoid(p_loc).squeeze().cpu().numpy()

    # Cls: Sigmoid -> [0,1]. Show Max confidence across all 6 anchors.
    pred_cls_map = torch.sigmoid(p_cls).max(dim=1)[0].squeeze().cpu().numpy()

    # Width: Just show the first anchor channel for visual check
    pred_width_map = p_width.squeeze(0)[0].cpu().numpy()
    gt_width_map = gt_width.squeeze()[0].cpu().numpy()

    print(f"Prediction Stats:")
    print(f"  Max Loc Confidence: {pred_loc_map.max():.4f}")
    print(f"  Max Cls Probability: {pred_cls_map.max():.4f}")

    # 6. Plot Dashboard
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Model Evaluation (Image {TEST_IMAGE_IDX})", fontsize=16)

    # --- Row 1: Location / Confidence ---
    ax[0, 0].imshow(original_rgb_img.resize((640, 360)))
    ax[0, 0].set_title("Input RGB")
    ax[0, 0].axis('off')

    ax[0, 1].imshow(gt_loc_full_np, cmap='jet', vmin=0, vmax=1)
    ax[0, 1].set_title("GT Loc Map (Full Res)")
    ax[0, 1].axis('off')

    im1 = ax[0, 2].imshow(pred_loc_map, cmap='jet', vmin=0, vmax=1)
    ax[0, 2].set_title("Pred Loc Map (Low Res)")
    ax[0, 2].axis('off')
    fig.colorbar(im1, ax=ax[0, 2], fraction=0.046, pad=0.04)

    # --- Row 2: Attributes (Anchors & Width) ---
    # GT Cls (Sum over anchors to see all valid spots)
    gt_cls_sum = gt_cls.sum(dim=0).cpu().numpy()
    ax[1, 0].imshow(gt_cls_sum, cmap='viridis')
    ax[1, 0].set_title("GT Anchor Mask (Sum)")
    ax[1, 0].axis('off')

    # Pred Cls (Max over anchors)
    im2 = ax[1, 1].imshow(pred_cls_map, cmap='viridis')
    ax[1, 1].set_title("Pred Anchor Prob (Max)")
    ax[1, 1].axis('off')
    fig.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    # Pred Width (Anchor 0)
    im3 = ax[1, 2].imshow(pred_width_map, cmap='magma')
    ax[1, 2].set_title("Pred Width Offset (Anchor 0)")
    ax[1, 2].axis('off')
    fig.colorbar(im3, ax=ax[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE_NAME)

    print(f"✅ Success! Saved evaluation dashboard to {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    evaluate()