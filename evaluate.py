import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from model import FINNCompatibleGHM_MultiOutput # Must be the same model
from dataset import GraspNetHeatmapDataset # We reuse this for its file-finding logic
from heatmap_generator import HeatmapGenerator # We reuse this to get the full-res GT

# --- Config ---
GRASPNET_ROOT = "data/graspnet" # UPDATE THIS IF WRONG
CAMERA = 'kinect'
MODEL_PATH = "trained_model.pth"
TEST_IMAGE_IDX = 900 # Which image from the dataset to test
OUTPUT_IMAGE_NAME = "prediction_dashboard.png"
# ----------------

def evaluate():
    print("Loading model and data for evaluation...")
    device = torch.device("cpu")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please run 'train_cpu.py' first.")
        return

    # 1. Load Model
    model = FINNCompatibleGHM_MultiOutput().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Load Data for a Single Test Image
    try:
        test_dataset = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA)
    except FileNotFoundError:
        print(f"ERROR: Could not find dataset at {GRASPNET_ROOT}")
        return

    # Get all the paths and data for our chosen image
    rgb_path, depth_path, label_path = test_dataset.file_list[TEST_IMAGE_IDX]
    x_image, (y_conf, y_theta, y_reg) = test_dataset[TEST_IMAGE_IDX]

    # Load the original, un-normalized RGB image for plotting
    original_rgb_img = Image.open(rgb_path)

    # Add a batch dimension (B, C, H, W)
    x_image = x_image.unsqueeze(0).to(device)

    # 3. Run Inference
    with torch.no_grad():
        pred_conf, pred_theta, pred_reg = model(x_image)

    # 4. Get Full-Resolution Ground Truth (for a nice visual)
    # --- THIS IS THE FIX ---
    # We must load the raw .npz data using the *correct keys*
    try:
        npz_data = np.load(label_path)
        centers = npz_data['centers_2d']
        thetas = npz_data['thetas_rad']
        widths = npz_data['widths_2d']
        depths = npz_data['center_z_depths']

        projected_grasps = np.hstack([
            centers,
            thetas[:, np.newaxis],
            widths[:, np.newaxis],
            depths[:, np.newaxis]
        ])
    except Exception as e:
        print(f"Error loading {label_path}, possible empty grasps: {e}")
        projected_grasps = np.array([]) # Create an empty array
    # --- END OF FIX ---

    gt_conf_full, _, _ = test_dataset.generator.generate_ground_truth(projected_grasps, None)
    gt_conf_full_np = gt_conf_full.squeeze(0).cpu().numpy()

    # 5. Post-process all other maps for plotting
    pred_conf_map = torch.sigmoid(pred_conf).squeeze().cpu().numpy()
    pred_width_map = pred_reg.squeeze(0)[0].cpu().numpy() # 0 is width, 1 is depth

    gt_conf_low_res_map = y_conf.squeeze().cpu().numpy()
    gt_width_map = y_reg.squeeze(0)[0].cpu().numpy()

    print(f"Prediction complete. Max confidence: {pred_conf_map.max():.4f}")

    # 6. Create and Save the 2x3 Dashboard Plot
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Evaluation Dashboard (Image {TEST_IMAGE_IDX})", fontsize=16)

    # --- Row 1: Primary Comparison ---
    ax[0, 0].imshow(original_rgb_img)
    ax[0, 0].set_title("Original RGB")
    ax[0, 0].axis('off')

    ax[0, 1].imshow(gt_conf_full_np, cmap='hot', vmin=0, vmax=1)
    ax[0, 1].set_title("Ground Truth (Full Res)")
    ax[0, 1].axis('off')

    im = ax[0, 2].imshow(pred_conf_map, cmap='hot', vmin=0, vmax=1)
    ax[0, 2].set_title("Predicted Confidence (Low Res)")
    ax[0, 2].axis('off')

    # --- Row 2: Attribute Comparison (Width) ---
    ax[1, 0].imshow(gt_conf_low_res_map, cmap='hot', vmin=0, vmax=1)
    ax[1, 0].set_title("Ground Truth (Low Res Target)")
    ax[1, 0].axis('off')

    ax[1, 1].imshow(gt_width_map, cmap='viridis')
    ax[1, 1].set_title("Ground Truth Width")
    ax[1, 1].axis('off')

    im_w = ax[1, 2].imshow(pred_width_map, cmap='viridis')
    ax[1, 2].set_title("Predicted Width")
    ax[1, 2].axis('off')

    # Add colorbars
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046, pad=0.04)
    fig.colorbar(im_w, ax=ax[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE_NAME)

    print(f"Success! Dashboard saved to {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    evaluate()