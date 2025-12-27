import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Fix Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import FINNCompatibleGHM_MultiOutput
from dataset import GraspNetHeatmapDataset

# --- CONFIGURATION ---
IMG_INDEX = 3000           # <--- Change this to select image
CAMERA = 'kinect'
GRASPNET_ROOT = "../data/graspnet"
MODEL_PATH = "../trained_model_hggd.pth"
OUTPUT_FILENAME = f"prediction_raw_{IMG_INDEX}.png" # Name of saved file

def verify_pred():
    device = torch.device("cpu")

    # 1. Load Model & Data
    model = FINNCompatibleGHM_MultiOutput(in_channels=4, num_angles=6).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"❌ Model not found at {os.path.abspath(MODEL_PATH)}"); return
    model.eval()

    try:
        ds = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except:
        print("❌ Error loading dataset."); return

    if IMG_INDEX >= len(ds):
        print(f"❌ Index {IMG_INDEX} out of range."); return

    print(f"🤖 Predicting Image {IMG_INDEX}...")

    # 2. Inference
    x_tensor, _ = ds[IMG_INDEX]
    with torch.no_grad():
        p_loc, p_cls, p_theta, p_width, p_depth = model(x_tensor.unsqueeze(0).to(device))

    # 3. Process Maps (RAW - No Masking)
    rgb = x_tensor[:3].permute(1, 2, 0).numpy()

    # Confidence (Sigmoid)
    pred_conf = torch.sigmoid(p_loc).squeeze().numpy()

    # Anchor Class (Sigmoid -> Max across 6 anchors)
    pred_anchor = torch.sigmoid(p_cls).squeeze().max(dim=0)[0].numpy()

    # Regression Heads (Raw Output)
    # We take channel 0 (1st anchor) to visualize the raw dense map
    pred_theta = p_theta.squeeze().numpy()[0]
    pred_width = p_width.squeeze().numpy()[0]
    pred_depth = p_depth.squeeze().numpy()[0]

    # 4. Plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    fig.suptitle(f"Model Prediction (RAW): Image {IMG_INDEX}", fontsize=16)

    # Inputs
    axs[0,0].imshow(rgb)
    axs[0,0].set_title("Input RGB")

    # Heatmaps
    im1 = axs[0,1].imshow(pred_conf, cmap='jet', vmin=0, vmax=1)
    axs[0,1].set_title("Pred Confidence (c)")
    fig.colorbar(im1, ax=axs[0,1])

    im2 = axs[0,2].imshow(pred_anchor, cmap='viridis', vmin=0, vmax=1)
    axs[0,2].set_title("Pred Anchor Class (θ bin)")
    fig.colorbar(im2, ax=axs[0,2])

    # Regression (Unmasked - will show background noise)
    im3 = axs[1,0].imshow(pred_theta, cmap='twilight')
    axs[1,0].set_title("Pred Theta Offset (θ delta)")
    fig.colorbar(im3, ax=axs[1,0])

    im4 = axs[1,1].imshow(pred_width, cmap='magma')
    axs[1,1].set_title("Pred Width Offset (w)")
    fig.colorbar(im4, ax=axs[1,1])

    im5 = axs[1,2].imshow(pred_depth, cmap='coolwarm')
    axs[1,2].set_title("Pred Depth Offset (d)")
    fig.colorbar(im5, ax=axs[1,2])

    for ax in axs.flat: ax.axis('off')
    plt.tight_layout()

    # --- SAVE TO PNG ---
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"✅ Saved prediction to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    verify_pred()