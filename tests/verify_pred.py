import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import ast

# --- 1. Fix Imports ---
# Go up to Project Root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models import NAS_GHM_Model  # <--- Use the New Model
from dataset import GraspNetHeatmapDataset

# --- CONFIGURATION ---
IMG_INDEX = 3000
CAMERA = 'kinect'
# Paths relative to 'tests/' folder
GRASPNET_ROOT = "../data/graspnet"
MODEL_PATH = "trained_model_hggd.pth"
GENOME_PATH = "best_genome.txt"      # <--- We need this now!
OUTPUT_FILENAME = f"prediction_nas_{IMG_INDEX}.png"

def verify_pred():
    device = torch.device("cpu") # CPU is fine for single image inference

    print(f"📂 Loading Genome from {GENOME_PATH}...")
    if not os.path.exists(GENOME_PATH):
        print(f"❌ Error: {GENOME_PATH} not found."); return

    with open(GENOME_PATH, "r") as f:
        genome = ast.literal_eval(f.read().strip())

    print(f"🏗️ Building NAS Model...")
    # 1. Initialize the correct architecture
    model = NAS_GHM_Model(genome).to(device)

    # 2. Load Weights
    print(f"⚖️ Loading Weights from {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"❌ Weight Mismatch: {e}")
            print("   (Did you train this model using this exact genome file?)")
            return
    else:
        print(f"❌ Model file not found."); return

    model.eval()

    # 3. Load Data
    try:
        ds = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}"); return

    if IMG_INDEX >= len(ds):
        print(f"❌ Index {IMG_INDEX} out of range."); return

    print(f"🤖 Predicting Image {IMG_INDEX}...")

    # 4. Inference
    x_tensor, _ = ds[IMG_INDEX]
    with torch.no_grad():
        # Add batch dimension [1, 4, H, W]
        p_loc, p_cls, p_theta, p_width, p_depth = model(x_tensor.unsqueeze(0).to(device))

    # 5. Process Maps (RAW)
    rgb = x_tensor[:3].permute(1, 2, 0).numpy()

    # Confidence (Sigmoid)
    pred_conf = torch.sigmoid(p_loc).squeeze().numpy()

    # Anchor Class (Sigmoid -> Max across 6 anchors)
    pred_anchor = torch.sigmoid(p_cls).squeeze().max(dim=0)[0].numpy()

    # Regression Heads (Take 1st anchor for visualization)
    pred_theta = p_theta.squeeze().numpy()[0]
    pred_width = p_width.squeeze().numpy()[0]
    pred_depth = p_depth.squeeze().numpy()[0]

    # 6. Plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    fig.suptitle(f"NAS Model Prediction: Image {IMG_INDEX}", fontsize=16)

    # Inputs
    axs[0,0].imshow(rgb)
    axs[0,0].set_title("Input RGB")

    # Heatmaps
    im1 = axs[0,1].imshow(pred_conf, cmap='jet', vmin=0, vmax=1)
    axs[0,1].set_title("Pred Confidence")
    fig.colorbar(im1, ax=axs[0,1])

    im2 = axs[0,2].imshow(pred_anchor, cmap='viridis', vmin=0, vmax=1)
    axs[0,2].set_title("Pred Angle Class")
    fig.colorbar(im2, ax=axs[0,2])

    # Regression
    im3 = axs[1,0].imshow(pred_theta, cmap='twilight')
    axs[1,0].set_title("Pred Theta Offset")
    fig.colorbar(im3, ax=axs[1,0])

    im4 = axs[1,1].imshow(pred_width, cmap='magma')
    axs[1,1].set_title("Pred Width")
    fig.colorbar(im4, ax=axs[1,1])

    im5 = axs[1,2].imshow(pred_depth, cmap='coolwarm')
    axs[1,2].set_title("Pred Depth")
    fig.colorbar(im5, ax=axs[1,2])

    for ax in axs.flat: ax.axis('off')
    plt.tight_layout()

    # --- SAVE ---
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"✅ Saved prediction to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    verify_pred()