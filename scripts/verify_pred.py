import os
import torch
import matplotlib.pyplot as plt

from core.models import load_nas_model
from core.dataset import GraspNetHeatmapDataset

# --- config paths
# genome is saved alongside weights during training to prevent arch/weight mismatch
GENOME_FILE = "outputs/gen33_genome.txt"
WEIGHTS_FILE = "outputs/gen33_full_model.pth"
GRASPNET_ROOT = "../data/graspnet"

# --- settings
IMG_INDEX = 2000
CAMERA = 'kinect'
OUTPUT_FILENAME = f"outputs/prediction_{IMG_INDEX}.png"

def main():
    print(f"starting prediction verification for image {IMG_INDEX}...")

    # 1. build model and load weights
    try:
        model = load_nas_model(GENOME_FILE, weights_path=WEIGHTS_FILE, device='cpu')
        print("model built and weights loaded successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    # 2. load data
    if not os.path.exists(GRASPNET_ROOT):
        print(f"data not found at {GRASPNET_ROOT}")
        return
    try:
        ds = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=8)
    except Exception as e:
        print(f"error loading dataset: {e}")
        return

    if IMG_INDEX >= len(ds):
        print(f"index {IMG_INDEX} out of range (dataset size: {len(ds)})")
        return

    # 3. inference
    print(f"running inference...")
    x_tensor, targets = ds[IMG_INDEX]
    gt_loc = targets[0].squeeze().numpy()

    with torch.no_grad():
        # add batch dimension [1, 4, H, W]
        inputs = x_tensor.unsqueeze(0)
        p_loc, p_cls, p_theta, p_width, p_depth = model(inputs)

    # 4. process and plot
    rgb = x_tensor[:3].permute(1, 2, 0).numpy()

    # confidence (sigmoid)
    pred_conf = torch.sigmoid(p_loc).squeeze().numpy()

    # anchor class (sigmoid -> max across 6 anchors)
    pred_anchor = torch.sigmoid(p_cls).squeeze().max(dim=0)[0].numpy()

    # regression heads (take 1st anchor for visualization)
    pred_theta = p_theta.squeeze().numpy()[0]
    pred_width = p_width.squeeze().numpy()[0]
    pred_depth = p_depth.squeeze().numpy()[0]

    # plot setup
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), facecolor='white')
    fig.suptitle(f"NAS Model Prediction: Image {IMG_INDEX}", fontsize=16)

    # inputs
    axs[0,0].imshow(rgb)
    axs[0,0].set_title("Input RGB")

    # gt heatmap
    im0 = axs[0,1].imshow(gt_loc, cmap='jet', vmin=0, vmax=1)
    axs[0,1].set_title("GT Heatmap")
    fig.colorbar(im0, ax=axs[0,1])

    # heatmaps
    im1 = axs[0,2].imshow(pred_conf, cmap='jet', vmin=0, vmax=1)
    axs[0,2].set_title("Pred Confidence")
    fig.colorbar(im1, ax=axs[0,2])

    im2 = axs[0,3].imshow(pred_anchor, cmap='viridis', vmin=0, vmax=1)
    axs[0,3].set_title("Pred Angle Class")
    fig.colorbar(im2, ax=axs[0,3])

    # regression
    im3 = axs[1,0].imshow(pred_theta, cmap='twilight')
    axs[1,0].set_title("Pred Theta Offset")
    fig.colorbar(im3, ax=axs[1,0])

    im4 = axs[1,1].imshow(pred_width, cmap='magma')
    axs[1,1].set_title("Pred Width")
    fig.colorbar(im4, ax=axs[1,1])

    im5 = axs[1,2].imshow(pred_depth, cmap='coolwarm')
    axs[1,2].set_title("Pred Depth")
    fig.colorbar(im5, ax=axs[1,2])

    # overlay: pred vs gt side by side
    diff = abs(pred_conf - gt_loc)
    im6 = axs[1,3].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axs[1,3].set_title("Error |Pred - GT|")
    fig.colorbar(im6, ax=axs[1,3])

    for ax in axs.flat: ax.axis('off')
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"prediction saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()