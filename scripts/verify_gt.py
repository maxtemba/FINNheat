import os
import matplotlib.pyplot as plt

from core.dataset import GraspNetHeatmapDataset

# --- settings
# paths
GRASPNET_ROOT = "../data/graspnet"

# parameters
IMG_INDEX = 3000
CAMERA = 'kinect'
DOWNSAMPLE_FACTOR = 8
OUTPUT_FILENAME = f"outputs/gt_{IMG_INDEX}.png"

def main():
    print(f"starting ground truth verification for image {IMG_INDEX}...")

    # 1. load data
    if not os.path.exists(GRASPNET_ROOT):
        print(f"data not found at {GRASPNET_ROOT}")
        return

    try:
        ds = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=DOWNSAMPLE_FACTOR)
    except Exception as e:
        print(f"error loading dataset: {e}")
        return

    if IMG_INDEX >= len(ds):
        print(f"index {IMG_INDEX} out of range (dataset size: {len(ds)})")
        return


    # 2. get gt data
    x_tensor, targets = ds[IMG_INDEX]
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets

    # 3. process for visualization
    # permute rgb to HWC for plotting
    rgb = x_tensor[:3].permute(1, 2, 0).numpy()

    # confidence map
    loc = gt_loc.squeeze().numpy()

    # anchor class (max across channels)
    cls = gt_cls.max(dim=0)[0].numpy()

    # regression maps (raw - no masking)
    # visualizing channel 0 (corresponding to anchor 0)
    theta = gt_theta.numpy()[0]
    width = gt_width.numpy()[0]
    depth = gt_depth.numpy()[0]

    # 4. plot (paper style)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    fig.suptitle(f"Ground Truth (RAW): Image {IMG_INDEX}", fontsize=16)

    # inputs
    axs[0,0].imshow(rgb)
    axs[0,0].set_title("Input RGB")

    # heatmaps
    im1 = axs[0,1].imshow(loc, cmap='jet', vmin=0, vmax=1)
    axs[0,1].set_title("GT Confidence (c)")
    fig.colorbar(im1, ax=axs[0,1])

    im2 = axs[0,2].imshow(cls, cmap='viridis', vmin=0, vmax=1)
    axs[0,2].set_title("GT Anchor Class (θ bin)")
    fig.colorbar(im2, ax=axs[0,2])

    # regression (unmasked)
    im3 = axs[1,0].imshow(theta, cmap='twilight')
    axs[1,0].set_title("GT Theta Offset (Channel 0)")
    fig.colorbar(im3, ax=axs[1,0])

    im4 = axs[1,1].imshow(width, cmap='magma')
    axs[1,1].set_title("GT Width Offset (Channel 0)")
    fig.colorbar(im4, ax=axs[1,1])

    im5 = axs[1,2].imshow(depth, cmap='coolwarm')
    axs[1,2].set_title("GT Depth Offset (Channel 0)")
    fig.colorbar(im5, ax=axs[1,2])

    # cleanup layout
    for ax in axs.flat: ax.axis('off')
    plt.tight_layout()

    # 5. save output
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"visualization saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()