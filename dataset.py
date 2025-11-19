import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from heatmap_generator import HeatmapGenerator

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class GraspNetHeatmapDataset(Dataset):
    """
    PyTorch Dataset for GraspNet that:
      - loads RGB-D + projected 2D grasps from .npz
      - uses HeatmapGenerator to build ground-truth heatmaps on-the-fly
    """
    def __init__(self, graspnet_root, camera='kinect', downsample_factor=8):
        self.graspnet_root = graspnet_root
        self.camera = camera
        self.downsample_factor = downsample_factor

        self.image_dir_base = os.path.join(graspnet_root, "scenes")
        self.label_dir_base = os.path.join(graspnet_root, f"dataset_{camera}")

        # Target image size (H, W)
        self.image_hw = (360, 640)
        self.grid_size = downsample_factor

        # Heatmap generator (implements Gaussian + grid strategy)
        self.generator = HeatmapGenerator(
            full_hw=self.image_hw,
            grid_size=self.grid_size
        )

        # Build list of all (rgb, depth, label) triplets
        self.scenes = sorted(
            d for d in os.listdir(self.label_dir_base)
            if os.path.isdir(os.path.join(self.label_dir_base, d))
        )

        self.file_list = []
        for label_scene_id in self.scenes:
            try:
                scene_num = int(label_scene_id.split('_')[-1])
                img_scene_id = f"scene_{scene_num:04d}"
            except ValueError:
                continue

            img_rgb_dir = os.path.join(
                self.image_dir_base, img_scene_id, self.camera, "rgb"
            )
            img_depth_dir = os.path.join(
                self.image_dir_base, img_scene_id, self.camera, "depth"
            )
            label_scene_folder = os.path.join(
                self.label_dir_base, label_scene_id, "grasp_labels"
            )

            if (not os.path.isdir(img_rgb_dir)
                    or not os.path.isdir(img_depth_dir)
                    or not os.path.isdir(label_scene_folder)):
                continue

            for img_name in os.listdir(img_rgb_dir):
                if not img_name.endswith(".png"):
                    continue

                img_idx_str = img_name.split('.')[0]
                rgb_path = os.path.join(img_rgb_dir, img_name)
                depth_path = os.path.join(img_depth_dir, img_name)
                grasp_file_name = f"{int(img_idx_str)}_view.npz"
                label_path = os.path.join(label_scene_folder, grasp_file_name)

                if (os.path.exists(rgb_path)
                        and os.path.exists(depth_path)
                        and os.path.exists(label_path)):
                    self.file_list.append((rgb_path, depth_path, label_path))

        print(f"Found {len(self.file_list)} valid image-label pairs for camera '{camera}'.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rgb_path, depth_path, label_path = self.file_list[idx]

        # Load RGB-D image (original resolution)
        rgb = np.array(Image.open(rgb_path)) / 255.0
        depth = np.array(Image.open(depth_path)) / 1000.0
        depth = np.expand_dims(depth, axis=-1)

        orig_h, orig_w = rgb.shape[:2]

        # Scale factors original -> target (360x640)
        scale_y = self.image_hw[0] / orig_h
        scale_x = self.image_hw[1] / orig_w

        # Resize to target size if needed
        if orig_h != self.image_hw[0] or orig_w != self.image_hw[1]:
            rgb = np.array(
                Image.fromarray((rgb * 255).astype(np.uint8))
                .resize((self.image_hw[1], self.image_hw[0]))
            ) / 255.0
            depth = np.array(
                Image.fromarray(depth.squeeze())
                .resize((self.image_hw[1], self.image_hw[0]))
            ) / 1000.0
            depth = np.expand_dims(depth, axis=-1)

        rgbd = np.concatenate([rgb, depth], axis=-1)
        x_image = torch.from_numpy(rgbd).permute(2, 0, 1).float()  # [4,H,W]

        # Load projected grasps from .npz
        try:
            npz_data = np.load(label_path)

            centers = npz_data['centers_2d'].astype(np.float32)   # (N,2) [u,v]
            thetas  = npz_data['thetas_rad'].astype(np.float32)   # (N,)
            widths  = npz_data['widths_2d'].astype(np.float32)    # (N,)
            depths_z  = npz_data['center_z_depths'].astype(np.float32)  # (N,)

            # Rescale centers and widths to 360x640 if needed
            if orig_h != self.image_hw[0] or orig_w != self.image_hw[1]:
                centers[:, 0] *= scale_x
                centers[:, 1] *= scale_y
                widths *= scale_x

            # Convert depths to meters
            depths = depths_z / 1000.0

            projected_grasps = np.hstack([
                centers,
                thetas[:, np.newaxis],
                widths[:, np.newaxis],
                depths[:, np.newaxis]
            ])

        except Exception:
            projected_grasps = np.zeros((0, 5), dtype=np.float32)

        # Generate ground-truth heatmaps
        gt_conf, gt_theta, gt_reg = self.generator.generate_ground_truth(
            grasps_raw=projected_grasps,
            camera_intrinsics=None
        )

        # Downsample confidence to grid resolution
        gt_conf_downsampled = F.avg_pool2d(
            gt_conf,
            kernel_size=self.grid_size,
            stride=self.grid_size
        )

        y_conf_low_res = gt_conf_downsampled  # [1, H/r, W/r]
        y_theta_low_res = gt_theta            # [6, H/r, W/r]
        y_reg_low_res = gt_reg               # [2, H/r, W/r]

        return x_image, (y_conf_low_res, y_theta_low_res, y_reg_low_res)


def draw_grasps_on_image(ax, projected_grasps, h, w):
    """Draw projected grasps [u, v, theta, width, depth] on top of an image."""
    for u, v, theta, width, depth_offset in projected_grasps:
        ax.plot(u, v, 'g.')
        half_w = width / 2.0
        dx = half_w * np.cos(theta)
        dy = half_w * np.sin(theta)
        p1_u, p1_v = u - dx, v - dy
        p2_u, p2_v = u + dx, v + dy
        ax.plot([p1_u, p2_u], [p1_v, p2_v], 'r-', linewidth=2)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')


def visualize_one_sample(dataset, idx=0, save_path="graspnet_sample_debug.png"):
    """Load one sample from the dataset and save a 2x2 debug figure."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; cannot visualize.")
        return

    rgb_path, depth_path, label_path = dataset.file_list[idx]

    # Load RGB for display (resized same as in __getitem__)
    rgb = np.array(Image.open(rgb_path)) / 255.0
    H, W = dataset.image_hw
    if rgb.shape[0] != H or rgb.shape[1] != W:
        rgb = np.array(
            Image.fromarray((rgb * 255).astype(np.uint8))
            .resize((W, H))
        ) / 255.0

    # Load grasps and rescale same way
    npz_data = np.load(label_path)
    centers = npz_data['centers_2d'].astype(np.float32)
    thetas  = npz_data['thetas_rad'].astype(np.float32)
    widths  = npz_data['widths_2d'].astype(np.float32)
    depths_z  = npz_data['center_z_depths'].astype(np.float32)

    orig_h, orig_w = np.array(Image.open(rgb_path)).shape[:2]
    scale_y = H / orig_h
    scale_x = W / orig_w
    if orig_h != H or orig_w != W:
        centers[:, 0] *= scale_x
        centers[:, 1] *= scale_y
        widths *= scale_x
    depths = depths_z / 1000.0

    projected_grasps = np.hstack([
        centers,
        thetas[:, np.newaxis],
        widths[:, np.newaxis],
        depths[:, np.newaxis]
    ]).astype(np.float32)

    gt_conf, gt_theta, gt_reg = dataset.generator.generate_ground_truth(
        grasps_raw=projected_grasps,
        camera_intrinsics=None
    )

    conf_map_np = gt_conf.squeeze().numpy()
    theta_map_np = torch.argmax(gt_theta, dim=0).squeeze().numpy()
    width_map_np = gt_reg[0].numpy()

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Random GraspNet Sample - Data Generation Check", fontsize=16)

    ax[0, 0].imshow(rgb)
    draw_grasps_on_image(ax[0, 0], projected_grasps, H, W)
    ax[0, 0].set_title("RGB + Real Projected Grasps")

    im_conf = ax[0, 1].imshow(conf_map_np, cmap='hot', vmin=0, vmax=1)
    ax[0, 1].set_title("Confidence Heatmap Qc (Full Res)")
    ax[0, 1].axis('off')
    fig.colorbar(im_conf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im_theta = ax[1, 0].imshow(
        theta_map_np, cmap='viridis',
        vmin=0, vmax=dataset.generator.num_angles - 1
    )
    ax[1, 0].set_title("Theta Heatmap Qθ (Grid)")
    ax[1, 0].axis('off')
    fig.colorbar(im_theta, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im_width = ax[1, 1].imshow(width_map_np, cmap='viridis')
    ax[1, 1].set_title("Width Heatmap Qw (Grid)")
    ax[1, 1].axis('off')
    fig.colorbar(im_width, ax=ax[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"Saved debug visualization to {save_path}")


if __name__ == "__main__":
    GRASPNET_ROOT = "data/graspnet"

    dataset = GraspNetHeatmapDataset(
        graspnet_root=GRASPNET_ROOT,
        camera="kinect",
        downsample_factor=8
    )

    if len(dataset) == 0:
        print("No samples found; check GRASPNET_ROOT and camera.")
    else:
        test_idx = 0
        _ = dataset[test_idx]
        visualize_one_sample(dataset, idx=test_idx,
                             save_path="graspnet_sample_debug.png")
