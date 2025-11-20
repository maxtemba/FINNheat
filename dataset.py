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
      - Loads RGB-D images and projected 2D grasps from .npz files.
      - Uses HeatmapGenerator (HGGD logic) to build 5 ground-truth tensors on-the-fly.
    """
    def __init__(self, graspnet_root, camera='kinect', downsample_factor=8):
        self.graspnet_root = graspnet_root
        self.camera = camera
        self.downsample_factor = downsample_factor

        self.image_dir_base = os.path.join(graspnet_root, "scenes")
        self.label_dir_base = os.path.join(graspnet_root, f"dataset_{camera}")

        # Target image size (H, W) - Must match model input
        self.image_hw = (360, 640)
        self.grid_size = downsample_factor

        # Initialize the HGGD-compliant Heatmap Generator
        self.generator = HeatmapGenerator(
            full_hw=self.image_hw,
            grid_size=self.grid_size,
            num_angles=6,   # Default HGGD anchor count
            anchor_w=50.0,  # Default HGGD reference width
            anchor_z=20.0,  # Default HGGD reference depth
            sigma=10        # Fixed sigma for confidence map
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

        # --- 1. Load Images ---
        rgb = np.array(Image.open(rgb_path)) / 255.0
        depth = np.array(Image.open(depth_path)) / 1000.0 # Convert mm to meters
        depth = np.expand_dims(depth, axis=-1)

        orig_h, orig_w = rgb.shape[:2]

        # --- 2. Resize Images (if needed) ---
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

        # Combine RGB and Depth -> [4, H, W]
        rgbd = np.concatenate([rgb, depth], axis=-1)
        x_image = torch.from_numpy(rgbd).permute(2, 0, 1).float()

        # --- 3. Load and Scale Labels ---
        try:
            npz_data = np.load(label_path)

            centers = npz_data['centers_2d'].astype(np.float32)   # (N,2) [u,v]
            thetas  = npz_data['thetas_rad'].astype(np.float32)   # (N,)
            widths  = npz_data['widths_2d'].astype(np.float32)    # (N,)
            depths_z  = npz_data['center_z_depths'].astype(np.float32)  # (N,)

            # Scale factors original -> target (e.g. 720 -> 360)
            scale_y = self.image_hw[0] / orig_h
            scale_x = self.image_hw[1] / orig_w

            # Resize label coordinates if image was resized
            if orig_h != self.image_hw[0] or orig_w != self.image_hw[1]:
                centers[:, 0] *= scale_x
                centers[:, 1] *= scale_y
                widths *= scale_x

            # Depth Target: delta = grasp_z - scene_depth
            # We need the scene depth at the grasp center to compute the delta.
            # Because doing this lookup for thousands of grasps is slow in Python,
            # HGGD often pre-calculates it or approximates.
            # Ideally: delta = depths_z/1000.0 - scene_depth_at_center
            # Here we pass the raw depth (converted to meters) and let the generator handle normalization.
            grasp_depths_m = depths_z / 1000.0

            # Calculate Depth Delta (Grasp Depth - Scene Depth at center)
            # Fast approximation: sample the resized depth map we just loaded
            center_uv_int = centers.astype(int)
            # Clip to bounds
            np.clip(center_uv_int[:, 0], 0, self.image_hw[1]-1, out=center_uv_int[:, 0])
            np.clip(center_uv_int[:, 1], 0, self.image_hw[0]-1, out=center_uv_int[:, 1])

            scene_depths_at_center = depth[center_uv_int[:, 1], center_uv_int[:, 0], 0]
            depth_deltas = grasp_depths_m - scene_depths_at_center

            # Pack into (N, 5) array: [u, v, theta, width, depth_delta]
            projected_grasps = np.hstack([
                centers,
                thetas[:, np.newaxis],
                widths[:, np.newaxis],
                depth_deltas[:, np.newaxis]
            ])

        except Exception:
            projected_grasps = np.zeros((0, 5), dtype=np.float32)

        # --- 4. Generate Ground Truth (5 Tensors) ---
        # gt_loc: [1, H, W] (Full Res)
        # gt_cls, gt_theta, gt_width, gt_depth: [K, H/r, W/r] (Grid Res)
        gt_loc, gt_cls, gt_theta, gt_width, gt_depth = \
            self.generator.generate_ground_truth(projected_grasps)

        # --- 5. Downsample Location Map ---
        # The network outputs loc_map at grid resolution (H/8, W/8), so we must downsample the GT.
        # Use AvgPool to preserve the gaussian peaks (Max pool might be too aggressive for soft targets)
        # or MaxPool if strictly following CornerNet style. HGGD uses draw_gaussian on a downsampled grid in some versions,
        # but since we drew it at full res, AvgPool is a safe way to alias it down.
        gt_loc_downsampled = F.avg_pool2d(
            gt_loc,
            kernel_size=self.grid_size,
            stride=self.grid_size
        )

        return x_image, (gt_loc_downsampled, gt_cls, gt_theta, gt_width, gt_depth)


# ==============================================================================
# HELPER FUNCTIONS (For debugging / Visualization)
# ==============================================================================

def draw_grasps_on_image(ax, projected_grasps, h, w):
    """Draw projected grasps on top of an image."""
    for u, v, theta, width, depth_delta in projected_grasps:
        ax.plot(u, v, 'g.', markersize=3)
        half_w = width / 2.0
        dx = half_w * np.cos(theta)
        dy = half_w * np.sin(theta)
        p1_u, p1_v = u - dx, v - dy
        p2_u, p2_v = u + dx, v + dy
        ax.plot([p1_u, p2_u], [p1_v, p2_v], 'r-', linewidth=1)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')


if __name__ == "__main__":
    # Quick test block
    GRASPNET_ROOT = "data/graspnet"
    dataset = GraspNetHeatmapDataset(
        graspnet_root=GRASPNET_ROOT,
        camera="kinect",
        downsample_factor=8
    )

    if len(dataset) > 0:
        img, targets = dataset[0]
        gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets

        print("Dataset Test Successful:")
        print(f"  Image Shape: {img.shape}")
        print(f"  Loc Map:   {gt_loc.shape}")
        print(f"  Cls Map:   {gt_cls.shape}")
        print(f"  Theta Map: {gt_theta.shape}")
        print(f"  Width Map: {gt_width.shape}")
        print(f"  Depth Map: {gt_depth.shape}")
    else:
        print("No data found. Check paths.")