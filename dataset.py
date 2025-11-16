import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
from heatmap_generator import HeatmapGenerator

class GraspNetHeatmapDataset(Dataset):
    """
    Loads raw 6-DoF grasp data and generates heatmaps on-the-fly.
    This version is fixed to use the correct .npz keys.
    """
    def __init__(self, graspnet_root, camera='kinect', downsample_factor=8):
        self.graspnet_root = graspnet_root
        self.camera = camera
        self.downsample_factor = downsample_factor

        self.image_dir_base = os.path.join(graspnet_root, "scenes")
        self.label_dir_base = os.path.join(graspnet_root, f"dataset_{camera}")

        self.image_hw = (360, 640)
        self.grid_size = downsample_factor

        self.generator = HeatmapGenerator(
            full_hw=self.image_hw,
            grid_size=self.grid_size
        )

        self.scenes = sorted([d for d in os.listdir(self.label_dir_base) if os.path.isdir(os.path.join(self.label_dir_base, d))])

        self.file_list = []
        for label_scene_id in self.scenes:
            try:
                scene_num = int(label_scene_id.split('_')[-1])
                img_scene_id = f"scene_{scene_num:04d}"
            except ValueError:
                continue

            img_rgb_dir = os.path.join(self.image_dir_base, img_scene_id, self.camera, "rgb")
            img_depth_dir = os.path.join(self.image_dir_base, img_scene_id, self.camera, "depth")
            label_scene_folder = os.path.join(self.label_dir_base, label_scene_id, "grasp_labels")

            if not os.path.isdir(img_rgb_dir) or not os.path.isdir(img_depth_dir) or not os.path.isdir(label_scene_folder):
                continue

            for img_name in os.listdir(img_rgb_dir):
                if not img_name.endswith(".png"):
                    continue

                img_idx_str = img_name.split('.')[0]
                rgb_path = os.path.join(img_rgb_dir, img_name)
                depth_path = os.path.join(img_depth_dir, img_name)
                grasp_file_name = f"{int(img_idx_str)}_view.npz"
                label_path = os.path.join(label_scene_folder, grasp_file_name)

                if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(label_path):
                    self.file_list.append((rgb_path, depth_path, label_path))

        print(f"Found {len(self.file_list)} valid image-label pairs for camera '{camera}'.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rgb_path, depth_path, label_path = self.file_list[idx]

        # 1. Load Image (X)
        rgb = np.array(Image.open(rgb_path)) / 255.0
        depth = np.array(Image.open(depth_path)) / 1000.0
        depth = np.expand_dims(depth, axis=-1)

        if rgb.shape[0] != self.image_hw[0] or rgb.shape[1] != self.image_hw[1]:
            rgb = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((self.image_hw[1], self.image_hw[0]))) / 255.0
            depth = np.array(Image.fromarray(depth.squeeze()).resize((self.image_hw[1], self.image_hw[0]))) / 1000.0
            depth = np.expand_dims(depth, axis=-1)

        rgbd = np.concatenate([rgb, depth], axis=-1)
        x_image = torch.from_numpy(rgbd).permute(2, 0, 1).float()

        # 2. Load Ground Truth RAW Grasps (Y)
        try:
            # --- THIS IS THE FIX ---
            # Instead of loading one key, we load all the keys we need
            npz_data = np.load(label_path)

            centers = npz_data['centers_2d']        # (N, 2) array for [u, v]
            thetas = npz_data['thetas_rad']         # (N,)   array for theta
            widths = npz_data['widths_2d']          # (N,)   array for width
            depths = npz_data['center_z_depths']  # (N,)   array for depth_offset

            # Our generator needs (N, 5) [u, v, theta, width, depth_offset]
            # We must stack them.
            projected_grasps = np.hstack([
                centers,
                thetas[:, np.newaxis],  # (N,) -> (N, 1)
                widths[:, np.newaxis],  # (N,) -> (N, 1)
                depths[:, np.newaxis]   # (N,) -> (N, 1)
            ])
            # --- END OF FIX ---

        except KeyError as e:
            print(f"\nFATAL ERROR: A key was missing from {label_path}.")
            print(f"Missing key: {e}")
            print(f"Actual keys: {list(np.load(label_path).keys())}")
            return None, (None, None, None) # Signal an error
        except Exception as e:
            # Handle cases where there are 0 grasps
            if 'centers_2d' in str(e): # A guess, but if centers_2d fails, it's probably empty
                projected_grasps = np.array([]) # Create an empty array
            else:
                print(f"Unhandled error loading {label_path}: {e}")
                return None, (None, None, None)


        # 3. Generate Heatmaps On-the-fly
        gt_conf, gt_theta, gt_reg = self.generator.generate_ground_truth(
            grasps_raw=projected_grasps,
            camera_intrinsics=None
        )

        # 4. Downsample the FULL resolution confidence map
        gt_conf_downsampled = F.avg_pool2d(
            gt_conf,
            kernel_size=self.grid_size,
            stride=self.grid_size
        )

        y_conf_low_res = gt_conf_downsampled # [1, H/r, W/r]
        y_theta_low_res = gt_theta           # [6, H/r, W/r]
        y_reg_low_res = gt_reg             # [2, H/r, W/r]

        return x_image, (y_conf_low_res, y_theta_low_res, y_reg_low_res)