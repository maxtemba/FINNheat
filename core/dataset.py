import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# relative import since they are now siblings in core/
from .heatmap_generator import HeatmapGenerator

class GraspNetHeatmapDataset(Dataset):
    """
    clean pytorch dataset for graspnet.
    handles file indexing, resizing, and hggd ground truth generation.
    """
    def __init__(self, root, camera='kinect', downsample_factor=8, img_cache_dir=None):
        self.root = root
        self.camera = camera
        self.factor = downsample_factor
        self.hw = (360, 640) # target input size (h, w)
        self.img_cache_dir = img_cache_dir

        # physics engine for heatmaps
        self.generator = HeatmapGenerator(
            full_hw=self.hw, grid_size=self.factor, num_angles=6,
            anchor_w=50.0, anchor_z=20.0, sigma=10
        )

        # build file index once (also populates self.scene_of)
        self.scene_of = []
        self.files = self._build_index()
        print(f"found {len(self.files)} valid pairs for camera '{camera}'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb_path, depth_path, label_path = self.files[idx]

        # 1. load and resize images (from npy cache if available)
        if self.img_cache_dir is not None:
            rgb   = np.load(os.path.join(self.img_cache_dir, f"{idx:06d}_r.npy")).astype(np.float32) / 255.0
            depth = np.load(os.path.join(self.img_cache_dir, f"{idx:06d}_d.npy")).astype(np.float32) / 1000.0
            h_old, w_old = self.hw
        else:
            rgb, h_old, w_old = self._load_img(rgb_path, is_depth=False)
            depth, _, _       = self._load_img(depth_path, is_depth=True)

        # 2. combine to input tensor [4, H, W]
        rgbd = np.concatenate([rgb, depth[..., None]], axis=-1)
        x_in = torch.from_numpy(rgbd).permute(2, 0, 1).float()

        # 3. load and process labels
        grasps = self._process_labels(label_path, depth, h_old, w_old)

        # 4. generate ground truth heatmaps
        gt_loc, gt_cls, gt_theta, gt_width, gt_depth = \
            self.generator.generate_ground_truth(grasps)

        # 5. downsample localization map (max_pool keeps peaks sharp)
        gt_loc = F.max_pool2d(gt_loc, kernel_size=self.factor, stride=self.factor)

        return x_in, (gt_loc, gt_cls, gt_theta, gt_width, gt_depth)

    # --- helper methods ---

    def get_scene_splits(self, train_frac=0.8, seed=42):
        """splits dataset indices by scene to prevent data leakage between train and val."""
        rng = np.random.default_rng(seed)
        scene_arr   = np.array(self.scene_of)
        unique_scenes = np.unique(scene_arr)
        rng.shuffle(unique_scenes)
        n_train = max(1, int(len(unique_scenes) * train_frac))
        train_scenes = set(unique_scenes[:n_train].tolist())
        val_scenes   = set(unique_scenes[n_train:].tolist())
        train_idx = [i for i, s in enumerate(scene_arr) if s in train_scenes]
        val_idx   = [i for i, s in enumerate(scene_arr) if s in val_scenes]
        return train_idx, val_idx

    def _build_index(self):
        """scans directories to match rgb, depth, and label files."""
        valid = []
        base_img = os.path.join(self.root, "scenes")
        base_lbl = os.path.join(self.root, f"dataset_{self.camera}")

        if not os.path.exists(base_lbl): return []

        # find all scene folders
        scenes = sorted([d for d in os.listdir(base_lbl) if os.path.isdir(os.path.join(base_lbl, d))])

        for s_idx, s_id in enumerate(scenes):
            try:
                # scene_0000 -> 0
                s_num = int(s_id.split('_')[-1])
                img_dir = os.path.join(base_img, f"scene_{s_num:04d}", self.camera)
                lbl_dir = os.path.join(base_lbl, s_id, "grasp_labels")
            except: continue

            if not os.path.exists(img_dir): continue

            # match files inside the scene
            rgb_dir = os.path.join(img_dir, "rgb")
            dep_dir = os.path.join(img_dir, "depth")

            if not os.path.exists(rgb_dir): continue

            for name in os.listdir(rgb_dir):
                if not name.endswith(".png"): continue

                idx = name.split('.')[0]
                paths = (
                    os.path.join(rgb_dir, name),
                    os.path.join(dep_dir, name),
                    os.path.join(lbl_dir, f"{int(idx)}_view.npz")
                )

                if all(os.path.exists(p) for p in paths):
                    valid.append(paths)
                    self.scene_of.append(s_idx)
        return valid

    def _load_img(self, path, is_depth):
        """loads image, resizes if needed, and normalizes."""
        img = Image.open(path)
        w_old, h_old = img.size

        if (h_old, w_old) != self.hw:
            method = Image.NEAREST if is_depth else Image.BILINEAR
            img = img.resize((self.hw[1], self.hw[0]), method)

        arr = np.array(img).astype(np.float32)
        if is_depth:
            return arr / 1000.0, h_old, w_old # mm to meters
        return arr / 255.0, h_old, w_old

    def _process_labels(self, path, depth_map, h_old, w_old):
        """loads npz, scales coords, and computes depth delta."""
        try:
            data = np.load(path)
            centers = data['centers_2d'].astype(np.float32)
            widths  = data['widths_2d'].astype(np.float32)
            z_depth = data['center_z_depths'].astype(np.float32) / 1000.0

            # calculate scale factors
            sy, sx = self.hw[0] / h_old, self.hw[1] / w_old

            if sx != 1.0 or sy != 1.0:
                centers[:, 0] *= sx
                centers[:, 1] *= sy
                widths *= sx

            # calculate depth delta (grasp z - surface z)
            uv = centers.astype(int)
            np.clip(uv[:, 0], 0, self.hw[1]-1, out=uv[:, 0])
            np.clip(uv[:, 1], 0, self.hw[0]-1, out=uv[:, 1])

            surface_z = depth_map[uv[:, 1], uv[:, 0]]
            deltas = z_depth - surface_z

            # return format: [u, v, theta, width, depth_delta]
            return np.hstack([
                centers,
                data['thetas_rad'][:, None],
                widths[:, None],
                deltas[:, None]
            ])
        except:
            # return empty if file is corrupt
            return np.zeros((0, 5), dtype=np.float32)