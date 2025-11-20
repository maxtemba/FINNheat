import numpy as np
import torch

class HeatmapGenerator:
    """
    HGGD-compliant Heatmap Generator.
    Generates training targets compatible with AnchorGraspNet and your NAS model.
    """
    def __init__(self, full_hw, grid_size=8, num_angles=6,
                 anchor_w=50.0, anchor_z=20.0, sigma=10):
        self.full_h, self.full_w = full_hw
        self.grid_size = grid_size
        self.grid_h = self.full_h // self.grid_size
        self.grid_w = self.full_w // self.grid_size
        self.num_angles = num_angles # K=6

        # HGGD Hyperparameters
        self.anchor_w = anchor_w   # Reference width (pixels)
        self.anchor_z = anchor_z   # Reference depth (mm or meters, must match input scale)
        self.sigma = sigma         # Gaussian radius for loc_map

        # Theta Anchors: [-pi/2, pi/2] split into K bins
        self.theta_range = np.pi
        self.anchor_step = self.theta_range / self.num_angles

    def generate_ground_truth(self, grasps_raw):
        """
        Generates 5 ground truth tensors from raw grasp data.

        Args:
            grasps_raw: (N, 5) array [u, v, theta, width, depth_delta]

        Returns:
            tuple of 5 torch.Tensors:
            - loc_map:      [1, H, W] (Full Resolution)
            - cls_mask:     [K, H/r, W/r] (Grid Resolution)
            - theta_offset: [K, H/r, W/r]
            - width_offset: [K, H/r, W/r]
            - depth_offset: [K, H/r, W/r]
        """

        # 1. Initialize Maps
        # Location Map: [H, W] -> Will become [1, H, W]
        loc_map = np.zeros((self.full_h, self.full_w), dtype=np.float32)

        # Attribute Maps: [K, H/r, W/r]
        shape_lr = (self.num_angles, self.grid_h, self.grid_w)
        cls_mask_map = np.zeros(shape_lr, dtype=np.float32)
        theta_offset_map = np.zeros(shape_lr, dtype=np.float32)
        width_offset_map = np.zeros(shape_lr, dtype=np.float32)
        depth_offset_map = np.zeros(shape_lr, dtype=np.float32)

        if len(grasps_raw) == 0:
            return self.format_outputs(loc_map, cls_mask_map, theta_offset_map, width_offset_map, depth_offset_map)

        # 2. Gaussian Confidence Map (Full Resolution)
        # HGGD uses a fixed sigma (default 10) defined in config.
        centers_int = grasps_raw[:, :2].astype(np.int32)
        for center in centers_int:
            # Check bounds before drawing
            if (0 <= center[0] < self.full_w) and (0 <= center[1] < self.full_h):
                loc_map = self.draw_umich_gaussian(loc_map, center, self.sigma)

        # 3. Anchor-based Attribute Maps (Grid Resolution)
        # Loop through every grasp to populate the anchor grids
        for u, v, theta, width, depth_delta in grasps_raw:
            # Downsample coords to grid
            u_grid = int(u // self.grid_size)
            v_grid = int(v // self.grid_size)

            # Bounds Check
            if not (0 <= u_grid < self.grid_w and 0 <= v_grid < self.grid_h):
                continue

            # --- A. Theta Encoding (Anchor Binning) ---
            # Clip to [-pi/2, pi/2] to avoid boundary errors
            theta = np.clip(theta, -self.theta_range/2 + 1e-8, self.theta_range/2 - 1e-8)

            # Calculate which bin (anchor) this angle belongs to
            # Formula: (theta + pi/2) / step_size
            g_pos, delta_theta = divmod(theta + self.theta_range / 2, self.anchor_step)
            g_pos = int(g_pos)

            # Safety clamp for g_pos (rare edge cases)
            g_pos = max(0, min(g_pos, self.num_angles - 1))

            # Normalized Theta Offset [-0.5, 0.5]
            # (angle_remainder / step_size) - 0.5
            t_offset = delta_theta / self.anchor_step - 0.5

            # --- B. Width Encoding (Log Space) ---
            # log(width / anchor_reference)
            # Avoid log(0) or negative width
            safe_width = max(width, 1e-6)
            w_offset = np.log(safe_width / self.anchor_w)

            # --- C. Depth Encoding (Normalized) ---
            # delta_z / (anchor_z * 2), clipped to [-0.5, 0.5]
            # Note: anchor_z * 2 is the "range" covered by the anchor
            d_offset = depth_delta / (self.anchor_z * 2)
            d_offset = np.clip(d_offset, -0.5, 0.5)

            # --- D. Accumulate ---
            # Add values to the specific anchor channel at this grid location.
            # Multiple grasps might map to the same bin; we sum them now and average later.
            cls_mask_map[g_pos, v_grid, u_grid] += 1
            theta_offset_map[g_pos, v_grid, u_grid] += t_offset
            width_offset_map[g_pos, v_grid, u_grid] += w_offset
            depth_offset_map[g_pos, v_grid, u_grid] += d_offset

        # 4. Average the offsets
        # If multiple grasps hit the same (u, v, anchor_idx), average their offsets.
        count_map = cls_mask_map + (cls_mask_map == 0) # Add epsilon to avoid div by zero
        theta_offset_map /= count_map
        width_offset_map /= count_map
        depth_offset_map /= count_map

        # 5. Transform Classification Mask
        # HGGD uses a specific sigmoid transform to map counts to soft labels:
        # 0 count -> -1 score
        # 1 count -> ~0.46 score
        # High count -> 1.0 score
        cls_mask_map = 2 / (1 + np.exp(-cls_mask_map)) - 1

        return self.format_outputs(loc_map, cls_mask_map, theta_offset_map, width_offset_map, depth_offset_map)

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        """
        Draws a 2D Gaussian on the heatmap at the specified center.
        Standard implementation used in CornerNet/CenterNet/HGGD.
        """
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def format_outputs(self, loc, cls, theta, width, depth):
        # Convert numpy arrays to PyTorch Tensors
        # loc: [1, H, W]
        # others: [K, H/r, W/r]
        return (
            torch.from_numpy(loc).float().unsqueeze(0),
            torch.from_numpy(cls).float(),
            torch.from_numpy(theta).float(),
            torch.from_numpy(width).float(),
            torch.from_numpy(depth).float()
        )