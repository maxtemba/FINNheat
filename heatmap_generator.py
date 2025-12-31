import numpy as np
import torch

class HeatmapGenerator:
    """
    hggd heatmap generator.
    generates 5 ground truth (loc, cls, theta, width, depth).
    matches section iv-b of the paper.
    """
    def __init__(self, full_hw, grid_size=8, num_angles=6,
                 anchor_w=50.0, anchor_z=20.0, sigma=10):
        self.full_h, self.full_w = full_hw
        self.grid_size = grid_size
        self.grid_h = self.full_h // self.grid_size
        self.grid_w = self.full_w // self.grid_size
        self.num_angles = num_angles # K=6 anchors

        # hyperparams from paper/config
        self.anchor_w = anchor_w   # ref width (pixels)
        self.anchor_z = anchor_z   # ref depth (meters/mm depending on input)
        self.sigma = sigma         # gaussian radius for confidence map

        # theta anchors: [-pi/2, pi/2] split into K bins
        self.theta_range = np.pi
        self.anchor_step = self.theta_range / self.num_angles

    def generate_ground_truth(self, grasps_raw):
        """
        transforms raw grasp list into dense heatmaps.

        args:
            grasps_raw: (N, 5) array [u, v, theta, width, depth_delta]

        returns:
            tuple of 5 torch.Tensors:
            - loc_map:      [1, H, W] (full resolution)
            - cls_mask:     [K, H/r, W/r] (grid resolution)
            - theta_offset: [K, H/r, W/r]
            - width_offset: [K, H/r, W/r]
            - depth_offset: [K, H/r, W/r]
        """

        # 1. initialize blank maps
        loc_map = np.zeros((self.full_h, self.full_w), dtype=np.float32)

        shape_lr = (self.num_angles, self.grid_h, self.grid_w)
        cls_mask_map = np.zeros(shape_lr, dtype=np.float32)
        theta_offset_map = np.zeros(shape_lr, dtype=np.float32)
        width_offset_map = np.zeros(shape_lr, dtype=np.float32)
        depth_offset_map = np.zeros(shape_lr, dtype=np.float32)

        if len(grasps_raw) == 0:
            return self._format(loc_map, cls_mask_map, theta_offset_map, width_offset_map, depth_offset_map)

        # 2. gaussian confidence map (full res) - paper eq (1)
        centers_int = grasps_raw[:, :2].astype(np.int32)
        for center in centers_int:
            if (0 <= center[0] < self.full_w) and (0 <= center[1] < self.full_h):
                self._draw_gaussian(loc_map, center, self.sigma)

        # 3. anchor-based attribute maps (grid res)
        for u, v, theta, width, depth_delta in grasps_raw:
            # downsample to grid
            u_grid = int(u // self.grid_size)
            v_grid = int(v // self.grid_size)

            if not (0 <= u_grid < self.grid_w and 0 <= v_grid < self.grid_h):
                continue

            # a. theta encoding (anchor binning)
            theta = np.clip(theta, -self.theta_range/2 + 1e-8, self.theta_range/2 - 1e-8)
            g_pos, delta_theta = divmod(theta + self.theta_range / 2, self.anchor_step)
            g_pos = int(max(0, min(int(g_pos), self.num_angles - 1)))

            # normalized offset [-0.5, 0.5]
            t_offset = delta_theta / self.anchor_step - 0.5

            # b. width encoding (log space)
            # log(width / anchor_reference)
            safe_width = max(width, 1e-6)
            w_offset = np.log(safe_width / self.anchor_w)

            # c. depth encoding (normalized)
            # delta_z / (anchor_z * 2), clipped to [-0.5, 0.5]
            d_offset = np.clip(depth_delta / (self.anchor_z * 2), -0.5, 0.5)

            # d. accumulate
            cls_mask_map[g_pos, v_grid, u_grid] += 1
            theta_offset_map[g_pos, v_grid, u_grid] += t_offset
            width_offset_map[g_pos, v_grid, u_grid] += w_offset
            depth_offset_map[g_pos, v_grid, u_grid] += d_offset

        # 4. average the offsets (handle overlapping grasps)
        count_map = cls_mask_map + (cls_mask_map == 0) # avoid div/0
        theta_offset_map /= count_map
        width_offset_map /= count_map
        depth_offset_map /= count_map

        # 5. transform counts to soft classification scores
        # 0 -> -1, 1 -> 0.46, inf -> 1.0
        cls_mask_map = 2 / (1 + np.exp(-cls_mask_map)) - 1

        return self._format(loc_map, cls_mask_map, theta_offset_map, width_offset_map, depth_offset_map)

    def _draw_gaussian(self, heatmap, center, radius, k=1):
        """draws 2d gaussian on heatmap (standard centernet/hggd impl)."""
        diameter = 2 * radius + 1
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])
        h, w = heatmap.shape[0:2]

        left, right = min(x, radius), min(w - x, radius + 1)
        top, bottom = min(y, radius), min(h - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def _gaussian2d(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def _format(self, loc, cls, theta, width, depth):
        """converts numpy arrays to torch tensors."""
        return (
            torch.from_numpy(loc).float().unsqueeze(0),
            torch.from_numpy(cls).float(),
            torch.from_numpy(theta).float(),
            torch.from_numpy(width).float(),
            torch.from_numpy(depth).float()
        )