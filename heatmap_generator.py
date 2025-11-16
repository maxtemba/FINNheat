import numpy as np
import torch
import torch.nn.functional as F
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class HeatmapGenerator:
    """
    Implements the heatmap generation logic from the HGGD paper.
    - Gaussian encoding for the confidence map.
    - Grid-based strategy for attribute maps.
    """
    def __init__(self, full_hw, grid_size, num_angles=6):
        self.full_h, self.full_w = full_hw
        self.grid_size = grid_size
        self.grid_h = self.full_h // self.grid_size
        self.grid_w = self.full_w // self.grid_size
        self.num_angles = num_angles

        # Create anchor angles for classification
        self.anchors = np.linspace(-np.pi/2, np.pi/2, num_angles + 1)[:-1]

    def generate_ground_truth(self, grasps_raw, camera_intrinsics):
        """
        Main function to generate all ground-truth heatmaps from raw grasp data.

        grasps_raw: (N, 5) array [u, v, theta, width, depth_offset]
        """

        # 1. Initialize empty maps
        gt_conf_map_full = np.zeros((self.full_h, self.full_w), dtype=np.float32)

        # Grid-based maps (H/r, W/r, C)
        gt_theta_map = np.zeros((self.grid_h, self.grid_w, self.num_angles), dtype=np.float32)
        gt_width_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        gt_depth_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Counters for averaging
        width_counters = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        depth_counters = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

        if grasps_raw.shape[0] == 0:
            # No grasps in this view
            return self.format_outputs(gt_conf_map_full, gt_theta_map, gt_width_map, gt_depth_map)

        # --- Gaussian Encoding for Confidence Map (Full Res) ---
        for u, v, theta, width, depth_offset in grasps_raw:
            u, v = int(u), int(v)
            if 0 <= u < self.full_w and 0 <= v < self.full_h:
                # Use grasp width to determine sigma, as per paper
                sigma = max(1.0, width / 10.0)
                gt_conf_map_full = self.draw_gaussian(gt_conf_map_full, (u, v), sigma)

        # --- Grid-Based Strategy for Attributes (Low Res) ---
        for u, v, theta, width, depth_offset in grasps_raw:
            u_grid = int(u / self.grid_size)
            v_grid = int(v / self.grid_size)

            if 0 <= u_grid < self.grid_w and 0 <= v_grid < self.grid_h:
                # 1. Theta Map (Angle Classification)
                angle_diff = np.abs(self.anchors - theta)
                angle_diff = np.minimum(angle_diff, np.pi - angle_diff) # Handle wrap-around
                nearest_anchor_idx = np.argmin(angle_diff)
                gt_theta_map[v_grid, u_grid, nearest_anchor_idx] = 1.0 # One-hot

                # 2. Width Map (Average Regression)
                gt_width_map[v_grid, u_grid] += width
                width_counters[v_grid, u_grid] += 1

                # 3. Depth Map (Average Regression)
                gt_depth_map[v_grid, u_grid] += depth_offset
                depth_counters[v_grid, u_grid] += 1

        # Normalize the attribute maps
        gt_width_map[width_counters > 0] /= width_counters[width_counters > 0]
        gt_depth_map[depth_counters > 0] /= depth_counters[depth_counters > 0]

        return self.format_outputs(gt_conf_map_full, gt_theta_map, gt_width_map, gt_depth_map)

    def draw_gaussian(self, heatmap, center, sigma):
        u, v = center
        h, w = heatmap.shape
        ys, xs = np.indices(heatmap.shape)
        dist_sq = (ys - v) ** 2 + (xs - u) ** 2
        g = np.exp(-dist_sq / (2 * sigma**2))

        # This is faster than looping
        heatmap = np.maximum(heatmap, g)
        return heatmap

    def format_outputs(self, conf_map, theta_map, width_map, depth_map):
        # Convert to Tensors and correct shapes

        gt_conf = torch.from_numpy(conf_map).float().unsqueeze(0)        # [1, H, W]
        gt_theta = torch.from_numpy(theta_map).permute(2, 0, 1).float() # [6, H/r, W/r]
        gt_width = torch.from_numpy(width_map).float().unsqueeze(0)      # [1, H/r, W/r]
        gt_depth = torch.from_numpy(depth_map).float().unsqueeze(0)      # [1, H/r, W/r]

        gt_reg = torch.cat([gt_width, gt_depth], dim=0)                 # [2, H/r, W/r]

        return gt_conf, gt_theta, gt_reg

# ==============================================================================
# === NEW TEST FUNCTION AND HELPER =============================================
# ==============================================================================

def draw_grasps_on_image(ax, grasps_raw, h, w):
    """Plots the raw grasps on a matplotlib axis."""
    for u, v, theta, width, depth_offset in grasps_raw:
        # Grasp center
        ax.plot(u, v, 'g.') # Green dot

        # Calculate gripper points
        half_w = width / 2.0
        dx = half_w * np.cos(theta)
        dy = half_w * np.sin(theta)

        p1_u, p1_v = u - dx, v - dy
        p2_u, p2_v = u + dx, v + dy

        # Draw gripper line
        ax.plot([p1_u, p2_u], [p1_v, p2_v], 'r-', linewidth=2) # Red line

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0) # Inverted y-axis for images
    ax.axis('off')


def test_generator():
    """
    Runs a test of the HeatmapGenerator with random grasp data
    and saves the output dashboard as an image.
    """
    print("--- Running HeatmapGenerator Test ---")

    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib not found. Cannot save test images.")
        print("   Please run: pip install matplotlib")
        return

    # 1. Config
    IMG_H, IMG_W = 360, 640
    GRID_SIZE = 8
    NUM_GRASPS = 10

    # 2. Create Generator
    generator = HeatmapGenerator(full_hw=(IMG_H, IMG_W), grid_size=GRID_SIZE)

    # 3. Create Fake Grasp Data (N, 5)
    # [u, v, theta, width, depth_offset]
    us = np.random.randint(IMG_W * 0.1, IMG_W * 0.9, (NUM_GRASPS, 1)) # Keep grasps away from edge
    vs = np.random.randint(IMG_H * 0.1, IMG_H * 0.9, (NUM_GRASPS, 1))
    thetas = np.random.uniform(-np.pi/2, np.pi/2, (NUM_GRASPS, 1))
    widths = np.random.uniform(20, 100, (NUM_GRASPS, 1))
    depths = np.random.uniform(0.01, 0.1, (NUM_GRASPS, 1))

    fake_grasps = np.hstack([us, vs, thetas, widths, depths]).astype(np.float32)
    print(f"Generated {NUM_GRASPS} random grasps.")

    # 4. Run Generator
    gt_conf, gt_theta, gt_reg = generator.generate_ground_truth(fake_grasps, None)

    print(f"Generator outputs:")
    print(f"  Confidence Map (Full): {gt_conf.shape}")
    print(f"  Theta Map (Grid):    {gt_theta.shape}")
    print(f"  Regression Map (Grid): {gt_reg.shape}")

    # 5. Create visualizations

    # Create a blank white image
    blank_image = np.ones((IMG_H, IMG_W, 3), dtype=np.float32)

    # Squeeze batch/channel dim and move to numpy
    conf_map_np = gt_conf.squeeze().numpy()

    # Convert theta from one-hot to class index map
    theta_map_np = torch.argmax(gt_theta, dim=0).squeeze().numpy()

    # Get just the width map
    width_map_np = gt_reg.squeeze(0)[0].numpy()

    # 6. Create and Save the 2x2 Dashboard Plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("HeatmapGenerator Test Dashboard", fontsize=16)

    # --- Plot 1: Original Image + Grasps ---
    ax[0, 0].imshow(blank_image)
    draw_grasps_on_image(ax[0, 0], fake_grasps, IMG_H, IMG_W)
    ax[0, 0].set_title("Original Image + Raw Grasps")

    # --- Plot 2: Confidence Map ---
    im_conf = ax[0, 1].imshow(conf_map_np, cmap='hot', vmin=0, vmax=1)
    ax[0, 1].set_title("Generated Confidence Map (Full Res)")
    ax[0, 1].axis('off')
    fig.colorbar(im_conf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    # --- Plot 3: Theta Map ---
    im_theta = ax[1, 0].imshow(theta_map_np, cmap='viridis', vmin=0, vmax=generator.num_angles - 1)
    ax[1, 0].set_title("Generated Theta Map (Grid)")
    ax[1, 0].axis('off')
    fig.colorbar(im_theta, ax=ax[1, 0], fraction=0.046, pad=0.04)

    # --- Plot 4: Width Map ---
    im_width = ax[1, 1].imshow(width_map_np, cmap='viridis')
    ax[1, 1].set_title("Generated Width Map (Grid)")
    ax[1, 1].axis('off')
    fig.colorbar(im_width, ax=ax[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("test_generator_dashboard.png")

    print("\n✅ Success! Saved test dashboard:")
    print("  - test_generator_dashboard.png")
    print("  (Check this image to see the outputs)")


if __name__ == "__main__":
    # This block runs when you call 'python heatmap_generator.py'
    test_generator()