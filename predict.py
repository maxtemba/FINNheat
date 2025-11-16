import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from model import FINNCompatibleGHM_MultiOutput # Must be the same model

# --- 1. CONFIGURATION ---
MODEL_PATH = "trained_model.pth"
# --- 2. SET YOUR IMAGE PATH HERE ---
IMAGE_PATH = "img.jpeg" # <--- UPDATE THIS
# (Optional) If you have a depth image, put the path here
DEPTH_PATH = None
OUTPUT_NAME = "prediction_result.png"

# These MUST match your training settings
IMAGE_HW = (360, 640) # (Height, Width)
DOWNSAMPLE_FACTOR = 8
NUM_ANGLES = 6

def draw_grasps_on_image(ax, grasps_raw, h, w):
    """Helper function to plot grasps on a matplotlib axis."""
    for u, v, theta, width, depth_offset in grasps_raw:
        ax.plot(u, v, 'g.') # Green dot for center
        half_w = width / 2.0
        dx = half_w * np.cos(theta)
        dy = half_w * np.sin(theta)

        p1_u, p1_v = u - dx, v - dy
        p2_u, p2_v = u + dx, v + dy

        ax.plot([p1_u, p2_u], [p1_v, p2_v], 'r-', linewidth=2) # Red line for gripper

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0) # Inverted y-axis for images
    ax.axis('off')

def predict():
    print(f"--- Running Prediction ---")

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ ERROR: Image file not found at {IMAGE_PATH}")
        print("Please update the 'IMAGE_PATH' variable in this script.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        print("Please run 'train.py' first.")
        return

    device = torch.device("cpu")

    # 1. Load Model
    print(f"Loading trained model from {MODEL_PATH}...")
    model = FINNCompatibleGHM_MultiOutput().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Load and Pre-process Image
    print(f"Loading image from {IMAGE_PATH}...")
    rgb_img = Image.open(IMAGE_PATH).convert("RGB")
    original_size = rgb_img.size # (W, H)

    # Resize to the model's required input size (W, H)
    rgb_img_resized = rgb_img.resize((IMAGE_HW[1], IMAGE_HW[0]))
    rgb = np.array(rgb_img_resized) / 255.0

    if DEPTH_PATH:
        # If you provide a depth map, load and process it
        depth_img = Image.open(DEPTH_PATH)
        depth_img_resized = depth_img.resize((IMAGE_HW[1], IMAGE_HW[0]))
        depth = np.array(depth_img_resized) / 1000.0 # Assuming mm to meters
        depth = np.expand_dims(depth, axis=-1)
    else:
        # Create a "fake" depth map (all ones = 1 meter)
        print("No depth image provided, creating a fake depth map (1m).")
        depth = np.ones((IMAGE_HW[0], IMAGE_HW[1], 1), dtype=np.float32)

    # Stack to (H, W, 4) and convert to (B, C, H, W) tensor
    rgbd = np.concatenate([rgb, depth], axis=-1)
    x_image = torch.from_numpy(rgbd).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # 3. Run Inference
    print("Running model...")
    with torch.no_grad():
        pred_conf, pred_theta, pred_reg = model(x_image)

    # 4. Post-process the output
    # Apply sigmoid to confidence logits
    confidence_map = torch.sigmoid(pred_conf).squeeze().cpu().numpy() # (H/8, W/8)
    theta_map = pred_theta.squeeze().cpu().numpy() # (6, H/8, W/8)
    reg_map = pred_reg.squeeze().cpu().numpy() # (2, H/8, W/8)

    # 5. Find the single best grasp
    # Find the pixel (v, u) with the highest confidence
    v, u = np.unravel_index(np.argmax(confidence_map), confidence_map.shape)

    # Get all predictions at that single pixel
    confidence_score = confidence_map[v, u]

    # Find the most likely angle class
    angle_class = np.argmax(theta_map[:, v, u])
    # Convert class index back to an angle in radians
    anchors = np.linspace(-np.pi/2, np.pi/2, NUM_ANGLES + 1)[:-1]
    angle_rad = anchors[angle_class]

    # Get the predicted width and depth
    width = reg_map[0, v, u]
    depth_offset = reg_map[1, v, u] # Not used for 2D drawing

    # 6. Scale coordinates back to original image size
    # Model output is (45, 80), so we scale by 8
    u_img = (u * DOWNSAMPLE_FACTOR) + (DOWNSAMPLE_FACTOR // 2)
    v_img = (v * DOWNSAMPLE_FACTOR) + (DOWNSAMPLE_FACTOR // 2)

    # Create a 1-element grasp array for the drawing function
    # [u, v, theta, width, depth_offset]
    grasp_data = np.array([[u_img, v_img, angle_rad, width, 0.0]])

    print(f"\n✅ Prediction Complete:")
    print(f"  Best Grasp Found with {confidence_score*100:.1f}% confidence.")
    print(f"  Location (u,v): ({u_img}, {v_img})")
    print(f"  Angle: {np.degrees(angle_rad):.1f}°")
    print(f"  Width: {width:.1f} pixels")

    # 7. Create and Save Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Original Image + Grasp
    ax1.imshow(rgb_img_resized) # Show the resized image it saw
    draw_grasps_on_image(ax1, grasp_data, IMAGE_HW[0], IMAGE_HW[1])
    ax1.set_title("Predicted Best Grasp")

    # Plot 2: Predicted Heatmap
    im = ax2.imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
    ax2.plot(u, v, 'c+', markersize=10) # Mark the chosen pixel
    ax2.set_title("Model's Confidence Heatmap")
    ax2.axis('off')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_NAME)
    print(f"\nSuccess! Visualization saved to {OUTPUT_NAME}")

if __name__ == "__main__":
    predict()