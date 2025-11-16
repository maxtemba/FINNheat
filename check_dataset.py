import os

# --- CONFIG ---
GRASPNET_ROOT = "data/graspnet"
CAMERA = 'kinect'
# --- END CONFIG ---

print(f"--- 🕵️ Debugging GraspNetHeatmapDataset (v5) ---")
print(f"GRASPNET_ROOT: {GRASPNET_ROOT}")

image_dir_base = os.path.join(GRASPNET_ROOT, "scenes")
label_dir_base = os.path.join(GRASPNET_ROOT, f"dataset_{CAMERA}")

if not os.path.isdir(label_dir_base):
    print(f"\n❌ ERROR: Label directory not found at '{label_dir_base}'")
    exit()

scenes = sorted([d for d in os.listdir(label_dir_base) if os.path.isdir(os.path.join(label_dir_base, d))])
print(f"\nFound {len(scenes)} scene folders in '{label_dir_base}'.")
print("--- Checking for the first valid scene ---")

found_valid_scene = False

for label_scene_id in scenes:
    if found_valid_scene:
        break

    print(f"\nScanning Label Scene: '{label_scene_id}'")

    # --- 1. Convert label name to image name ---
    try:
        scene_num = int(label_scene_id.split('_')[-1])
        img_scene_id = f"scene_{scene_num:04d}" # 'scene_0' -> 'scene_0000'
        print(f"  -> Mapped to Image Scene: '{img_scene_id}'")
    except ValueError:
        print(f"  [SKIPPING] Unexpected folder name.")
        continue

    # --- 2. Check for all REQUIRED sub-directories ---
    # This is the new, correct path logic
    img_rgb_dir = os.path.join(image_dir_base, img_scene_id, CAMERA, "rgb")
    img_depth_dir = os.path.join(image_dir_base, img_scene_id, CAMERA, "depth")
    label_scene_folder = os.path.join(label_dir_base, label_scene_id, "grasp_labels")

    if not os.path.isdir(img_rgb_dir):
        print(f"  [SKIPPING] REASON: Image 'rgb' directory not found.")
        print(f"  Path: {img_rgb_dir}")
        continue
    if not os.path.isdir(img_depth_dir):
        print(f"  [SKIPPING] REASON: Image 'depth' directory not found.")
        print(f"  Path: {img_depth_dir}")
        continue
    if not os.path.isdir(label_scene_folder):
        print(f"  [SKIPPING] REASON: 'grasp_labels' folder not found.")
        print(f"  Path: {label_scene_folder}")
        continue

    print(f"  ✅ Found matching 'rgb', 'depth', and 'grasp_labels' folders!")
    found_valid_scene = True

    # --- 3. Now, try to match files inside ---
    # We will loop using the RGB images as the main reference
    try:
        img_files = os.listdir(img_rgb_dir)
        print(f"  Found {len(img_files)} files in 'rgb' dir. Checking first 5...")
    except Exception as e:
        print(f"  [SKIPPING] REASON: Could not read 'rgb' dir: {e}")
        continue

    match_attempts = 0
    matches_found = 0

    for img_name in img_files:
        if not img_name.endswith(".png"): # Assuming '0000.png', '0001.png'
            continue

        if match_attempts >= 5:
            break
        match_attempts += 1

        print(f"\n    --- Attempt {match_attempts} ---")
        print(f"    Found Image File: '{img_name}'")

        img_idx_str = img_name.split('.')[0] # e.g., "0000"
        print(f"    Parsed Image Index: '{img_idx_str}'")

        # --- 4. Build all paths based on this index ---
        grasp_file_name = f"{int(img_idx_str)}_view.npz" # "0000" -> "0_view.npz"
        print(f"    Calculated Label Name: '{grasp_file_name}'")

        rgb_path = os.path.join(img_rgb_dir, img_name)
        depth_path = os.path.join(img_depth_dir, f"{img_idx_str}.png") # Depth name is a guess
        label_path = os.path.join(label_scene_folder, grasp_file_name)

        # --- 5. Check if the pair exists ---
        rgb_exists = os.path.exists(rgb_path)
        depth_exists = os.path.exists(depth_path)
        label_exists = os.path.exists(label_path)

        print(f"    Checking for RGB file:   {rgb_path} (Exists: {rgb_exists})")
        print(f"    Checking for Depth file: {depth_path} (Exists: {depth_exists})")
        print(f"    Checking for Label file: {label_path} (Exists: {label_exists})")

        if rgb_exists and depth_exists and label_exists:
            print(f"    ✅ SUCCESS: Found a valid pair!")
            matches_found += 1
        else:
            print(f"    ❌ FAILURE: This pair is invalid.")
            if not depth_exists:
                print(f"    -> HINT: Is the depth file name '{img_idx_str}.png' correct?")
                # Let's check for the other common name
                depth_path_alt = os.path.join(img_depth_dir, f"{img_idx_str}.depth.png")
                print(f"    -> Trying alternate: {depth_path_alt} (Exists: {os.path.exists(depth_path_alt)})")


    print("\n--- 🏁 Scene Check Complete ---")
    if matches_found > 0:
        print(f"✅ RESULT: Found {matches_found} valid pairs in this scene!")
    else:
        print("❌ RESULT: 0 valid pairs found.")

if not found_valid_scene:
    print("\n--- 🏁 Debugging Complete ---")
    print("❌ RESULT: 0 valid scenes found.")