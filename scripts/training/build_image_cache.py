import os
import numpy as np
from PIL import Image
from multiprocessing import Pool

import config
from core.dataset import GraspNetHeatmapDataset

HW = (360, 640)  # cache content shape — must match GraspNetHeatmapDataset.hw


def convert_sample(args):
    idx, rgb_path, dep_path = args

    out_r = os.path.join(config.IMG_CACHE_DIR, f"{idx:06d}_r.npy")
    out_d = os.path.join(config.IMG_CACHE_DIR, f"{idx:06d}_d.npy")
    if os.path.exists(out_r) and os.path.exists(out_d):
        return

    # rgb: resize to 360x640, save as uint8
    rgb = Image.open(rgb_path)
    if (rgb.size[1], rgb.size[0]) != HW:
        rgb = rgb.resize((HW[1], HW[0]), Image.BILINEAR)
    np.save(out_r, np.array(rgb))

    # depth: resize to 360x640, save as uint16 (mm, raw)
    dep = Image.open(dep_path)
    if (dep.size[1], dep.size[0]) != HW:
        dep = dep.resize((HW[1], HW[0]), Image.NEAREST)
    np.save(out_d, np.array(dep))


def main():
    os.makedirs(config.IMG_CACHE_DIR, exist_ok=True)

    ds = GraspNetHeatmapDataset(config.DATA_DIR, camera=config.CAMERA, downsample_factor=config.DOWNSAMPLE)
    args = [(i, rgb, dep) for i, (rgb, dep, _) in enumerate(ds.files)]

    already = sum(1 for i in range(len(args))
                  if os.path.exists(os.path.join(config.IMG_CACHE_DIR, f"{i:06d}_r.npy")))
    print(f"converting {len(args) - already} samples ({already} already cached)...")

    with Pool(config.NUM_WORKERS) as p:
        p.map(convert_sample, args)

    print(f"done. cache at '{config.IMG_CACHE_DIR}'")


if __name__ == "__main__":
    main()
