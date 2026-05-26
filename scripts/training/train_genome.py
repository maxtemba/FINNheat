import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
from core.models import load_nas_model
from core.training import train_model
from core.dataset import GraspNetHeatmapDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

MAX_BATCHES = None  # full dataset


def main():
    print(f"starting training on {DEVICE}")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. build model and load weights
    try:
        model = load_nas_model(config.ACTIVE_GENOME, weights_path=None, device=DEVICE)
        print("model built successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    # 2. load data
    if not os.path.exists(config.DATA_DIR):
        print(f"data not found at {config.DATA_DIR}")
        return

    try:
        ds = GraspNetHeatmapDataset(config.DATA_DIR, camera=config.CAMERA,
                                    downsample_factor=config.DOWNSAMPLE, img_cache_dir=config.IMG_CACHE_DIR)

        # split by scene to prevent data leakage
        train_idx, val_idx = ds.get_scene_splits(train_frac=config.TRAIN_SPLIT)
        print(f"train: {len(train_idx)} samples | val: {len(val_idx)} samples")

        loader     = DataLoader(Subset(ds, train_idx), batch_size=config.BATCH_SIZE, shuffle=True,
                                num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(Subset(ds, val_idx),   batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    except Exception as e:
        print(f"error loading dataset: {e}")
        return

    # 3. train
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=config.EPOCHS,
        save_path=config.ACTIVE_WEIGHTS,
        max_batches=MAX_BATCHES,
        print_every=0,
        log_csv=config.TRAINING_LOG,
        val_loader=val_loader,
        scheduler=scheduler,
        heatmap_callback=None
    )


if __name__ == "__main__":
    main()
