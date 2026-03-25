import os
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from core.models import load_nas_model
from core.training import train_model
from core.dataset import GraspNetHeatmapDataset
from verify_training import save_prediction

# --- settings
# paths (relative to scripts/ folder)
GENOME_FILE = "../genomes/hybrid_genome.txt"
SAVE_PATH        = "outputs/hybrid_model.pth"
SAVE_GENOME_PATH = "outputs/hybrid_genome.txt"
DATA_PATH   = "../data/graspnet"
IMG_CACHE   = "../data/graspnet_img_cache"

# hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# hyperparameters
EPOCHS = 40
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8  # fraction of scenes used for training

# limits
MAX_BATCHES = None  # full dataset

def main():
    print(f"starting training on {DEVICE}")
    os.makedirs("outputs", exist_ok=True)

    # snapshot genome so weights and architecture always stay in sync
    shutil.copy(GENOME_FILE, SAVE_GENOME_PATH)

    # 1. build model and load weights
    try:
        model = load_nas_model(GENOME_FILE, weights_path=None, device=DEVICE)
        print("model built successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    # 2. load data
    if not os.path.exists(DATA_PATH):
        print(f"data not found at {DATA_PATH}")
        return

    try:
        ds = GraspNetHeatmapDataset(DATA_PATH, camera='kinect', downsample_factor=8, img_cache_dir=IMG_CACHE)

        # split by scene to prevent data leakage
        train_idx, val_idx = ds.get_scene_splits(train_frac=TRAIN_SPLIT)
        print(f"train: {len(train_idx)} samples | val: {len(val_idx)} samples")

        loader     = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(Subset(ds, val_idx),   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    except Exception as e:
        print(f"error loading dataset: {e}")
        return

    # 3. train
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    heatmap_cb = lambda m, ep: save_prediction(m, ds, 3000, ep, "outputs/heatmaps", DEVICE)

    train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=SAVE_PATH,
        max_batches=MAX_BATCHES,
        print_every=0,
        log_csv="training_log.csv",
        val_loader=val_loader,
        scheduler=scheduler,
        heatmap_callback=heatmap_cb
    )

if __name__ == "__main__":
    main()