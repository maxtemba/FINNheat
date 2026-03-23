import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.models import load_nas_model
from core.training import train_model
from core.dataset import GraspNetHeatmapDataset

# --- settings
# paths (relative to scripts/ folder)
GENOME_FILE = "../genomes/best_genome.txt"
SAVE_PATH   = "outputs/trained_model.pth"
DATA_PATH   = "../data/graspnet"
IMG_CACHE   = "../data/graspnet_img_cache"

# hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# limits
MAX_BATCHES = 5000  # none for full dataset

def main():
    print(f"starting training on {DEVICE}")
    os.makedirs("outputs", exist_ok=True)

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
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    except Exception as e:
        print(f"error loading dataset: {e}")
        return

    # 3. train
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=SAVE_PATH,
        max_batches=MAX_BATCHES,
        print_every=0,
        log_csv="training_log.csv"
    )

if __name__ == "__main__":
    main()