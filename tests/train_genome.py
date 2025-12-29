import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

sys.path.append("..")
from core.utils import load_nas_model  # <--- uses the new shared helper
from core.training import train_model
from dataset import GraspNetHeatmapDataset

# --- settings
# paths (relative to tests/ folder)
GENOME_FILE = "best_genome.txt"
SAVE_PATH   = "trained_model_hggd.pth"
DATA_PATH   = "../data/graspnet"

# hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# hyperparameters
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# limits
MAX_BATCHES = 100  # use none for full dataset

def main():
    print(f"starting training on {DEVICE}")

    # 1. build model and load weights uses logic from core/utils.py
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
        ds = GraspNetHeatmapDataset(DATA_PATH, camera='kinect', downsample_factor=8)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
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
        print_every=10
    )

if __name__ == "__main__":
    main()