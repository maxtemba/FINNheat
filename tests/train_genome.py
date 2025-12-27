import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import ast
import os
import sys

# Minimal import fix (Go up one level from 'tests/')
sys.path.append("..")

from core.models import NAS_GHM_Model
from core.training import train_model
from dataset import GraspNetHeatmapDataset

# ==============================================================================
# ⚙️ USER SETTINGS
# ==============================================================================
# Paths (Relative to 'tests/' folder)
GENOME_FILE = "best_genome.txt"
SAVE_PATH   = "trained_model_hggd.pth"
DATA_PATH   = "../data/graspnet"

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ⚡ LIMITER SETTING
MAX_BATCHES = 2000  # Set to None for full training

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"🚀 Starting Training on {DEVICE}")

    # 1. Load Genome
    if not os.path.exists(GENOME_FILE):
        print(f"❌ Error: Cannot find {GENOME_FILE}")
        return
    with open(GENOME_FILE, "r") as f:
        genome = ast.literal_eval(f.read().strip())
    print(f"🧬 Genome: {genome}")

    # 2. Build Model
    model = NAS_GHM_Model(genome).to(DEVICE)

    # 3. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data not found at {DATA_PATH}")
        return
    ds = GraspNetHeatmapDataset(DATA_PATH, camera='kinect', downsample_factor=8)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=SAVE_PATH,
        max_batches=MAX_BATCHES,
        print_every=10  # <--- Updates every 10 batches
    )

if __name__ == "__main__":
    main()