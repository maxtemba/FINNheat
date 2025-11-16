import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import os

from dataset import GraspNetHeatmapDataset
from model import FINNCompatibleGHM_MultiOutput

# --- Hyperparameters ---
GRASPNET_ROOT = "data/graspnet" # This path should be correct now
CAMERA = 'kinect'
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5
DOWNSAMPLE_FACTOR = 8
SAVE_PATH = "trained_model.pth"
# -------------------------

def combined_loss(pred_conf, pred_theta, pred_reg,
                  true_conf, true_theta, true_reg):

    # 1. Confidence Loss (BCEWithLogitsLoss)
    conf_loss_fn = nn.BCEWithLogitsLoss()
    loss_conf = conf_loss_fn(pred_conf, true_conf)

    mask = (true_conf > 0.8).float()

    # 2. Theta Loss (CrossEntropyLoss)
    true_theta_indices = torch.argmax(true_theta, dim=1)

    theta_loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss_theta_unmasked = theta_loss_fn(pred_theta, true_theta_indices)

    loss_theta_masked = loss_theta_unmasked * mask.squeeze(1)
    if mask.sum() > 0:
        loss_theta = loss_theta_masked.sum() / mask.sum()
    else:
        loss_theta = torch.tensor(0.0, device=pred_conf.device)

    # 3. Regression Loss (L1Loss for width, depth)
    attr_loss_fn = nn.L1Loss(reduction='none')

    loss_attr_unmasked = attr_loss_fn(pred_reg, true_reg)
    loss_attr_masked = loss_attr_unmasked * mask

    if mask.sum() > 0:
        loss_attr = loss_attr_masked.sum() / mask.sum()
    else:
        loss_attr = torch.tensor(0.0, device=pred_conf.device)

    # 4. Total Loss
    w_conf, w_theta, w_attr = 1.0, 0.5, 1.0
    total_loss = (w_conf * loss_conf) + (w_theta * loss_theta) + (w_attr * loss_attr)

    return total_loss, loss_conf, loss_theta, loss_attr

def train():
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FINNCompatibleGHM_MultiOutput().to(device)

    try:
        train_dataset = GraspNetHeatmapDataset(GRASPNET_ROOT, camera=CAMERA, downsample_factor=DOWNSAMPLE_FACTOR)
    except FileNotFoundError:
        print(f"ERROR: Could not find dataset at {GRASPNET_ROOT}")
        return

    if len(train_dataset) == 0:
        print("ERROR: Dataset 0 files. Check dataset.py and paths.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch = 0

        for i, (x_image, (y_conf, y_theta, y_reg)) in enumerate(train_loader):
            if x_image is None:
                print("Error: Dataset returned None. Check dataset.py for .npz key errors.")
                continue

            x_image = x_image.to(device)
            y_conf = y_conf.to(device)
            y_theta = y_theta.to(device)
            y_reg = y_reg.to(device)

            pred_conf, pred_theta, pred_reg = model(x_image)

            loss, loss_c, loss_t, loss_a = combined_loss(
                pred_conf, pred_theta, pred_reg,
                y_conf, y_theta, y_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

            if i % 20 == 0:
                print(f"E {epoch+1}, B {i}/{len(train_loader)}, Loss: {loss.item():.4f} (C: {loss_c.item():.4f}, T: {loss_t.item():.4f}, A: {loss_a.item():.4f})")

        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {total_loss_epoch / len(train_loader):.4f} ---")
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training complete. Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()