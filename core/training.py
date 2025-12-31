import os
import csv
import torch
import torch.nn.functional as F

def hggd_loss(preds, targets, device):
    """
    calculates error for the Grasp Heatmap Model (GHM).
    it implements the loss function from the HGGD paper and properly deals with 99% of background.

    :param preds: output from the model (loc, cls, theta, width, depth).
    :param targets: models ground truth (loc, cls, theta, width, depth).
    :param device: CPU/GPU.
    :return: single number as total loss/error.
    """

    pred_loc, pred_cls, pred_theta, pred_width, pred_depth = preds
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets
    eps = 1e-6 # log(0) errors

    # localization loss
    pred_loc = torch.clamp(torch.sigmoid(pred_loc), eps, 1 - eps)
    pos_inds = gt_loc.eq(1).float()
    neg_inds = gt_loc.lt(1).float()
    neg_weights = torch.pow(1 - gt_loc, 4)
    loss_pos = torch.log(pred_loc) * torch.pow(1 - pred_loc, 2) * pos_inds
    loss_neg = torch.log(1 - pred_loc) * torch.pow(pred_loc, 2) * neg_weights * neg_inds
    loss_loc = -(loss_pos.sum() + loss_neg.sum()) / (pos_inds.sum() + 1)

    # regression loss
    mask = pos_inds.expand_as(pred_theta)
    loss_reg = torch.tensor(0.0, device=device)
    if mask.sum() > 0:
        loss_reg = (
                F.smooth_l1_loss(pred_theta[mask>0], gt_theta[mask>0]) +
                F.smooth_l1_loss(pred_width[mask>0], gt_width[mask>0]) +
                F.smooth_l1_loss(pred_depth[mask>0], gt_depth[mask>0])
        )
    return loss_loc + loss_reg


def train_model(model, loader, optimizer, device, epochs=1, save_path=None, max_batches=None, print_every=10, generation=None, log_csv=None):
    """
    main training loop that both handles fast NAS proxy training and full training.

    :param model: pytorch model.
    :param loader: dataloader.
    :param optimizer: optimizer.
    :param device: CPU/GPU.
    :param epochs: number of epochs.
    :param save_path: (optional) path to save the trained model.
    :param max_batches: (optional) maximum number of batches.
    :param print_every: print settings.
    :param generation: (optional) current generation number for logging.
    :param log_csv: (optional) path to csv file for logging loss.
    :return: best loss.
    """

    model.train()
    best_loss = float('inf')

    # console logging
    mode = "training" if save_path else "proxy search"
    limit_str = f"{max_batches} batches" if max_batches else "full dataset"
    if save_path:
        print(f"\n{mode}: {epochs} epochs x [{limit_str}]")

    # -- epoch loop
    for epoch in range(epochs):
        if epochs > 1: print(f"--- epoch {epoch+1}/{epochs} ---")

        total_loss = 0
        count = 0
        iterator = iter(loader)
        limit = max_batches if max_batches else len(loader)

        for i in range(limit):
            try:
                x, targets = next(iterator)
            except StopIteration:
                break

            x = x.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            preds = model(x)
            loss = hggd_loss(preds, targets, device)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            count += 1

            if print_every and (i + 1) % print_every == 0:
                print(f"      batch {i+1}/{limit} | loss: {loss.item():.4f}")

        # --- finished epoch
        avg_loss = total_loss / count if count > 0 else 999.0

        # csv logging
        if log_csv and generation is not None:
            file_exists = os.path.isfile(log_csv)
            with open(log_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['generation', 'epoch', 'avg_loss'])
                writer.writerow([generation, epoch + 1, avg_loss])

        if epochs > 1:
            print(f"avg loss: {avg_loss:.4f}")

        # checks for best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"saved best model to {save_path}")

    return best_loss