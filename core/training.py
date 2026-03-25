import os
import csv
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

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

    # localization loss (penalty-reduced focal loss, pos threshold >= 0.99 as in HGGD)
    pred_loc = torch.clamp(torch.sigmoid(pred_loc.float().clamp(-20, 20)), eps, 1 - eps)
    pos_inds = gt_loc.ge(0.99).float()
    neg_inds = gt_loc.lt(0.99).float()
    neg_weights = torch.pow(1 - gt_loc, 4)
    loss_pos = torch.log(pred_loc) * torch.pow(1 - pred_loc, 2) * pos_inds
    loss_neg = torch.log(1 - pred_loc) * torch.pow(pred_loc, 2) * neg_weights * neg_inds
    loss_loc = -(loss_pos.sum() + loss_neg.sum()) / (pos_inds.sum() + 1)

    # classification loss (angle anchor focal loss, thres=0.5 alpha=0.25 as in HGGD)
    pred_cls = torch.clamp(torch.sigmoid(pred_cls.float().clamp(-20, 20)), eps, 1 - eps)
    cls_pos = gt_cls.ge(0.5).float()
    cls_neg = gt_cls.lt(0.5).float()
    loss_cls_pos = 0.25 * torch.log(pred_cls) * torch.pow(1 - pred_cls, 2) * cls_pos
    loss_cls_neg = 0.75 * torch.log(1 - pred_cls) * torch.pow(pred_cls, 2) * cls_neg
    loss_cls = -(loss_cls_pos.sum() + loss_cls_neg.sum()) / (cls_pos.sum() + 1)

    # regression loss (masked by cls confidence as in HGGD)
    cls_mask = cls_pos.expand_as(pred_theta)
    loss_reg = torch.tensor(0.0, device=device)
    if cls_mask.sum() > 0:
        loss_reg = (
                F.smooth_l1_loss(pred_theta[cls_mask>0], gt_theta[cls_mask>0]) +
                F.smooth_l1_loss(pred_width[cls_mask>0], gt_width[cls_mask>0]) +
                F.smooth_l1_loss(pred_depth[cls_mask>0], gt_depth[cls_mask>0])
        )
    return loss_loc + loss_cls + 5.0 * loss_reg


def train_model(model, loader, optimizer, device, epochs=1, save_path=None, max_batches=None, print_every=10, generation=None, log_csv=None, val_loader=None, scheduler=None, heatmap_callback=None):
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
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

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

            x = x.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                preds = model(x)
                loss = hggd_loss(preds, targets, device)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1

            if print_every and (i + 1) % print_every == 0:
                print(f"      batch {i+1}/{limit} | loss: {loss.item():.4f}")

        # --- finished epoch
        avg_loss = total_loss / count if count > 0 else 999.0

        if epochs > 1:
            print(f"train loss: {avg_loss:.4f}")

        # --- validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total, val_count = 0, 0
            with torch.no_grad():
                for x_v, targets_v in val_loader:
                    x_v = x_v.to(device, non_blocking=True)
                    targets_v = [t.to(device, non_blocking=True) for t in targets_v]
                    with autocast('cuda', enabled=use_amp):
                        preds_v = model(x_v)
                        loss_v  = hggd_loss(preds_v, targets_v, device)
                    val_total += loss_v.item()
                    val_count += 1
            val_loss = val_total / val_count if val_count > 0 else 999.0
            model.train()
            print(f"val loss:   {val_loss:.4f}")

        # csv logging
        if log_csv:
            file_exists = os.path.isfile(log_csv)
            with open(log_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['generation', 'epoch', 'train_loss', 'val_loss'])
                writer.writerow([generation, epoch + 1, avg_loss, val_loss])

        if scheduler is not None:
            scheduler.step()

        if heatmap_callback is not None and (epoch + 1) % 5 == 0:
            heatmap_callback(model, epoch + 1)

        # save on best monitored loss (val if available, else train)
        monitor = val_loss if val_loss is not None else avg_loss
        if monitor < best_loss:
            best_loss = monitor
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"saved best model to {save_path}")

    return best_loss