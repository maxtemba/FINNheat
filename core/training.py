import os
import csv
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# loss decomposition returned by hggd_loss and aggregated by _run_epoch / _validate
LossParts = namedtuple('LossParts', ['total', 'loc', 'cls', 'reg'])


def hggd_loss(preds, targets, device):
    """
    calculates error for the Grasp Heatmap Model (GHM).
    implements the loss function from the HGGD paper and properly deals with 99% of background.

    :param preds: output from the model (loc, cls, theta, width, depth).
    :param targets: models ground truth (loc, cls, theta, width, depth).
    :param device: CPU/GPU.
    :return: LossParts(total, loc, cls, reg).
    """

    pred_loc, pred_cls, pred_theta, pred_width, pred_depth = preds
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets
    eps = 1e-6 # log(0) errors

    # localization loss (penalty-reduced focal loss, gamma=4, pos threshold >= 0.7)
    pred_loc = torch.clamp(torch.sigmoid(pred_loc.float().clamp(-20, 20)), eps, 1 - eps)
    pos_inds = gt_loc.ge(0.7).float()
    neg_inds = gt_loc.lt(0.7).float()
    neg_weights = torch.pow(1 - gt_loc, 4)
    loss_pos = torch.log(pred_loc) * torch.pow(1 - pred_loc, 4) * pos_inds
    loss_neg = torch.log(1 - pred_loc) * torch.pow(pred_loc, 4) * neg_weights * neg_inds
    loss_loc = -(loss_pos.sum() + loss_neg.sum()) / (pos_inds.sum() + 1)

    # classification loss (angle anchor focal loss, thres=0.5 alpha=0.25 as in HGGD)
    pred_cls = torch.clamp(torch.sigmoid(pred_cls.float().clamp(-20, 20)), eps, 1 - eps)
    cls_pos = gt_cls.ge(0.5).float()
    cls_neg = gt_cls.lt(0.5).float()
    loss_cls_pos = 0.25 * torch.log(pred_cls) * torch.pow(1 - pred_cls, 2) * cls_pos
    loss_cls_neg = 0.75 * torch.log(1 - pred_cls) * torch.pow(pred_cls, 2) * cls_neg
    loss_cls = -(loss_cls_pos.sum() + loss_cls_neg.sum()) / (cls_pos.sum() + 1)

    # regression loss (masked by cls confidence as in HGGD, clamped + averaged over 3 targets)
    cls_mask = cls_pos.expand_as(pred_theta)
    loss_reg = torch.tensor(0.0, device=device)
    if cls_mask.sum() > 0:
        pred_theta_c = torch.clamp(pred_theta, -0.5, 0.5)
        pred_width_c = torch.clamp(pred_width, -0.5, 0.5)
        pred_depth_c = torch.clamp(pred_depth, -0.5, 0.5)
        n = cls_mask.sum().float() + eps
        loss_reg = (
                (F.smooth_l1_loss(pred_theta_c * cls_mask, gt_theta * cls_mask, reduction='sum') / n +
                 F.smooth_l1_loss(pred_width_c * cls_mask, gt_width * cls_mask, reduction='sum') / n +
                 F.smooth_l1_loss(pred_depth_c * cls_mask, gt_depth * cls_mask, reduction='sum') / n) / 3.0
        )
    total = loss_loc + 0.2 * loss_cls + 0.5 * loss_reg
    return LossParts(total, loss_loc, loss_cls, loss_reg)


def _run_epoch(model, loader, optimizer, scaler, device, max_batches=None, print_every=0):
    """train one epoch; returns mean LossParts (or all-zeros LossParts with total=999 on empty)."""
    model.train()
    use_amp = device.type == 'cuda'
    total = loc = cls = reg = 0.0
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
            parts = hggd_loss(preds, targets, device)

        if not torch.isfinite(parts.total):
            continue

        scaler.scale(parts.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total += parts.total.item()
        loc   += parts.loc.item()
        cls   += parts.cls.item()
        reg   += parts.reg.item()
        count += 1

        if print_every and (i + 1) % print_every == 0:
            print(f"      batch {i+1}/{limit} | loss: {parts.total.item():.4f}")

    if count == 0:
        return LossParts(999.0, 0.0, 0.0, 0.0)
    return LossParts(total/count, loc/count, cls/count, reg/count)


def _validate(model, val_loader, device, use_amp):
    """returns mean LossParts on the validation set."""
    model.eval()
    total = loc = cls = reg = 0.0
    count = 0
    with torch.no_grad():
        for x, targets in val_loader:
            x = x.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]
            with autocast('cuda', enabled=use_amp):
                preds = model(x)
                parts = hggd_loss(preds, targets, device)
            total += parts.total.item()
            loc   += parts.loc.item()
            cls   += parts.cls.item()
            reg   += parts.reg.item()
            count += 1
    if count == 0:
        return LossParts(999.0, 0.0, 0.0, 0.0)
    return LossParts(total/count, loc/count, cls/count, reg/count)


def _log_csv(log_csv, generation, epoch, train_loss, val_loss):
    new_file = not os.path.isfile(log_csv)
    with open(log_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['generation', 'epoch', 'train_loss', 'val_loss'])
        writer.writerow([generation, epoch, train_loss, val_loss])


def train_model(model, loader, optimizer, device, epochs=1, save_path=None,
                max_batches=None, print_every=10, generation=None, log_csv=None,
                val_loader=None, scheduler=None, heatmap_callback=None):
    """
    main training loop that handles both fast NAS proxy training and full training.

    :param model: pytorch model.
    :param loader: dataloader.
    :param optimizer: optimizer.
    :param device: CPU/GPU.
    :param epochs: number of epochs.
    :param save_path: (optional) path to save the trained model.
    :param max_batches: (optional) maximum number of batches per epoch.
    :param print_every: batch-level print frequency (0 disables).
    :param generation: (optional) generation number for csv logging.
    :param log_csv: (optional) path to csv file for per-epoch loss logging.
    :param val_loader: (optional) validation dataloader.
    :param scheduler: (optional) lr scheduler stepped after each epoch.
    :param heatmap_callback: (optional) callable(model, epoch) fired every 5 epochs.
    :return: best monitored loss (val if available, else train).
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)
    best_loss = float('inf')
    full_log = save_path is not None  # full training prints per-component losses

    if full_log:
        limit_str = f"{max_batches} batches" if max_batches else "full dataset"
        print(f"\ntraining: {epochs} epochs x [{limit_str}]")

    for epoch in range(epochs):
        if epochs > 1: print(f"--- epoch {epoch+1}/{epochs} ---")

        train = _run_epoch(model, loader, optimizer, scaler, device,
                           max_batches=max_batches, print_every=print_every)

        if epochs > 1:
            if full_log:
                print(f"train loss: {train.total:.4f}  (loc:{train.loc:.4f}  cls:{train.cls:.4f}  reg:{train.reg:.4f})")
            else:
                print(f"train loss: {train.total:.4f}")

        val = _validate(model, val_loader, device, use_amp) if val_loader is not None else None
        if val is not None:
            if full_log:
                print(f"val loss:   {val.total:.4f}  (loc:{val.loc:.4f}  cls:{val.cls:.4f}  reg:{val.reg:.4f})")
            else:
                print(f"val loss:   {val.total:.4f}")

        if log_csv:
            _log_csv(log_csv, generation, epoch + 1, train.total, val.total if val else None)

        if scheduler is not None:
            scheduler.step()

        if heatmap_callback is not None and (epoch + 1) % 5 == 0:
            heatmap_callback(model, epoch + 1)

        # save on best monitored loss (val if available, else train)
        monitor = val.total if val is not None else train.total
        if monitor < best_loss:
            best_loss = monitor
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"saved best model to {save_path}")

    return best_loss
