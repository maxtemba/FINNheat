import torch
import torch.nn.functional as F
import os

# ==============================================================================
# 1. LOSS FUNCTION
# ==============================================================================
def hggd_loss(preds, targets, device):
    pred_loc, pred_cls, pred_theta, pred_width, pred_depth = preds
    gt_loc, gt_cls, gt_theta, gt_width, gt_depth = targets
    eps = 1e-6

    # Loc Loss
    pred_loc = torch.clamp(torch.sigmoid(pred_loc), eps, 1 - eps)
    pos_inds = gt_loc.eq(1).float()
    neg_inds = gt_loc.lt(1).float()
    neg_weights = torch.pow(1 - gt_loc, 4)
    loss_pos = torch.log(pred_loc) * torch.pow(1 - pred_loc, 2) * pos_inds
    loss_neg = torch.log(1 - pred_loc) * torch.pow(pred_loc, 2) * neg_weights * neg_inds
    loss_loc = -(loss_pos.sum() + loss_neg.sum()) / (pos_inds.sum() + 1)

    # Reg Loss
    mask = pos_inds.expand_as(pred_theta)
    loss_reg = torch.tensor(0.0, device=device)
    if mask.sum() > 0:
        loss_reg = (
                F.smooth_l1_loss(pred_theta[mask>0], gt_theta[mask>0]) +
                F.smooth_l1_loss(pred_width[mask>0], gt_width[mask>0]) +
                F.smooth_l1_loss(pred_depth[mask>0], gt_depth[mask>0])
        )
    return loss_loc + loss_reg

# ==============================================================================
# 2. THE UNIVERSAL TRAINING LOOP
# ==============================================================================
def train_model(model, loader, optimizer, device, epochs=1, save_path=None, max_batches=None, print_every=10):
    """
    Handles both 'NAS Search' (fast, no save) and 'Full Training' (slow, save best).
    """
    model.train()
    best_loss = float('inf')

    # Logging info
    mode = "Training" if save_path else "Proxy Search"
    limit_str = f"{max_batches} batches" if max_batches else "Full Dataset"
    if save_path: # Only print this header for full training to keep NAS log clean
        print(f"\n🚀 {mode}: {epochs} Epochs x [{limit_str}]")

    for epoch in range(epochs):
        if epochs > 1: print(f"--- Epoch {epoch+1}/{epochs} ---")

        # --- INNER BATCH LOOP ---
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

            # Optional: Clip gradients to prevent real explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            count += 1

            if print_every and (i + 1) % print_every == 0:
                print(f"      Batch {i+1}/{limit} | Loss: {loss.item():.4f}")

        # --- EPOCH FINISHED ---
        avg_loss = total_loss / count if count > 0 else 999.0

        if epochs > 1:
            print(f"   📉 Avg Loss: {avg_loss:.4f}")

        # --- FIX: ALWAYS UPDATE BEST_LOSS, BUT ONLY SAVE IF PATH EXISTS ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"   💾 Saved Best Model to {save_path}")

    return best_loss