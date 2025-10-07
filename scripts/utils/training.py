import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def to_device(batch, device: torch.device):
    """Recursively move tensors (and nested dict/list/tuple) to device."""
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    elif torch.is_tensor(batch):
        return batch.to(device)
    return batch

class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0) -> None:
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.delta: float = delta
    
    def __call__(self, val_loss: float, model: nn.Module, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving best checkpoint...")
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss

def kl_gaussian(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    return kl.mean()

def compute_metrics(
    preds: torch.Tensor,        # (b,a,t,2)
    target: torch.Tensor,       # (b,a,t,2)
    mask: torch.Tensor = None  # (b,a,t,1) with 0/1 or None
) -> Dict[str, torch.Tensor]:

    diff = preds - target
    l1 = diff.abs()
    l2 = diff.pow(2)

    if mask is not None:
        # Ensure dtypes/shapes
        mask2 = mask.expand_as(preds).to(preds.dtype)  # (b,a,t,2)

        # Counts for normalization
        valid_coord_count = mask2.sum()                         # counts coords (…×2)
        valid_point_count = mask[..., 0].sum()                  # counts points (no ×2)

        # Guard against empty masks (keep differentiable)
        eps = torch.finfo(preds.dtype).eps
        denom_coords = torch.clamp(valid_coord_count, min=eps)
        denom_points = torch.clamp(valid_point_count, min=eps)

        # Scalar tensor metrics (differentiable)
        mae_t = (l1 * mask2).sum() / denom_coords
        mse_t = (l2 * mask2).sum() / denom_coords
        rmse_t = torch.sqrt(mse_t + eps)

        # ADE: L2 distance per point, averaged over valid points
        disp = torch.norm(diff * mask2, dim=-1)     # (b,a,t)
        ade_t = disp.sum() / denom_points

        # FDE: final step only
        final_mask = mask[:, :, -1, 0]                          # (b,a)
        final_pred = preds[:, :, -1, :]
        final_tgt = target[:, :, -1, :]
        fde_dist = torch.norm(final_pred - final_tgt, dim=-1)  # (b,a)
        denom_final = torch.clamp(final_mask.sum(), min=eps)
        fde_t = (fde_dist * final_mask).sum() / denom_final
    else:
        # No mask: compute metrics over all values
        total_coords = torch.tensor(preds.numel(), dtype=preds.dtype, device=preds.device)
        total_points = torch.tensor(preds.shape[0] * preds.shape[1] * preds.shape[2], 
                                  dtype=preds.dtype, device=preds.device)
        total_final_points = torch.tensor(preds.shape[0] * preds.shape[1], 
                                       dtype=preds.dtype, device=preds.device)
        
        eps = torch.finfo(preds.dtype).eps
        denom_coords = torch.clamp(total_coords, min=eps)
        denom_points = torch.clamp(total_points, min=eps)
        denom_final = torch.clamp(total_final_points, min=eps)

        # Scalar tensor metrics (differentiable)
        mae_t = l1.sum() / denom_coords
        mse_t = l2.sum() / denom_coords
        rmse_t = torch.sqrt(mse_t + eps)

        # ADE: L2 distance per point, averaged over all points
        disp = torch.norm(diff, dim=-1)  # (b,a,t)
        ade_t = disp.sum() / denom_points

        # FDE: final step only
        final_pred = preds[:, :, -1, :]
        final_tgt = target[:, :, -1, :]
        fde_dist = torch.norm(final_pred - final_tgt, dim=-1)  # (b,a)
        fde_t = fde_dist.sum() / denom_final

    metrics = {"mae": mae_t, "mse": mse_t, "rmse": rmse_t, "ade": ade_t, "fde": fde_t}
    
    return metrics

import torch
import torch.nn.functional as F

def compute_occupancy_loss(
    preds: torch.Tensor, 
    raw_grid: torch.Tensor, 
    scene_coord: torch.Tensor
) -> torch.Tensor:
    """
    Computes a loss for predicted points that fall into obstacle regions.

    Args:
        preds (torch.Tensor): Predicted trajectories in world coordinates. 
                              Shape: (batch, n_agents, n_timesteps, 2).
        raw_grid (torch.Tensor): The occupancy grid. Shape: (batch, 1, height, width).
                                 0 = free space, >0 = obstacle.
        scene_coord (torch.Tensor): Maps grid cells to world coordinates. 
                                    Shape: (batch, 2, height, width).

    Returns:
        torch.Tensor: A single scalar loss value.
    """
    b, a, t, _ = preds.shape
    
    # 1. Find the min/max world coordinates to normalize the predictions.
    # These define the boundaries of the grid in the world coordinate system.
    # Using amin/amax for batched min/max operations.
    x_min = scene_coord[:, 0, ...].amin(dim=(1, 2), keepdim=True)
    x_max = scene_coord[:, 0, ...].amax(dim=(1, 2), keepdim=True)
    y_min = scene_coord[:, 1, ...].amin(dim=(1, 2), keepdim=True)
    y_max = scene_coord[:, 1, ...].amax(dim=(1, 2), keepdim=True)

    # 2. Normalize the predicted (x, y) coordinates to be in the range [-1, 1].
    # grid_sample expects coordinates in this range, where (-1,-1) is the top-left corner.
    pred_x, pred_y = preds.split(1, dim=-1) # Split into x and y

    # Unsqueeze to allow broadcasting with preds shape (b, a, t, 1)
    x_min, x_max = x_min.unsqueeze(-1), x_max.unsqueeze(-1)
    y_min, y_max = y_min.unsqueeze(-1), y_max.unsqueeze(-1)

    norm_x = 2 * (pred_x - x_min) / (x_max - x_min) - 1
    norm_y = 2 * (pred_y - y_min) / (y_max - y_min) - 1
    
    # Combine back into a single tensor for grid_sample
    normalized_preds = torch.cat([norm_x, norm_y], dim=-1)
    
    # 3. Reshape the normalized predictions to match the expected input for grid_sample.
    # The 'grid' argument for grid_sample should be (N, H_out, W_out, 2).
    # We can treat all agent and time predictions as a single list of points to check.
    # New shape: (batch_size, num_points, 1, 2)
    grid_sampler = normalized_preds.view(b, a * t, 1, 2)

    # 4. Sample the occupancy grid at the predicted locations.
    # We use 'bilinear' interpolation to get a smooth, differentiable output.
    # 'padding_mode="border"' clamps out-of-bound predictions to the edge value.
    sampled_occupancy_values = F.grid_sample(
        input=raw_grid.float(),       # The grid must be float
        grid=grid_sampler,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )
    
    # The output shape will be (N, C, H_out, W_out) -> (b, 1, a*t, 1)
    # We want the loss to be the average "obstacle value" across all predicted points.
    # Since free space is 0, any positive value contributes to the loss.
    occupancy_loss = torch.mean(sampled_occupancy_values)

    return occupancy_loss

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    metric: str = 'mae',
) -> float:
    model.train()
    running_loss, n_batches = 0.0, 0
    bar = tqdm(dataloader, total=len(dataloader), desc="Train", leave=False)

    for past_seq, future_seq in bar:
        past_seq = to_device(past_seq, device)
        future_seq = to_device(future_seq, device)

        preds, raw_grid, scene_coord = model(past_seq)  # (b,a,t,2)
        # Check if prediction_pos_mask exists in future_seq
        mask = future_seq.get("prediction_pos_mask", None)
        metrics = compute_metrics(
            preds,
            future_seq["prediction_pos"],
            mask,
        )
        track_loss = metrics[metric]
        occ_loss = compute_occupancy_loss(preds, raw_grid, scene_coord)
        loss = track_loss + 0.4*occ_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        n_batches += 1
        bar.set_postfix(loss=running_loss / n_batches)

    return running_loss / max(1, n_batches)

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str,
    metric: str = 'ade',
) -> float:

    model.eval()
    v_loss, n_batches = 0.0, 0
    bar = tqdm(dataloader, total=len(dataloader), desc=description, leave=False)
    
    with torch.no_grad():
        for past_seq, future_seq in bar:
            past_seq = to_device(past_seq, device)
            future_seq = to_device(future_seq, device)

            preds, raw_grid, scene_coord = model(past_seq)
            mask = future_seq.get("prediction_pos_mask", None)
            metrics = compute_metrics(preds, future_seq["prediction_pos"], mask)
            track_loss = metrics[metric]
            occ_loss = compute_occupancy_loss(preds, raw_grid, scene_coord)
            
            loss = track_loss + 0.4*occ_loss

            v_loss += loss
            n_batches += 1
            bar.set_postfix(loss=v_loss / n_batches)

    return v_loss / max(1, n_batches)


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str,
    ckpt_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Lists to store all batches
    all_past_seq = []
    all_future_seq = []
    all_preds = []
    
    bar = tqdm(dataloader, total=len(dataloader), desc=description, leave=False)
    with torch.no_grad():
        for past_seq, future_seq in bar:
            past_seq = to_device(past_seq, device)
            future_seq = to_device(future_seq, device)
            preds,_,_= model(past_seq)
            
            # Store each batch's results
            all_past_seq.append(past_seq)
            all_future_seq.append(future_seq)
            all_preds.append(preds)
    
    return all_past_seq, all_future_seq, all_preds

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    ckpt_dir: str,
    metric: str = 'ade',
) -> Dict[str, List[float]]:

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "test_loss": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, metric=metric)

        # Validate
        val_loss = validate(model, val_loader, device, description="Validation", metric=metric)
        scheduler.step(val_loss)

        # Test after each epoch
        test_loss = validate(model, test_loader, device, description="Test", metric=metric)

        print(f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | test_loss: {test_loss:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        # Save epoch checkpoint
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            },
            os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:03d}.pth"),
        )
        print(f"✅ Saved epoch checkpoint to {ckpt_dir}/model_epoch_{epoch + 1:03d}.pth")

        # Early stopping on val loss
        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()

    return history