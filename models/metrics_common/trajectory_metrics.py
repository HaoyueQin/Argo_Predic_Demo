import torch
import numpy as np

def compute_ade(pred, target, mask=None):
    """
    Compute Average Displacement Error
    Args:
        pred: [B, T, 2]
        target: [B, T, 2]
        mask: [B, T]
    Returns:
        ade: float
    """
    diff = pred - target
    dist = torch.norm(diff, p=2, dim=-1)
    
    if mask is not None:
        dist = dist * mask
        return (dist.sum() / (mask.sum() + 1e-6)).item()
    
    return dist.mean().item()

def compute_fde(pred, target, mask=None):
    """
    Compute Final Displacement Error
    Args:
        pred: [B, T, 2]
        target: [B, T, 2]
        mask: [B, T]
    Returns:
        fde: float
    """
    diff = pred[:, -1] - target[:, -1]
    dist = torch.norm(diff, p=2, dim=-1)
    
    if mask is not None:
        if mask.dim() == 2:
            mask = mask[:, -1]
        dist = dist * mask
        return (dist.sum() / (mask.sum() + 1e-6)).item()
        
    return dist.mean().item()

def compute_min_ade(preds, target, mask=None):
    """
    Compute Minimum Average Displacement Error (for multi-modal predictions),
    matching the official Argoverse 1 metric: minADE is the ADE of the
    trajectory that has the minimum FDE (see
    argoverse.evaluation.eval_forecasting.get_displacement_errors_and_miss_rate).
    Args:
        preds: [B, K, T, 2]
        target: [B, T, 2]
        mask: [B, T]
    Returns:
        min_ade: float
    """
    # preds: [B, K, T, 2]
    # target: [B, 1, T, 2]
    target_expanded = target.unsqueeze(1)

    diff = preds - target_expanded
    dist = torch.norm(diff, p=2, dim=-1)  # [B, K, T]

    if mask is not None:
        mask_expanded = mask.unsqueeze(1)  # [B, 1, T]
        dist = dist * mask_expanded
        ade_per_mode = dist.sum(dim=-1) / (mask_expanded.sum(dim=-1) + 1e-6)  # [B, K]
    else:
        ade_per_mode = dist.mean(dim=-1)  # [B, K]

    fde_per_mode = dist[..., -1]  # [B, K]
    # Official caliber: both minADE and minFDE correspond to the trajectory
    # with the minimum FDE (not the trajectory with the minimum ADE).
    min_idx = torch.argmin(fde_per_mode, dim=1)  # [B]
    min_ade = ade_per_mode[torch.arange(ade_per_mode.shape[0]), min_idx]  # [B]
    return min_ade.mean().item()

def compute_min_fde(preds, target, mask=None):
    """
    Compute Minimum Final Displacement Error
    Args:
        preds: [B, K, T, 2]
        target: [B, T, 2]
        mask: [B, T]
    Returns:
        min_fde: float
    """
    # Take last time step
    preds_last = preds[:, :, -1, :] # [B, K, 2]
    target_last = target[:, -1, :] # [B, 2]
    target_last_expanded = target_last.unsqueeze(1) # [B, 1, 2]
    
    diff = preds_last - target_last_expanded
    dist = torch.norm(diff, p=2, dim=-1) # [B, K]
    
    min_fde, _ = torch.min(dist, dim=1) # [B]
    
    if mask is not None:
        if mask.dim() == 2:
            mask_last = mask[:, -1] # [B]
        else:
            mask_last = mask
        min_fde = min_fde * mask_last
        return (min_fde.sum() / (mask_last.sum() + 1e-6)).item()
        
    return min_fde.mean().item()

def compute_miss_rate(preds, target, mask=None, threshold=2.0):
    """
    Compute Miss Rate (ratio of FDE > threshold)
    Usually based on best mode for multi-modal
    Args:
        preds: [B, K, T, 2]
        target: [B, T, 2]
        mask: [B, T]
        threshold: float (meters)
    Returns:
        miss_rate: float
    """
    preds_last = preds[:, :, -1, :] # [B, K, 2]
    target_last = target[:, -1, :] # [B, 2]
    target_last_expanded = target_last.unsqueeze(1)
    
    diff = preds_last - target_last_expanded
    dist = torch.norm(diff, p=2, dim=-1) # [B, K]
    
    min_dist, _ = torch.min(dist, dim=1) # [B]
    
    miss = (min_dist > threshold).float()
    
    if mask is not None:
        if mask.dim() == 2:
            mask_last = mask[:, -1]
        else:
            mask_last = mask
        miss = miss * mask_last
        return (miss.sum() / (mask_last.sum() + 1e-6)).item()
        
    return miss.mean().item()
