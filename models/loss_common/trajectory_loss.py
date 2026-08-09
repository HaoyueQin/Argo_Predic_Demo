import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSmoothL1Loss(nn.Module):
    """
    Weighted Smooth L1 Loss for trajectory prediction.
    """
    def __init__(self, beta=1.0, reduction='mean'):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        """
        Args:
            pred: Prediction tensor [B, T, 2]
            target: Target tensor [B, T, 2]
            weights: Optional weights [B, T] or [B]
        """
        loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.beta)
        # Sum over coordinates
        loss = loss.sum(dim=-1)
        
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            loss = loss * weights
            
        if self.reduction == 'mean':
            if weights is not None:
                return loss.sum() / (weights.sum() + 1e-6)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def ade_loss(pred, target, mask=None):
    """
    Average Displacement Error Loss (L2 distance).
    Args:
        pred: [B, T, 2]
        target: [B, T, 2]
        mask: [B, T] or None
    """
    diff = pred - target
    dist = torch.norm(diff, p=2, dim=-1)
    
    if mask is not None:
        dist = dist * mask
        return dist.sum() / (mask.sum() + 1e-6)
    
    return dist.mean()

def fde_loss(pred, target, mask=None):
    """
    Final Displacement Error Loss (L2 distance at last step).
    Args:
        pred: [B, T, 2]
        target: [B, T, 2]
        mask: [B, T] or None - typically mask[:, -1]
    """
    diff = pred[:, -1] - target[:, -1]
    dist = torch.norm(diff, p=2, dim=-1)
    
    if mask is not None:
        # If mask is full sequence [B, T], take last step
        if mask.dim() == 2 and mask.shape[1] == pred.shape[1]:
            mask = mask[:, -1]
        dist = dist * mask
        return dist.sum() / (mask.sum() + 1e-6)
        
    return dist.mean()

