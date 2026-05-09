import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class HybridLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss with positive weighting.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        # Targets are usually (B, H, W), logits are (B, 1, H, W)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1).float()
        
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def get_loss(loss_type='hybrid', pos_weight=None):
    if loss_type == 'hybrid':
        return HybridLoss(pos_weight=pos_weight)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == 'dice':
        return smp.losses.DiceLoss(mode='binary')
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
