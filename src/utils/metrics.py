import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(preds, targets):
    """
    preds: (B, H, W) - binary
    targets: (B, H, W) - binary
    """
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        
    def update(self, preds, targets):
        # Move to CPU and store
        self.all_preds.append(preds.detach().cpu())
        self.all_targets.append(targets.detach().cpu())
        
    def compute(self):
        preds = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)
        return calculate_metrics(preds, targets)
