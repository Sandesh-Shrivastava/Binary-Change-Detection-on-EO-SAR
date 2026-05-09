import numpy as np
import matplotlib.pyplot as plt

def visualize_prediction(idx, inputs, targets, outputs, save_path):
    """
    Standard visualization for EO-SAR change detection.
    inputs: (C, H, W) -> EO (0:3), SAR (3)
    targets: (H, W)
    outputs: (H, W) - binary
    """
    # Unnormalize EO (ImageNet stats)
    eo = inputs[:3].permute(1, 2, 0).cpu().numpy()
    eo = (eo * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    eo = np.clip(eo, 0, 1)
    
    # SAR (Custom SAR stats: mean=0.15, std=0.15)
    sar = inputs[3].cpu().numpy()
    sar = (sar * 0.15 + 0.15)
    sar = np.clip(sar, 0, 1)
    
    # GT and Pred
    gt = targets.cpu().numpy()
    pred = outputs.cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(eo)
    axes[0].set_title('Pre-event EO')
    
    axes[1].imshow(sar, cmap='gray')
    axes[1].set_title('Post-event SAR')
    
    # Use fixed vmin/vmax for binary masks so 0 is always blue and 1 is always red in jet
    axes[2].imshow(gt, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth (Mask)')
    
    axes[3].imshow(pred, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('Prediction')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
