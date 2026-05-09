import segmentation_models_pytorch as smp
import torch.nn as nn

def create_model(architecture='unet', encoder='resnet34', in_channels=4, num_classes=1):
    """
    Factory function to create models using SMP.
    """
    if architecture.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None # Use logits for stability with BCE
        )
    elif architecture.lower() == 'attention_unet':
        # SMP doesn't have Attention Unet by default, but we can use UnetPlusPlus or implement a custom one.
        # For simplicity and standard practices, UnetPlusPlus is often a better alternative.
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
        
    return model
