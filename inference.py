import os
import argparse
import torch
import numpy as np
import rasterio
from PIL import Image
from src.models.model_factory import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

def predict(model, eo_path, sar_path, device, img_size=512):
    # Load EO
    with rasterio.open(eo_path) as src:
        eo_img = src.read().transpose(1, 2, 0)
        profile = src.profile # Keep profile for output
        
    # Load SAR
    with rasterio.open(sar_path) as src:
        sar_img = src.read(1)
        
    if sar_img.ndim == 2:
        sar_img = np.expand_dims(sar_img, axis=-1)
        
    # Stack
    input_img = np.concatenate([eo_img, sar_img], axis=-1)
    
    # Transform
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406, 0.15), std=(0.229, 0.224, 0.225, 0.15)),
        ToTensorV2()
    ])
    
    input_tensor = transform(image=input_img)['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred = (torch.sigmoid(logits) > 0.5).long().squeeze().cpu().numpy()
        
    # Resize back to original if needed
    if pred.shape != (profile['height'], profile['width']):
        # Simple resize using PIL
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img = pred_img.resize((profile['width'], profile['height']), Image.NEAREST)
        pred = np.array(pred_img)
        
    return pred, profile

def main(eo_path, sar_path, weights_path, output_path):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Model (Default settings)
    model = create_model(architecture='unet', encoder='resnet34', in_channels=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    pred, profile = predict(model, eo_path, sar_path, device)
    
    # Save as GeoTIFF
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)
    
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eo', type=str, required=True)
    parser.add_argument('--sar', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--output', type=str, default='prediction.tif')
    args = parser.parse_args()
    
    main(args.eo, args.sar, args.weights, args.output)
