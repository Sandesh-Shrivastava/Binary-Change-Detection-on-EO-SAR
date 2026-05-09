import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image

class GalaxEyeDataset(Dataset):
    """
    Custom Dataset for GalaxEye EO-SAR Change Detection.
    Expects directory structure:
    root/
    ├── pre-event/ (EO - 3 channels)
    ├── post-event/ (SAR - 1 channel)
    └── target/ (Mask - 1 channel)
    """
    def __init__(self, root_dir, transform=None, remap_labels=False):
        self.root_dir = root_dir
        self.transform = transform
        self.remap_labels = remap_labels
        
        # Get all file paths
        self.pre_event_dir = os.path.join(root_dir, 'pre-event')
        self.post_event_dir = os.path.join(root_dir, 'post-event')
        self.target_dir = os.path.join(root_dir, 'target')
        
        self.file_names = sorted([f for f in os.listdir(self.pre_event_dir) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # Paths
        pre_path = os.path.join(self.pre_event_dir, file_name)
        post_path = os.path.join(self.post_event_dir, file_name)
        target_path = os.path.join(self.target_dir, file_name)
        
        # Load images
        # Pre-event (EO): (H, W, 3)
        with rasterio.open(pre_path) as src:
            pre_img = src.read().transpose(1, 2, 0) # (H, W, C)
            
        # Post-event (SAR): (H, W)
        with rasterio.open(post_path) as src:
            post_img = src.read(1) # (H, W)
            
        # Target (Mask): (H, W)
        with rasterio.open(target_path) as src:
            target_img = src.read(1)
            
        # Ensure correct shapes and dtypes
        if post_img.ndim == 2:
            post_img = np.expand_dims(post_img, axis=-1)
            
        # Stack EO and SAR: (H, W, 4)
        input_img = np.concatenate([pre_img, post_img], axis=-1)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=input_img, mask=target_img)
            input_img = augmented['image']
            target_img = augmented['mask']
        else:
            # Basic to tensor if no transform
            input_img = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
            target_img = torch.from_numpy(target_img).long()
            
        # Smart label remapping
        if self.remap_labels:
            max_val = target_img.max()
            if max_val > 1:
                # 0: Background, 1: Intact -> 0 (No Change)
                # 2: Damaged, 3: Destroyed -> 1 (Change)
                binary_mask = torch.zeros_like(target_img)
                binary_mask[target_img >= 2] = 1
                target_img = binary_mask
            else:
                # Already binary {0, 1}
                target_img = (target_img > 0).long()
        else:
            # Basic binarization: anything > 0 is change
            target_img = (target_img > 0).long()
            
        return input_img, target_img, file_name
