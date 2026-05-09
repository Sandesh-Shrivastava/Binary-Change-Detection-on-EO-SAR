import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from .change_detection_dataset import GalaxEyeDataset

def get_transforms(img_size=512, mode='train'):
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.15), std=(0.229, 0.224, 0.225, 0.15)), # Adjusted for SAR
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.15), std=(0.229, 0.224, 0.225, 0.15)),
            ToTensorV2()
        ])

class GalaxEyeDataModule:
    def __init__(self, train_dir, val_dir, test_dir, batch_size=8, img_size=512, val_scenes=None):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_scenes = val_scenes # e.g. ['scene_07']
        
    def setup(self):
        # Create full datasets
        train_ds_full = GalaxEyeDataset(self.train_dir, transform=get_transforms(self.img_size, 'train'), remap_labels=True)
        val_ds_full = GalaxEyeDataset(self.val_dir, transform=get_transforms(self.img_size, 'val'), remap_labels=True)
        test_ds_full = GalaxEyeDataset(self.test_dir, transform=get_transforms(self.img_size, 'val'), remap_labels=True)
        
        if self.val_scenes:
            # Perform scene-aware split from the TRAIN directory
            # (since TRAIN is a superset and has more samples)
            train_indices = []
            val_indices = []
            
            for idx, file_name in enumerate(train_ds_full.file_names):
                scene_id = "_".join(file_name.split("_")[:2])
                if scene_id in self.val_scenes:
                    val_indices.append(idx)
                else:
                    train_indices.append(idx)
            
            self.train_dataset = Subset(train_ds_full, train_indices)
            # Use the same train_ds_full but with val_transforms for validation subset
            val_ds_split = GalaxEyeDataset(self.train_dir, transform=get_transforms(self.img_size, 'val'), remap_labels=True)
            self.val_dataset = Subset(val_ds_split, val_indices)
        else:
            # Default: use provided folders
            self.train_dataset = train_ds_full
            self.val_dataset = val_ds_full
            
        self.test_dataset = test_ds_full
        
    def get_loaders(self):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        return train_loader, val_loader, test_loader
