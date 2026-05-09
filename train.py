import argparse
import torch
import torch.optim as optim
from src.utils.config import load_config
from src.datasets.datamodule import GalaxEyeDataModule
from src.models.model_factory import create_model
from src.losses.losses import get_loss
from src.trainers.trainer import Trainer

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config_path):
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['training'].get('seed', 42))
    
    # Device
    device_type = config['training']['device']
    if device_type == 'mps' and not torch.backends.mps.is_available():
        device_type = 'cpu'
    elif device_type == 'cuda' and not torch.cuda.is_available():
        device_type = 'cpu'
    device = torch.device(device_type)
    print(f"Using device: {device}")
    
    # Data
    dm = GalaxEyeDataModule(
        train_dir=config['data']['train_dir'],
        val_dir=config['data']['val_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['data']['batch_size'],
        img_size=config['data']['img_size'],
        val_scenes=config['data']['val_scenes']
    )
    dm.setup()
    train_loader, val_loader, test_loader = dm.get_loaders()
    
    # Model
    model = create_model(
        architecture=config['model']['architecture'],
        encoder=config['model']['encoder'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Optimizer & Criterion
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    # Calculate pos_weight (imbalance 6:1)
    pos_weight = torch.tensor([6.0]).to(device)
    criterion = get_loss(config['training']['loss_type'], pos_weight=pos_weight)
    
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=config['output']['checkpoint_dir']
    )
    
    # Training Loop
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        
    print("\nTraining Complete. Evaluating on Test Set...")
    test_metrics = trainer.test(test_loader, weights_path=os.path.join(config['output']['checkpoint_dir'], 'best_model.pth'))
    print(f"Test IoU: {test_metrics['iou']:.4f} | Test F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    import os # Needed for weights path
    main(args.config)
