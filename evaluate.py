import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.datasets.datamodule import GalaxEyeDataModule
from src.models.model_factory import create_model
from src.trainers.trainer import Trainer
from src.utils.metrics import calculate_metrics
from src.visualization.visualize import visualize_prediction

def main(config_path, weights_path, split='test'):
    config = load_config(config_path)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Data
    dm = GalaxEyeDataModule(
        train_dir=config['data']['train_dir'],
        val_dir=config['data']['val_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=1, # One by one for visualization
        img_size=config['data']['img_size'],
        val_scenes=config['data']['val_scenes']
    )
    dm.setup()
    _, val_loader, test_loader = dm.get_loaders()
    loader = test_loader if split == 'test' else val_loader
    
    # Model
    model = create_model(
        architecture=config['model']['architecture'],
        encoder=config['model']['encoder'],
        in_channels=config['model']['in_channels']
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Eval
    all_preds = []
    all_targets = []
    
    os.makedirs(os.path.join(config['output']['results_dir'], 'visualizations'), exist_ok=True)
    
    print(f"Evaluating on {split} split...")
    with torch.no_grad():
        for i, (inputs, targets, file_names) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
            # Save first 5 visualizations
            if i < 5:
                save_path = os.path.join(config['output']['results_dir'], 'visualizations', f'{split}_{file_names[0]}.png')
                visualize_prediction(i, inputs[0], targets[0], preds[0], save_path)
                
    # Compute metrics
    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    metrics = calculate_metrics(preds_cat, targets_cat)
    
    print(f"\nResults for {split} split:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k.capitalize()}: {v:.4f}")
        else:
            print(f"{k.upper()}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()
    
    main(args.config, args.weights, args.split)
