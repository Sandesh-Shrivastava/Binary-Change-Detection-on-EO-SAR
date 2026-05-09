import torch
from tqdm import tqdm
import os
from ..utils.metrics import MetricTracker

class Trainer:
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_iou = 0.0
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        tracker = MetricTracker()
        
        # Initialize Scaler for Mixed Precision
        scaler = torch.amp.GradScaler('mps' if self.device.type == 'mps' else 'cuda', enabled=(self.device.type != 'cpu'))
        
        pbar = tqdm(loader, desc='Training')
        for inputs, targets, _ in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Use Autocast for Mixed Precision
            with torch.amp.autocast(device_type=('mps' if self.device.type == 'mps' else 'cuda'), enabled=(self.device.type != 'cpu')):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Scale Loss and Step
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            tracker.update(preds, targets)
            
            pbar.set_postfix({'loss': loss.item()})
            
        metrics = tracker.compute()
        metrics['loss'] = total_loss / len(loader)
        return metrics
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        tracker = MetricTracker()
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                tracker.update(preds, targets)
                
        metrics = tracker.compute()
        metrics['loss'] = total_loss / len(loader)
        
        # Save best model
        if metrics['iou'] > self.best_iou:
            self.best_iou = metrics['iou']
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pth'))
            
        return metrics

    def test(self, loader, weights_path=None):
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model.eval()
        tracker = MetricTracker()
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(loader, desc='Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                tracker.update(preds, targets)
                
        return tracker.compute()
