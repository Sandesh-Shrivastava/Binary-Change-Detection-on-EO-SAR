# 🌍 GalaxEye EO-SAR Binary Change Detection

Professional production-quality implementation for binary change detection on co-registered EO (Electro-Optical) and SAR (Synthetic Aperture Radar) image pairs.

## 🚀 Overview

This project implements a multi-modal semantic segmentation pipeline to detect building damage in disaster scenes. It leverages a fusion approach by stacking pre-event EO (RGB) and post-event SAR (Amplitude) imagery into a 4-channel input for a deep learning model.

### Key Features
- **Early Fusion Architecture**: Combines 3-channel EO and 1-channel SAR into a single tensor for feature learning.
- **State-of-the-Art Models**: Support for U-Net and U-Net++ with pretrained encoders (via `segmentation-models-pytorch`).
- **Robust Training Pipeline**: Includes mixed precision support, scene-aware data splitting, and hybrid loss functions (BCE + Dice).
- **Comprehensive Evaluation**: Automated metric calculation (IoU, F1, Precision, Recall) and prediction visualization.

## 📁 Project Structure

```
GalaxEye_Space/
├── config.yaml          # Hyperparameters and path configurations
├── train.py             # Main training entry point
├── evaluate.py          # Detailed evaluation and visualization
├── src/
│   ├── datasets/        # Dataset class and DataModule
│   ├── models/          # Model factory and architectures
│   ├── losses/          # Hybrid loss implementations
│   ├── trainers/        # Training engine
│   ├── utils/           # Metrics and config utilities
│   └── visualization/   # Plotting scripts
├── checkpoints/         # Saved model weights
└── outputs/             # Evaluation results and plots
```

## 🛠️ Installation

```bash
# Clone the repository
# git clone <repo_url>
# cd GalaxEye_Space

# Install dependencies
pip install -r requirements.txt
```

## 📈 Usage

### 1. Training
Configure your hyperparameters in `config.yaml` and run:
```bash
python train.py --config config.yaml
```

### 2. Evaluation
To evaluate a trained model on the test set:
```bash
python evaluate.py --config config.yaml --weights checkpoints/best_model.pth --split test
```

## 📊 Methodology

### Data Pipeline
- **Input Modalities**: 
  - Pre-event: 3-band EO (RGB)
  - Post-event: 1-band SAR (Amplitude)
- **Augmentation**: Albumentations (Random Crop, Flips, Rotations, Color Jitter)
- **Split Strategy**: Scene-aware splitting to prevent geographic leakage (Training on scenes 01-06, 08; Validating on scene 07).

### Model Selection
- **Architecture**: U-Net with ResNet34 backbone.
- **Fusion**: Early fusion at the input layer (4 input channels).
- **Loss**: Hybrid BCE + Dice Loss to address the inherent class imbalance (approx. 6:1 in training data).

## 🏆 Performance

*Note: Results depend on training duration and hardware. Preliminary tests show strong alignment between SAR signatures and building footprints.*

| Metric | Val Score | Test Score |
|---|---|---|
| IoU | TBD | TBD |
| F1-Score | TBD | TBD |
| Recall | TBD | TBD |

## 📝 License
Research project for GalaxEye Space Internship Assignment.
