# 🌍 Binary Change Detection on EO-SAR Image Pairs

Professional production-quality implementation for binary change detection on co-registered EO (Electro-Optical) and SAR (Synthetic Aperture Radar) image pairs, developed for the GalaxEye Space Technical Assignment.

## 🚀 Overview
This project implements a multi-modal semantic segmentation pipeline to detect building damage in disaster scenes. It leverages a fusion approach by stacking pre-event EO (RGB) and post-event SAR (Amplitude) imagery into a 4-channel input for a deep learning model.

### Key Features
- **Early Fusion Architecture**: Combines 3-channel EO and 1-channel SAR into a single tensor.
- **Optimized for Imbalance**: Uses Weighted Hybrid Loss (BCE + Dice) with a $6.0\times$ positive weight to handle sparse change pixels.
- **Cross-Modal Normalization**: Custom normalization stats for SAR amplitude to ensure robust feature extraction.

## 📁 Dataset Structure
Place the `Data/` folder in the project root. The expected layout is:
```text
Data/
├── train/
│   ├── pre-event/    # EO (.tif)
│   ├── post-event/   # SAR (.tif)
│   └── target/       # Masks (.tif)
├── val/ (same structure)
└── test/ (same structure)
```

## 🛠️ Environment Setup
Requires **Python 3.10+**.

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 📈 Usage

### 1. Training from Scratch
```bash
python train.py --config config.yaml
```
*The training script is optimized for Apple Silicon (MPS) and NVIDIA (CUDA) automatically.*

### 2. Evaluation on Test Data
```bash
python evaluate.py --config config.yaml --weights checkpoints/best_model.pth --split test
```

## 💾 Model Weights
The final trained checkpoint is hosted on the public GitHub repository:
[Download best_model.pth](https://github.com/Sandesh-Shrivastava/Binary-Change-Detection-on-EO-SAR/blob/main/checkpoints/best_model.pth)

## 🏆 Results
Results reported on the held-out **Test Set** (Scene 09 & 10):

| Metric | Score |
|---|---|
| **IoU** | **19.25%** |
| **F1-Score** | **32.28%** |
| **Recall (Sensitivity)** | **67.43%** |
| **Precision** | **21.22%** |

*The high Recall (67.4%) demonstrates the model's effectiveness in rapid disaster damage identification.*

## 📚 Citations & References
1. **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015).
2. **ResNet**: He et al. "Deep Residual Learning for Image Recognition" (2016).
3. **Segmentation Models**: Yakubovskiy, P. "Segmentation Models PyTorch" (2020).
4. **EO-SAR Fusion**: Hafner et al. "Urban Change Detection from EO-SAR Image Pairs" (2021).

---
**Author**: Sandesh Shrivastava  
**Project**: GalaxEye Space AI Research Internship Assignment
