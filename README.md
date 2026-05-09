# 🌍 Binary Change Detection on EO-SAR Image Pairs

## 1. Project Title & Description
**Title**: Cross-modal Change Detection for Disaster Analysis  
**Description**: This project implements a production-quality binary change detection system to identify building damage from co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery. The approach uses an Early Fusion U-Net architecture with a ResNet34 backbone, optimized with a Weighted Hybrid Loss (BCE + Dice) to handle the significant class imbalance inherent in disaster-response datasets.

## 2. Requirements
- **Python Version**: 3.10 or higher
- **Dependencies**: All required libraries are pinned in the `requirements.txt` file. Key dependencies include:
  - `torch`, `torchvision` (Deep Learning Framework)
  - `segmentation-models-pytorch` (Model Architectures)
  - `rasterio` (Geospatial data handling)
  - `albumentations` (Image augmentations)
  - `matplotlib`, `tqdm`, `pyyaml` (Utilities)

## 3. Environment
Follow these steps to set up the execution environment:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Dataset Structure & Training
### Dataset Structure
Ensure your data is organized as follows in the root directory:
```text
Data/
├── train/
│   ├── pre-event/    # EO (.tif)
│   ├── post-event/   # SAR (.tif)
│   └── target/       # Masks (.tif)
├── val/ (same structure)
└── test/ (same structure)
```

### Training Command
To run the training from scratch using the provided configuration:
```bash
python train.py --config config.yaml
```
*The script automatically detects and utilizes MPS (Apple Silicon) or CUDA (NVIDIA) if available.*

## 5. Evaluation
To evaluate the model on the test data by passing the weights and config:
```bash
python evaluate.py --config config.yaml --weights checkpoints/best_model.pth --split test
```

## 6. Model Weights
The final trained checkpoint is publicly hosted on GitHub:
[Download best_model.pth](https://github.com/Sandesh-Shrivastava/Binary-Change-Detection-on-EO-SAR/blob/main/checkpoints/best_model.pth)

## 7. Results
Reported performance on the held-out **Test Set** (Scenes 09 & 10):

| Metric | Validation Set | Test Set |
|---|---|---|
| **IoU** | 22.62% | **19.25%** |
| **F1-Score** | 36.89% | **32.28%** |
| **Recall (Sensitivity)** | 61.40% | **67.43%** |
| **Precision** | 26.35% | **21.22%** |

*Note: The high Recall (67.43%) demonstrates the model's high sensitivity to structural damage.*

## 8. Citation / References
1. **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015).
2. **ResNet**: He et al. "Deep Residual Learning for Image Recognition" (2016).
3. **Segmentation Models**: Yakubovskiy, P. "Segmentation Models PyTorch" (2020).
4. **EO-SAR Fusion**: Hafner et al. "Urban Change Detection from EO-SAR Image Pairs" (2021).

---
**Author**: Sandesh Shrivastava  
**Project**: GalaxEye Space AI Research Internship Technical Assignment
