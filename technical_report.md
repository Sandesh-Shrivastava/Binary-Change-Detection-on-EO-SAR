# 📑 Technical Report: Binary Change Detection on EO-SAR Image Pairs

**Author:** Sandesh Shrivastava
**Date:** May 9, 2026
**Topic:** Cross-modal Change Detection for Disaster Analysis

## 1. Abstract
This report details the implementation of a robust, production-quality binary change detection system designed to identify building damage from co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery. By employing an early-fusion U-Net architecture and a hybrid loss function, the system effectively addresses the challenges of cross-modal feature correspondence and extreme class imbalance.

## 2. Introduction
Change detection in remote sensing is critical for disaster response. While EO imagery provides high-resolution semantic context, it is limited by weather and lighting. SAR imagery, being an active sensor, can penetrate clouds and smoke but is characterized by speckle noise and different backscatter physics. Fusing these modalities allows for more resilient damage assessment.

## 3. Methodology

### 3.1 Data Fusion Strategy
An **Early Fusion** approach was adopted. Pre-event EO (RGB) and post-event SAR (Amplitude) were stacked into a 4-channel input tensor. This allows the encoder to learn low-level cross-modal features (e.g., how the optical texture of a building roof relates to its radar backscatter signature).

### 3.2 Model Architecture
The core model is a **U-Net** with a **ResNet34** encoder pretrained on ImageNet. 
- **Encoder**: Extracts hierarchical features from the 4-channel input.
- **Decoder**: Up-samples features to the original 1024x1024 resolution.
- **Skip Connections**: Preserve high-frequency spatial details necessary for precise building boundary detection.

### 3.3 Loss Function
To handle the class imbalance (approx. 6:1 ratio of no-change to change), a **Hybrid Weighted Loss** was used:
- **BCE (Binary Cross Entropy)**: Weighted with $pos\_weight=6.0$ to prioritize the sparse "Change" class.
- **Dice Loss**: Directly optimizes the intersection-over-union, making the model robust to region overlap issues.

### 3.4 Experimental Setup
- **Optimizer**: AdamW with a learning rate of $10^{-4}$.
- **Batch Size**: 8 samples per batch.
- **Augmentations**: Horizontal/Vertical flips, Random Rotate 90, and Random Brightness/Contrast to improve generalization.
- **Hardware**: Metal Performance Shaders (MPS) on Apple Silicon.

## 4. Literature Survey
The use of U-Net architectures has become a standard in remote sensing change detection due to their ability to capture multi-scale features through an encoder-decoder structure with skip connections. Research in EO-SAR fusion (e.g., *Hafner et al.*) highlights that early fusion of optical and radar modalities allows the network to learn complex backscatter-reflectance relationships that are vital for disaster assessment where cloud cover often obscures optical sensors.

## 5. Results & Discussion

### 5.1 Quantitative Metrics
The optimized model (Weighted Hybrid Loss + SAR-tuned normalization) achieved a significant performance breakthrough on the held-out test set compared to the initial baseline.

| Metric | Baseline (Untrained/Imbalanced) | **Final Optimized Model** |
|---|---|---|
| **Test IoU** | 0.0007 | **0.1925** |
| **Test F1-Score** | 0.0014 | **0.3228** |
| **Recall (Sensitivity)**| 0.0000 | **0.6743** |
| **Precision** | 0.0000 | **0.2122** |

### 5.2 Qualitative Analysis
As seen in the `outputs/visualizations` directory, the model successfully localizes building footprints and identifies structural changes. The high **Recall (67.4%)** indicates that the system is highly effective at finding damage, which is the primary requirement for rapid disaster response.

## 6. Future Work
1. **Siamese Encoders**: Processing EO and SAR through separate backbones before fusion.
2. **Speckle Filtering**: Preprocessing SAR data to reduce noise.
3. **Temporal Ensembling**: Using multiple post-event SAR passes to improve confidence.

## 7. Conclusion
This project successfully demonstrates the feasibility of cross-modal change detection for building damage assessment. By combining EO and SAR data, we achieve a more resilient monitoring system that is less dependent on favorable atmospheric conditions.

---
*Developed as part of the GalaxEye AI Research Internship Technical Assignment.*
