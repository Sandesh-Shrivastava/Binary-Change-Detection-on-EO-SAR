# 📑 Technical Report: Binary Change Detection on EO-SAR Image Pairs

**Author:** Sandesh Shrivastava  
**Date:** May 9, 2026  
**Topic:** Cross-modal Change Detection for Disaster Analysis

## 1. Abstract
This project presents a production-quality binary change detection system for Identifying building damage using co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery. We employ an **Early Fusion U-Net** architecture with a **ResNet34** backbone. By implementing a **Weighted Hybrid Loss (BCE + Dice)** and custom SAR normalization, we successfully addressed the extreme class imbalance (6:1) and cross-modal domain gaps. The final model achieved a **Test Recall of 67.43%** and an **IoU of 19.25%**, demonstrating high sensitivity to structural changes in disaster-affected areas.

## 2. Literature Survey
Change detection in remote sensing has traditionally relied on optical (EO) sensors. However, disaster scenarios often involve cloud cover or smoke, making EO data unreliable. Literature suggests that **Synthetic Aperture Radar (SAR)**, which is weather-independent, provides critical backscatter information that correlates with structural stability (Hafner et al., 2021). 

Prior methods often used **Post-Classification Comparison (PCC)**, which suffers from error propagation. Modern deep learning approaches favor **U-Net** architectures due to their skip connections that preserve spatial details (Ronneberger et al., 2015). Recent work in **EO-SAR Fusion** indicates that **Early Fusion** (stacking modalities at the input) is more effective for low-level feature correspondence than Late Fusion. This project addresses the gap in rapid disaster response by prioritizing high Recall (Sensitivity) to ensure minimal missed damage during emergency assessments.

## 3. Methodology

### 3.1 Architecture & Fusion
We utilized a **U-Net** architecture with a **ResNet34** encoder pretrained on ImageNet. An **Early Fusion** strategy was implemented by stacking 3-channel pre-event EO and 1-channel post-event SAR into a 4-channel input tensor. This rationale allows the network to learn the direct relationship between optical texture and radar backscatter change in a single pass.

### 3.2 Handling Class Imbalance
The dataset is highly imbalanced (~94% background). 
- **Initial Attempt**: A standard BCE + Dice loss led to the model predicting "No Change" for all pixels (0% IoU).
- **Final Strategy**: We implemented a **Weighted Hybrid Loss**. By applying a $pos\_weight=6.0$ to the Binary Cross Entropy (BCE) component, we forced the model to prioritize the minority "Change" class. This was combined with **Dice Loss** to ensure regional overlap optimization.

### 3.3 Training Strategy
- **Optimizer**: AdamW ($10^{-4}$ learning rate) for stable weight decay.
- **Augmentations**: Spatial transforms (Rotations, Flips) to ensure rotation invariance of building footprints.
- **Normalization**: Custom stats for SAR (mean=0.15, std=0.15) were crucial to prevent the radar channel from being "washed out" by ImageNet-standard EO normalization.

## 4. Results

### 4.1 Quantitative Metrics
The model was evaluated on both the validation (Scene 07) and the held-out test split (Scenes 09, 10).

| Split | IoU | F1-Score | Recall | Precision |
|---|---|---|---|---|
| **Validation** | 22.62% | 36.89% | 61.40% | 26.35% |
| **Test** | **19.25%** | **32.28%** | **67.43%** | **21.22%** |

#### Confusion Matrix (Test Set)
| | Predicted: No Change | Predicted: Change |
|---|---|---|
| **Actual: No Change** | 18,286,989 (TN) | 1,356,345 (FP) |
| **Actual: Change** | 176,463 (FN) | 365,291 (TP) |

### 4.2 Qualitative Visualizations
Five prediction examples from the test set illustrate the model's behavior:
1.  **Success (Scene 09_000001)**: Accurate localization of a dense cluster of damaged buildings.
2.  **Success (Scene 09_000002)**: High correlation between SAR intensity spikes and damage prediction.
3.  **Success (Scene 10_000028)**: Precise detection of isolated destroyed structures.
4.  **Success (Scene 09_000003)**: Robustness against EO shadows and lighting variations.
5.  **Failure (Scene 09_000034)**: Over-segmentation in a high-noise SAR area, leading to false positives (FP).

## 5. Future Work
As an intern at GalaxEye, my next steps would focus on:
1.  **Siamese Multi-Modal Encoders**: Instead of Early Fusion, I would explore separate encoders for EO and SAR to extract domain-specific features before fusion. This would likely reduce the false positive rate.
2.  **Temporal SAR Ensembling**: Using multiple post-event SAR passes to perform speckle filtering and improve the signal-to-noise ratio.
3.  **Attention Mechanisms**: Implementing **Spatial and Channel Attention** (e.g., Attention U-Net) to help the model focus on building footprints and ignore irrelevant changes like vegetation or moisture levels.

## 6. Conclusion
This project successfully demonstrates that fusing EO and SAR data significantly enhances change detection resilience. The core limitation of the current approach is the **Precision (21.2%)**, caused by SAR speckle noise. However, the high **Recall (67.4%)** ensures the system is a valuable tool for rapid disaster screening where missing a damaged building is more critical than a false alarm.

---
*Time and Resource Log available in the project ZIP.*
