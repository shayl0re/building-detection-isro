# Automatic Building Detection Using Machine Learning

## Project Overview

Automatic building detection from high-resolution satellite and aerial imagery is a
critical task in **urban planning**, **disaster management**, and **geospatial analysis**.
Manual image analysis is time-intensive and error-prone when dealing with the massive
volumes of data produced by modern satellites and drones.

This project applies **deep learning** — specifically a **U-Net** architecture with a
**ResNet-18 encoder** — to automate the segmentation and detection of buildings in
satellite imagery. A dataset of **200 labelled samples** was used, and the model
achieved high pixel-level accuracy despite the modest dataset size, demonstrating the
viability of CNN-based approaches for this task.

---

## Methodology

```
Satellite Image
     │
     ▼
┌─────────────┐    ┌──────────────────────────────────┐
│ Pre-process │    │  U-Net (ResNet-18 Encoder)        │
│ • Resize    │───▶│  • Encoder: Hierarchical features │
│ • Augment   │    │  • Decoder: 4× Upsample blocks    │
│ • Normalise │    │  • Head: Conv2d → Sigmoid         │
└─────────────┘    └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │  Post-processing                  │
                   │  • Probability thresholding (0.5) │
                   │  • Morphological operations       │
                   │  • Connected component analysis   │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                        Binary Building Mask
```

### Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Architecture | U-Net | Encoder-decoder design preserves spatial info |
| Backbone | ResNet-18 (pretrained) | Transfer learning reduces training data needs |
| Loss function | Binary Cross-Entropy | Suitable for pixel-wise binary classification |
| Optimiser | Adam (lr = 1e-4) | Adaptive LR converges faster on small datasets |
| Augmentation | Random crop, flip, brightness | Improves generalisation |
| Image size | 128 × 128 | Memory-efficient for training on modest hardware |

---

## Repository Structure

```
building-detection/
│
├── src/                        ← Python source modules
│   ├── dataset.py                 ← CustomDataset (PyTorch)
│   ├── model.py                   ← U-Net architecture
│   ├── train.py                   ← Training script (CLI)
│   ├── evaluate.py                ← Evaluation + confusion matrix
│   ├── inference.py               ← Single-image prediction
│   └── explore_data.py            ← OpenCV data exploration (Objective 2)
│
├── notebooks/
│   ├── 01_unet_training_pipeline.ipynb   ← End-to-end training (Objective 1)
│   └── 02_data_exploration_opencv.ipynb  ← Dataset visualisation (Objective 2)
│
├── data/
│   ├── images/                    ← Satellite images (.JPG)
│   └── masks/                     ← Segmentation masks (.PNG)
│
├── models/                     ← Saved model weights (.pth)
├── results/                    ← Output plots and predictions
├── docs/                       ← Additional documentation
│
├── requirements.txt
├── .gitignore
└── README.md                      ← You are here
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/building-detection.git
cd building-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Linux / macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Data

```
data/
├── images/
│   ├── triple_0.JPG
│   ├── triple_1.JPG
│   └── ...   (200 images)
└── masks/
    ├── triple_0.PNG
    ├── triple_1.PNG
    └── ...   (200 masks)
```

> **Note:** Each image must have a corresponding mask with the same base filename.

---

## Usage

### Option A — Jupyter Notebooks (Recommended for Beginners)

```bash
jupyter notebook
```

Open **`notebooks/01_unet_training_pipeline.ipynb`** for the full training pipeline,
or **`notebooks/02_data_exploration_opencv.ipynb`** to explore the dataset.

---

### Option B — Command-Line Scripts

#### Explore the Dataset (Objective 2)
```bash
python src/explore_data.py \
    --image_dir data/images \
    --mask_dir  data/masks  \
    --num_pairs 4
```
*Outputs:* `results/sample_pairs.png`

#### Train the Model (Objective 1)
```bash
python src/train.py \
    --image_dir  data/images \
    --mask_dir   data/masks  \
    --epochs     25          \
    --batch_size 16          \
    --lr         1e-4        \
    --img_size   128
```
*Outputs:* `models/unet_building.pth`, `results/training_loss_curve.png`

#### Evaluate on Test Set
```bash
python src/evaluate.py \
    --image_dir  data/images              \
    --mask_dir   data/masks               \
    --model_path models/unet_building.pth
```
*Outputs:* Accuracy, classification report, `results/confusion_matrix.png`

#### Run Inference on a New Image
```bash
python src/inference.py \
    --image_path  data/images/triple_0.JPG   \
    --model_path  models/unet_building.pth   \
    --output_path results/prediction.png
```
*Outputs:* `results/prediction.png` (side-by-side: input image + predicted mask)

---

## Results

| Metric | Value |
|---|---|
| Training epochs | 25 |
| Final training loss | ~0.0706 |
| Dataset size | 200 samples |
| Train / Test split | 90 / 10 |
| Pixel accuracy | High (see confusion matrix) |

### Training Loss Curve
The model converged rapidly — loss dropped from **0.607 → 0.07** within the first 5 epochs
and plateaued around **0.0706**, indicating stable learning.

---

## 🔧 Module Reference

### `src/dataset.py`
```python
from dataset import CustomDataset, get_transform

transform = get_transform(img_size=128)
dataset   = CustomDataset('data/images', 'data/masks', transform=transform)
```

### `src/model.py`
```python
from model import build_model

model = build_model(num_classes=1, device='cpu')  # or 'cuda'
```

### `src/train.py`
```python
from train import train_model, plot_loss
```

### `src/inference.py`
```python
from inference import inference, save_prediction
```

---

## Key Concepts

### U-Net Architecture
U-Net uses an encoder–decoder structure with skip connections that combine
high-resolution spatial details from the encoder with the deeper semantic
features of the decoder, enabling precise pixel-level segmentation.

### Why ResNet-18 as Encoder?
ResNet-18, pretrained on ImageNet, has already learned general feature detectors
(edges, textures, shapes). Fine-tuning it on satellite imagery via transfer learning
significantly reduces the amount of labelled training data required.

### Evaluation Metrics
- **IoU (Intersection over Union):** Measures overlap between predicted and ground-truth masks.
- **Dice Coefficient:** Similar to IoU; more sensitive to small building regions.
- **Precision / Recall:** Evaluates false positives (incorrectly flagged non-buildings) and
  false negatives (missed buildings).

---

## Future Work

- Scale to larger datasets (SpaceNet, Inria Aerial Image Dataset)
- Add skip connections between encoder and decoder (full U-Net)
- Experiment with multi-spectral imagery (NIR, SAR channels)
- Integrate attention mechanisms for better focus on building regions
- Deploy via a GIS platform (QGIS plugin / Bhuvan Geo-portal)
- Explore Generative Adversarial Networks (GANs) for data augmentation

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* arXiv:1505.04597
2. Audebert, N., Le Saux, B., & Lefèvre, S. (2017). *Segment-before-Detect: Vehicle Detection and Classification through Semantic Segmentation of Aerial Images.* Remote Sensing.
3. Mnih, V. (2013). *Machine Learning for Aerial Image Labeling.* PhD thesis, University of Toronto.
4. Zhang, Y., He, G., & Tao, Y. (2019). *Multi-Scale Building Extraction from Remote Sensing Images Based on CNNs.* IEEE Transactions on Geoscience and Remote Sensing.

---

## 📄 Licence

© 2024 Shivani Ram R. A. — Amrita School of Engineering, Bengaluru
