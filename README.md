#  Mammography Classification with Optimized Deep_Learning

A modular and production ready deep learning framework for classifying breast abnormalities using mammographic imaging. 

This project leverages tailored CNN architectures and strategic data augmentation to enhance diagnostic accuracy on the MIAS dataset.

---

##  Overview

This repository implements and compares several pretrained deep learning models on the [MIAS](https://www.kaggle.com/datasets/aryashah2k/mias-mammography-dataset) mammography dataset, focusing on improved generalization through:

- Smart data augmentation
- Label balancing with optional inclusion of normal tissue
- Freezing pre-trained convolutional backbones
- Modular training and evaluation pipeline

---

##  Reference

>  **Citation**  
This implementation is inspired by the paper:

**“Optimized Deep Learning for Mammography: Augmentation and Tailored Architectures”**  
Authors: Syed Ibrar, Hussain et al.  
Published in *MDPI Information*, 2025  
📎 [Read the paper](https://www.mdpi.com/2078-2489/16/5/359)

---

## Project Structure

---

## Dataset: MIAS Mammography

- **Name**: Mammographic Image Analysis Society (MIAS) Database
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/aryashah2k/mias-mammography-dataset)
- **Classes**:
  - `B`: Benign
  - `M`: Malignant
  - `N`: Normal (optional)
- **Image Size**: 1024x1024 `.pgm` grayscale images

Preprocessing includes:
- Resizing to `299x299`
- Image normalization verification
- 360° rotation augmentation in 6° steps

---

## Model Architectures

| Model Name     | Architecture           |
|----------------|------------------------|
| `mobilenetv3`  | MobileNetV3 Large      |
| `nasnetmobile` | NASNetMobile           |
| `resnetrs`     | ResNetRS101            |
| `xception`     | Xception               |
| `resnet152`    | ResNet152              |
| `densenet201`  | DenseNet201            |

Each model:
- Uses pretrained ImageNet weights
- Freezes the convolutional base
- Adds 3-layer custom classifier

---

##  Quick Start

### 1. Install Dependencies
bash
pip install -r requirements.txt


Download the data and place it in your main folder repository....:

./
    ├── Info.txt
    └── all-mias-norm/*.pgm


## Evaluation Performance Metrics
Each trained model reports:

Accuracy

Precision (weighted)

Recall (weighted)

F1 Score (weighted)

Cohen Kappa Score

Full classification report



## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

