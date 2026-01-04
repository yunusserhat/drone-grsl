# Hybrid CNN–Transformer Ensemble for Tank Detection in UAV Imagery

[![Paper](https://img.shields.io/badge/Paper-IEEE%20GRSL-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the official implementation of **"Hybrid CNN–Transformer Ensemble for Tank Detection in UAV Imagery"**.

## 📋 Abstract

Object detection in noisy UAV imagery remains a critical challenge due to motion blur, occlusion, and viewpoint instability. This work proposes a hybrid ensemble approach that combines transformers' global context modeling with CNNs' local feature extraction capabilities. The method uses **Weighted Boxes Fusion (WBF)** to combine predictions from four YOLO variants (YOLOv9e, YOLOv10x, YOLOv11x, YOLOv12x) and a transformer-based detector (RF-DETR).

## 🏆 Results

| Model | Parameters (M) | Public IoU | Private IoU | Inference (ms/img) |
|-------|---------------|------------|-------------|-------------------|
| YOLOv9e | ~58 | 0.9646 | 0.9640 | 27.07 |
| YOLOv10x | ~30 | 0.9652 | 0.9653 | **16.26** |
| YOLOv11x | ~57 | 0.9631 | 0.9631 | 17.96 |
| YOLOv12x | ~59 | 0.9620 | 0.9627 | 21.92 |
| RF-DETR | ~128 | 0.9644 | 0.9659 | 35.01 |
| **Ensemble** | - | **0.9676** | **0.9682** | 118.29 |

## 🗂️ Repository Structure

```
drone-grsl/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── create_dataset.py             # Dataset preprocessing script
│   ├── create_yolo_labels.py         # Convert COCO to YOLO format
│   ├── train_yolo.py                 # YOLO training script
│   ├── train_rfdetr.py               # RF-DETR training script
│   ├── inference_yolo.py             # YOLO inference script
│   ├── inference_rfdetr.py           # RF-DETR inference script
│   └── ensemble_wbf.py               # WBF ensemble inference
├── configs/
│   └── dataset.yaml                  # YOLO dataset configuration
└── models/                           # Pretrained model checkpoints (download separately)
    ├── yolov9e/
    ├── yolov10x/
    ├── yolov11x/
    ├── yolov12x/
    └── rfdetr_large/
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yunusserhat/drone-grsl.git
cd drone-grsl

# Create conda environment
conda create -n drone-grsl python=3.10 -y
conda activate drone-grsl

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- ultralytics >= 8.0
- rfdetr
- ensemble-boxes
- supervision
- opencv-python
- numpy
- pandas

## 📊 Dataset

The dataset is from the [DroneVision: Tank Detection from Aerial Videos](https://www.kaggle.com/competitions/dronevision-tank-detection) challenge on Kaggle.

### Preparing the Dataset

1. Download the dataset from Kaggle
2. Run the preprocessing script:

```bash
python scripts/create_dataset.py --source_dir /path/to/raw/data --target_dir /path/to/processed/data
```

3. Convert to YOLO format:

```bash
python scripts/create_yolo_labels.py --dataset_path /path/to/processed/data
```

## 🚀 Training

### Train YOLO Models

```bash
# Train YOLOv9e
python scripts/train_yolo.py --model yolov9e --epochs 1000 --batch 8

# Train YOLOv10x
python scripts/train_yolo.py --model yolov10x --epochs 1000 --batch 8

# Train YOLOv11x
python scripts/train_yolo.py --model yolov11x --epochs 1000 --batch 8

# Train YOLOv12x
python scripts/train_yolo.py --model yolov12x --epochs 1000 --batch 8
```

### Train RF-DETR

```bash
python scripts/train_rfdetr.py --epochs 500 --batch 8 --lr 1e-5
```

## 🔍 Inference

### Single Model Inference

```bash
# YOLO inference
python scripts/inference_yolo.py --model yolov12x --test_dir /path/to/test/images

# RF-DETR inference
python scripts/inference_rfdetr.py --checkpoint /path/to/checkpoint.pth --test_dir /path/to/test/images
```

### Ensemble Inference (WBF)

```bash
python scripts/ensemble_wbf.py \
    --yolo_models yolov9e yolov10x yolov11x yolov12x \
    --rfdetr_checkpoint /path/to/rfdetr/checkpoint.pth \
    --test_dir /path/to/test/images \
    --output_csv submission.csv
```

## ⚙️ Ensemble Configuration

The ensemble uses Weighted Boxes Fusion with the following weights:

| Model | Weight |
|-------|--------|
| YOLOv9e | 0.15 |
| YOLOv10x | 0.15 |
| YOLOv11x | 0.20 |
| YOLOv12x | 0.25 |
| RF-DETR | 0.25 |

WBF parameters:
- `iou_thr`: 0.6
- `skip_box_thr`: 0.2

## 📈 Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 1000 (early stopping) |
| Batch size | 8 |
| Initial learning rate | 1×10⁻⁵ |
| Optimizer | AdamW |
| Early stopping patience | 100 epochs |
| Data split | 80/10/10 (train/val/test) |
| Resolution (YOLO) | 640×640 |
| Resolution (RF-DETR) | 560×560 |

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{bicakci2026hybrid,
  title={Hybrid CNN--Transformer Ensemble for Tank Detection in UAV Imagery},
  author={B{\i}cak{\c{c}}{\i}, Yunus Serhat},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026},
  volume={23},
  publisher={IEEE}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DroneVision Competition](https://www.kaggle.com/competitions/dronevision-tank-detection) organizers
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementations
- [Roboflow](https://github.com/roboflow/rf-detr) for RF-DETR
- [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) for WBF implementation

## 📧 Contact

Yunus Serhat Bıçakçı - [yunus.serhat@marmara.edu.tr](mailto:yunus.serhat@marmara.edu.tr)

Project Link: [https://github.com/yunusserhat/drone-grsl](https://github.com/yunusserhat/drone-grsl)
