# Model Checkpoints

This directory should contain the trained model checkpoints.

## Structure

```
models/
├── yolov9e/
│   └── best.pt
├── yolov10x/
│   └── best.pt
├── yolov11x/
│   └── best.pt
├── yolov12x/
│   └── best.pt
└── rfdetr_large/
    └── checkpoint_best_ema.pth
```

## Download

The pretrained checkpoints can be downloaded from:

- **Google Drive**: [Coming soon]
- **Hugging Face**: [Coming soon]

## Training Your Own Models

If you want to train the models yourself:

### YOLO Models

```bash
# Train YOLOv9e
python scripts/train_yolo.py --model yolov9e --yaml_path configs/dataset.yaml --epochs 1000

# Train YOLOv10x
python scripts/train_yolo.py --model yolov10x --yaml_path configs/dataset.yaml --epochs 1000

# Train YOLOv11x
python scripts/train_yolo.py --model yolov11x --yaml_path configs/dataset.yaml --epochs 1000

# Train YOLOv12x
python scripts/train_yolo.py --model yolov12x --yaml_path configs/dataset.yaml --epochs 1000
```

### RF-DETR

```bash
python scripts/train_rfdetr.py --dataset_dir /path/to/coco/dataset --epochs 500 --batch 8
```

## Note on Large Files

Model checkpoints are typically 100MB-500MB each. For GitHub:

1. Use Git LFS (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.pt"
   git lfs track "*.pth"
   git add .gitattributes
   ```

2. Or provide download links instead of committing large files directly.
