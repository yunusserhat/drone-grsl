#!/bin/bash
# Script to copy model checkpoints to the repository structure
# Run this script from the drone-grsl directory

# Source directories
YOLO_DIR="/mnt/home/yb25w/sharedscratch/drone/YOLO"
RFDETR_DIR="/mnt/home/yb25w/sharedscratch/drone/RFDETRLarge_Best_096589"

# Target directory
TARGET_DIR="./models"

# Create model directories
mkdir -p "$TARGET_DIR/yolov9e"
mkdir -p "$TARGET_DIR/yolov10x"
mkdir -p "$TARGET_DIR/yolov11x"
mkdir -p "$TARGET_DIR/yolov12x"
mkdir -p "$TARGET_DIR/rfdetr_large"

echo "📦 Copying YOLO model weights..."

# Copy YOLO weights (best.pt only to save space)
if [ -f "$YOLO_DIR/yolov9e/weights/best.pt" ]; then
    cp "$YOLO_DIR/yolov9e/weights/best.pt" "$TARGET_DIR/yolov9e/"
    echo "  ✓ YOLOv9e copied"
fi

if [ -f "$YOLO_DIR/yolov10x/weights/best.pt" ]; then
    cp "$YOLO_DIR/yolov10x/weights/best.pt" "$TARGET_DIR/yolov10x/"
    echo "  ✓ YOLOv10x copied"
fi

if [ -f "$YOLO_DIR/yolov11x/weights/best.pt" ]; then
    cp "$YOLO_DIR/yolov11x/weights/best.pt" "$TARGET_DIR/yolov11x/"
    echo "  ✓ YOLOv11x copied"
fi

if [ -f "$YOLO_DIR/yolov12x/weights/best.pt" ]; then
    cp "$YOLO_DIR/yolov12x/weights/best.pt" "$TARGET_DIR/yolov12x/"
    echo "  ✓ YOLOv12x copied"
fi

echo ""
echo "📦 Copying RF-DETR checkpoint..."

# Copy RF-DETR checkpoint (best EMA)
if [ -f "$RFDETR_DIR/checkpoint_best_ema.pth" ]; then
    cp "$RFDETR_DIR/checkpoint_best_ema.pth" "$TARGET_DIR/rfdetr_large/"
    echo "  ✓ RF-DETR Large (EMA) copied"
fi

# Also copy the total best if needed
if [ -f "$RFDETR_DIR/checkpoint_best_total.pth" ]; then
    cp "$RFDETR_DIR/checkpoint_best_total.pth" "$TARGET_DIR/rfdetr_large/"
    echo "  ✓ RF-DETR Large (Total) copied"
fi

echo ""
echo "📊 Model sizes:"
du -sh "$TARGET_DIR"/*

echo ""
echo "✅ Model copying completed!"
echo ""
echo "Note: For GitHub, you may want to use Git LFS for large model files:"
echo "  git lfs install"
echo "  git lfs track '*.pt'"
echo "  git lfs track '*.pth'"
