#!/usr/bin/env python3
"""
YOLO Training Script for DroneVision Tank Detection

This script trains YOLO models (v9e, v10x, v11x, v12x) on the tank detection dataset.

Usage:
    python train_yolo.py --model yolov12x --epochs 1000 --batch 8
"""

import os
import argparse
from ultralytics import YOLO


# Model mapping
MODEL_MAP = {
    'yolov9e': 'yolov9e.pt',
    'yolov10x': 'yolov10x.pt',
    'yolov11x': 'yolo11x.pt',
    'yolov12x': 'yolo12x.pt',
}


def train_yolo(
    model_name: str,
    yaml_path: str,
    output_path: str,
    epochs: int = 1000,
    batch_size: int = 8,
    lr: float = 1e-5,
    patience: int = 100,
    device: str = "0",
    imgsz: int = 640,
):
    """
    Train a YOLO model on the tank detection dataset.
    
    Args:
        model_name: Name of the YOLO variant (yolov9e, yolov10x, yolov11x, yolov12x)
        yaml_path: Path to dataset.yaml file
        output_path: Path to save training outputs
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        lr: Initial learning rate
        patience: Early stopping patience
        device: GPU device ID
        imgsz: Input image size
    """
    # Get model weights path
    if model_name in MODEL_MAP:
        model_weights = MODEL_MAP[model_name]
    else:
        model_weights = f"{model_name}.pt"
    
    print(f"🚀 Training {model_name}")
    print(f"   Weights: {model_weights}")
    print(f"   Dataset: {yaml_path}")
    print(f"   Output: {output_path}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    
    # Initialize model
    model = YOLO(model_weights)
    
    # Print model info
    model.info()
    
    # Train
    results = model.train(
        data=yaml_path,
        project=output_path,
        name=model_name,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        optimizer="AdamW",
        patience=patience,
        save_period=1000,  # Save checkpoint every N epochs (disabled effectively)
        device=device,
        imgsz=imgsz,
        plots=True,
        val=True,
    )
    
    print(f"\n✅ Training completed for {model_name}")
    print(f"   Best weights saved at: {output_path}/{model_name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO for tank detection')
    parser.add_argument('--model', type=str, default='yolov12x',
                        choices=['yolov9e', 'yolov10x', 'yolov11x', 'yolov12x'],
                        help='YOLO model variant')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='Path to dataset.yaml file')
    parser.add_argument('--output_path', type=str, default='./runs/yolo',
                        help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    
    args = parser.parse_args()
    
    train_yolo(
        model_name=args.model,
        yaml_path=args.yaml_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
