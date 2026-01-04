#!/usr/bin/env python3
"""
RF-DETR Training Script for DroneVision Tank Detection

This script trains RF-DETR (Large) model on the tank detection dataset.

Usage:
    python train_rfdetr.py --dataset_dir /path/to/coco/dataset --epochs 500 --batch 8
"""

import os
import json
import argparse
from rfdetr import RFDETRLarge, RFDETRBase, RFDETRMedium, RFDETRSmall, RFDETRNano


# Model mapping
MODEL_MAP = {
    'large': RFDETRLarge,
    'base': RFDETRBase,
    'medium': RFDETRMedium,
    'small': RFDETRSmall,
    'nano': RFDETRNano,
}


def print_dataset_info(dataset_path):
    """Print dataset statistics."""
    splits = ['train', 'valid', 'test']
    for split in splits:
        ann_path = os.path.join(dataset_path, split, '_annotations.coco.json')
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            print(f"  {split}: {len(data['images'])} images, {len(data['annotations'])} annotations")
        else:
            print(f"  {split}: annotations not found")


def train_rfdetr(
    model_variant: str,
    dataset_dir: str,
    output_dir: str,
    epochs: int = 500,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 1e-5,
    early_stopping_patience: int = 60,
    device: str = "cuda",
):
    """
    Train RF-DETR model on the tank detection dataset.
    
    Args:
        model_variant: RF-DETR variant (large, base, medium, small, nano)
        dataset_dir: Path to COCO-format dataset directory
        output_dir: Path to save training outputs
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        grad_accum_steps: Gradient accumulation steps
        lr: Learning rate
        early_stopping_patience: Early stopping patience
        device: Device to use (cuda or cpu)
    """
    print(f"🚀 Training RF-DETR {model_variant.upper()}")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    
    # Print dataset info
    print("\n📊 Dataset statistics:")
    print_dataset_info(dataset_dir)
    
    # Initialize model
    if model_variant not in MODEL_MAP:
        raise ValueError(f"Unknown model variant: {model_variant}. Choose from {list(MODEL_MAP.keys())}")
    
    ModelClass = MODEL_MAP[model_variant]
    model = ModelClass(device=device)
    
    # Train
    model.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        save_best_only=True,
        early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        checkpoint_interval=1000,  # Effectively disabled
    )
    
    print(f"\n✅ Training completed for RF-DETR {model_variant.upper()}")
    print(f"   Checkpoints saved at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR for tank detection')
    parser.add_argument('--model', type=str, default='large',
                        choices=['large', 'base', 'medium', 'small', 'nano'],
                        help='RF-DETR model variant')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to COCO-format dataset directory')
    parser.add_argument('--output_dir', type=str, default='./runs/rfdetr',
                        help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--grad_accum', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=60,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    train_rfdetr(
        model_variant=args.model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        early_stopping_patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
