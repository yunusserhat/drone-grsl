#!/usr/bin/env python3
"""
RF-DETR Inference Script for DroneVision Tank Detection

This script runs inference using trained RF-DETR models.

Usage:
    python inference_rfdetr.py --checkpoint /path/to/checkpoint.pth --test_dir /path/to/test
"""

import os
import glob
import csv
import random
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from rfdetr import RFDETRLarge, RFDETRBase


def run_inference(
    checkpoint_path: str,
    test_dir: str,
    output_csv: str,
    model_variant: str = 'large',
    conf_threshold: float = 0.1,
    device: str = "cuda",
):
    """
    Run RF-DETR inference on test images and generate submission CSV.
    
    Args:
        checkpoint_path: Path to trained RF-DETR checkpoint
        test_dir: Directory containing test images
        output_csv: Path to save submission CSV
        model_variant: RF-DETR variant (large, base)
        conf_threshold: Confidence threshold for detections
        device: Device (cuda or cpu)
    """
    print(f"🔍 Running RF-DETR inference")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Test dir: {test_dir}")
    print(f"   Output: {output_csv}")
    
    # Initialize model
    if model_variant == 'large':
        model = RFDETRLarge(device=device)
    else:
        model = RFDETRBase(device=device)
    
    # Load checkpoint
    model.load(checkpoint_path)
    
    # Find test images
    test_images = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_dir, "*.png"))
    test_images.sort()
    
    print(f"   Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return
    
    # Run inference
    results_list = []
    no_detection_count = 0
    
    for img_path in tqdm(test_images, desc="Inference"):
        filename = os.path.basename(img_path)
        image_id = filename.replace('frame_', '').replace('.jpg', '').replace('.png', '')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Run inference
        detections_sv = model.predict(image, threshold=conf_threshold)
        
        detections = []
        if len(detections_sv) > 0:
            boxes = detections_sv.xyxy
            scores = detections_sv.confidence
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'x_min': int(x1),
                    'y_min': int(y1),
                    'x_max': int(x2),
                    'y_max': int(y2),
                    'confidence': float(scores[i])
                })
        
        if detections:
            results_list.append({'image_id': image_id, 'detections': detections})
        else:
            no_detection_count += 1
            results_list.append({'image_id': image_id, 'detections': []})
    
    print(f"\n✅ Inference completed!")
    print(f"   Images with detections: {len(test_images) - no_detection_count}")
    print(f"   Images without detections: {no_detection_count}")
    
    # Create submission CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    all_test_image_ids = sorted(set([int(r['image_id']) for r in results_list]))
    predictions_by_image = {r['image_id']: r['detections'] for r in results_list}
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"])
        
        for image_id in all_test_image_ids:
            image_id_str = str(image_id)
            
            if image_id_str in predictions_by_image and predictions_by_image[image_id_str]:
                detections = predictions_by_image[image_id_str]
                
                x_min = ';'.join([str(det['x_min']) for det in detections])
                y_min = ';'.join([str(det['y_min']) for det in detections])
                x_max = ';'.join([str(det['x_max']) for det in detections])
                y_max = ';'.join([str(det['y_max']) for det in detections])
                class_id = 0
                
                writer.writerow([image_id, x_min, y_min, x_max, y_max, class_id])
            else:
                # Dummy values for no detection
                x_min = str(random.randint(1, 100))
                y_min = str(random.randint(1, 100))
                x_max = str(random.randint(200, 300))
                y_max = str(random.randint(200, 300))
                class_id = 0
                
                writer.writerow([image_id, x_min, y_min, x_max, y_max, class_id])
    
    print(f"\n📄 Submission saved: {output_csv}")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(description='RF-DETR inference for tank detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained RF-DETR checkpoint')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_csv', type=str, default='./submissions/rfdetr_submission.csv',
                        help='Path to save submission CSV')
    parser.add_argument('--model', type=str, default='large',
                        choices=['large', 'base'],
                        help='RF-DETR variant')
    parser.add_argument('--conf', type=float, default=0.1,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_csv=args.output_csv,
        model_variant=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )


if __name__ == "__main__":
    main()
