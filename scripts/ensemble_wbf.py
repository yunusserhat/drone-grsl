#!/usr/bin/env python3
"""
Weighted Boxes Fusion (WBF) Ensemble Script for DroneVision Tank Detection

This script combines predictions from multiple YOLO models and RF-DETR using
Weighted Boxes Fusion to produce the final ensemble predictions.

Usage:
    python ensemble_wbf.py \
        --yolo_models yolov9e yolov10x yolov11x yolov12x \
        --yolo_weights_dir /path/to/yolo/runs \
        --rfdetr_checkpoint /path/to/rfdetr/checkpoint.pth \
        --test_dir /path/to/test/images \
        --output_csv submission.csv
"""

import os
import glob
import csv
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from rfdetr import RFDETRLarge


# Default model weights for WBF
DEFAULT_WEIGHTS = {
    'yolov9e': 0.15,
    'yolov10x': 0.15,
    'yolov11x': 0.20,
    'yolov12x': 0.25,
    'rfdetr': 0.25,
}


def load_yolo_models(model_names, weights_dir):
    """Load multiple YOLO models from weights directory."""
    models = {}
    for name in model_names:
        weight_path = os.path.join(weights_dir, name, 'weights', 'best.pt')
        if os.path.exists(weight_path):
            print(f"  Loading {name} from {weight_path}")
            models[name] = YOLO(weight_path)
        else:
            print(f"  ⚠️ Weights not found for {name}: {weight_path}")
    return models


def load_rfdetr_model(checkpoint_path, device='cuda'):
    """Load RF-DETR model from checkpoint."""
    print(f"  Loading RF-DETR from {checkpoint_path}")
    model = RFDETRLarge(device=device)
    model.load(checkpoint_path)
    return model


def get_yolo_predictions(model, image_path, conf_threshold=0.25):
    """Get predictions from a YOLO model."""
    results = model(image_path, verbose=False, conf=conf_threshold)
    
    boxes = []
    scores = []
    
    if results and len(results) > 0 and results[0].boxes is not None:
        # Get image dimensions for normalization
        orig_shape = results[0].orig_shape
        img_h, img_w = orig_shape[0], orig_shape[1]
        
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            # Normalize to [0, 1]
            boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
            scores.append(float(confidences[i]))
    
    return boxes, scores


def get_rfdetr_predictions(model, image_path, conf_threshold=0.1):
    """Get predictions from RF-DETR model."""
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.size
    
    detections = model.predict(image, threshold=conf_threshold)
    
    boxes = []
    scores = []
    
    if len(detections) > 0:
        boxes_xyxy = detections.xyxy
        confidences = detections.confidence
        
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            # Normalize to [0, 1]
            boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
            scores.append(float(confidences[i]))
    
    return boxes, scores


def run_ensemble(
    yolo_models,
    rfdetr_model,
    test_dir,
    output_csv,
    model_weights=None,
    iou_threshold=0.6,
    skip_box_threshold=0.2,
    yolo_conf=0.25,
    rfdetr_conf=0.1,
):
    """
    Run ensemble inference using WBF.
    
    Args:
        yolo_models: Dict of YOLO model name -> model
        rfdetr_model: RF-DETR model (or None)
        test_dir: Directory containing test images
        output_csv: Path to save submission CSV
        model_weights: Dict of model name -> weight
        iou_threshold: IoU threshold for WBF
        skip_box_threshold: Skip boxes below this confidence
        yolo_conf: Confidence threshold for YOLO
        rfdetr_conf: Confidence threshold for RF-DETR
    """
    if model_weights is None:
        model_weights = DEFAULT_WEIGHTS
    
    # Find test images
    test_images = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_dir, "*.png"))
    test_images.sort()
    
    print(f"\n🔍 Running ensemble inference on {len(test_images)} images")
    print(f"   WBF parameters: iou_thr={iou_threshold}, skip_box_thr={skip_box_threshold}")
    
    # Get ordered list of models and their weights
    all_models = list(yolo_models.keys())
    if rfdetr_model is not None:
        all_models.append('rfdetr')
    
    weights_list = [model_weights.get(m, 0.2) for m in all_models]
    print(f"   Models: {all_models}")
    print(f"   Weights: {weights_list}")
    
    results_list = []
    no_detection_count = 0
    
    for img_path in tqdm(test_images, desc="Ensemble inference"):
        filename = os.path.basename(img_path)
        image_id = filename.replace('frame_', '').replace('.jpg', '').replace('.png', '')
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        # Collect predictions from all models
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # YOLO predictions
        for model_name, model in yolo_models.items():
            boxes, scores = get_yolo_predictions(model, img_path, yolo_conf)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append([0] * len(boxes))  # Single class
        
        # RF-DETR predictions
        if rfdetr_model is not None:
            boxes, scores = get_rfdetr_predictions(rfdetr_model, img_path, rfdetr_conf)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append([0] * len(boxes))
        
        # Apply WBF
        detections = []
        
        # Check if we have any predictions
        has_predictions = any(len(b) > 0 for b in all_boxes)
        
        if has_predictions:
            try:
                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                    all_boxes,
                    all_scores,
                    all_labels,
                    weights=weights_list,
                    iou_thr=iou_threshold,
                    skip_box_thr=skip_box_threshold,
                )
                
                # Convert back to pixel coordinates
                for i in range(len(fused_boxes)):
                    x1, y1, x2, y2 = fused_boxes[i]
                    detections.append({
                        'x_min': int(x1 * img_w),
                        'y_min': int(y1 * img_h),
                        'x_max': int(x2 * img_w),
                        'y_max': int(y2 * img_h),
                        'confidence': float(fused_scores[i])
                    })
            except Exception as e:
                print(f"WBF error for {filename}: {e}")
        
        if detections:
            results_list.append({'image_id': image_id, 'detections': detections})
        else:
            no_detection_count += 1
            results_list.append({'image_id': image_id, 'detections': []})
    
    print(f"\n✅ Ensemble inference completed!")
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
    parser = argparse.ArgumentParser(description='WBF Ensemble for tank detection')
    parser.add_argument('--yolo_models', nargs='+', 
                        default=['yolov9e', 'yolov10x', 'yolov11x', 'yolov12x'],
                        help='List of YOLO model names')
    parser.add_argument('--yolo_weights_dir', type=str, required=True,
                        help='Directory containing YOLO model weights')
    parser.add_argument('--rfdetr_checkpoint', type=str, default=None,
                        help='Path to RF-DETR checkpoint (optional)')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_csv', type=str, default='./ensemble_submission.csv',
                        help='Path to save submission CSV')
    parser.add_argument('--iou_thr', type=float, default=0.6,
                        help='IoU threshold for WBF')
    parser.add_argument('--skip_box_thr', type=float, default=0.2,
                        help='Skip boxes below this confidence')
    parser.add_argument('--yolo_conf', type=float, default=0.25,
                        help='Confidence threshold for YOLO models')
    parser.add_argument('--rfdetr_conf', type=float, default=0.1,
                        help='Confidence threshold for RF-DETR')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("🚀 Loading models...")
    
    # Load YOLO models
    yolo_models = load_yolo_models(args.yolo_models, args.yolo_weights_dir)
    
    # Load RF-DETR model
    rfdetr_model = None
    if args.rfdetr_checkpoint and os.path.exists(args.rfdetr_checkpoint):
        rfdetr_model = load_rfdetr_model(args.rfdetr_checkpoint, args.device)
    else:
        print("  ⚠️ RF-DETR checkpoint not provided or not found, skipping")
    
    # Run ensemble
    run_ensemble(
        yolo_models=yolo_models,
        rfdetr_model=rfdetr_model,
        test_dir=args.test_dir,
        output_csv=args.output_csv,
        iou_threshold=args.iou_thr,
        skip_box_threshold=args.skip_box_thr,
        yolo_conf=args.yolo_conf,
        rfdetr_conf=args.rfdetr_conf,
    )


if __name__ == "__main__":
    main()
