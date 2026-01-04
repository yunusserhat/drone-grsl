#!/usr/bin/env python3
"""
YOLO Inference Script for DroneVision Tank Detection

This script runs inference using trained YOLO models and generates submission CSV.

Usage:
    python inference_yolo.py --model yolov12x --model_path /path/to/best.pt --test_dir /path/to/test
"""

import os
import glob
import csv
import random
import argparse
from tqdm import tqdm
from ultralytics import YOLO


def run_inference(
    model_path: str,
    test_dir: str,
    output_csv: str,
    conf_threshold: float = 0.25,
    device: str = "0",
):
    """
    Run YOLO inference on test images and generate submission CSV.
    
    Args:
        model_path: Path to trained YOLO weights (best.pt)
        test_dir: Directory containing test images
        output_csv: Path to save submission CSV
        conf_threshold: Confidence threshold for detections
        device: GPU device ID
    """
    print(f"🔍 Running inference")
    print(f"   Model: {model_path}")
    print(f"   Test dir: {test_dir}")
    print(f"   Output: {output_csv}")
    
    # Load model
    model = YOLO(model_path)
    
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
        
        results = model(img_path, verbose=False, conf=conf_threshold)
        
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            
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
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
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
                # Dummy values for no detection (required by submission format)
                x_min = str(random.randint(1, 100))
                y_min = str(random.randint(1, 100))
                x_max = str(random.randint(200, 300))
                y_max = str(random.randint(200, 300))
                class_id = 0
                
                writer.writerow([image_id, x_min, y_min, x_max, y_max, class_id])
    
    print(f"\n📄 Submission saved: {output_csv}")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(description='YOLO inference for tank detection')
    parser.add_argument('--model', type=str, default='yolov12x',
                        help='Model name (for output naming)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained YOLO weights (best.pt)')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./submissions',
                        help='Output directory for submission CSV')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    output_csv = os.path.join(args.output_dir, f"{args.model}_submission.csv")
    
    run_inference(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_csv=output_csv,
        conf_threshold=args.conf,
        device=args.device,
    )


if __name__ == "__main__":
    main()
