#!/usr/bin/env python3
"""
Dataset Creation Script for DroneVision Tank Detection

This script converts the original drone dataset to a COCO-style format
with consistent image sizes for RF-DETR and YOLO training.

Usage:
    python create_dataset.py --source_dir /path/to/raw --target_dir /path/to/output
"""

import os
import json
import argparse
import random
from PIL import Image
from tqdm import tqdm

# Standard image size for consistency
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def polygon_to_bbox(polygon):
    """Convert polygon coordinates to bounding box format [x, y, width, height]"""
    x_coords = polygon[0::2]  # Even indices are x coordinates
    y_coords = polygon[1::2]  # Odd indices are y coordinates
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def create_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create COCO-format dataset from raw annotations.
    
    Args:
        source_dir: Path to source dataset directory
        target_dir: Path to output dataset directory
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
        seed: Random seed for reproducibility
    """
    print(f"Converting dataset from {source_dir} to {target_dir}")
    print(f"Split ratios: {train_ratio*100:.0f}% train, {val_ratio*100:.0f}% valid, {(1-train_ratio-val_ratio)*100:.0f}% test")

    # Create target directory structure
    os.makedirs(target_dir, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # Load all annotations from train split
    train_ann_file = os.path.join(source_dir, "train", "annotations.json")
    with open(train_ann_file, 'r') as f:
        all_annotations = json.load(f)
    
    print(f"Loaded {len(all_annotations)} annotation groups")
    
    # Collect training images
    train_images = []
    frames_dir = os.path.join(source_dir, "train", 'frames')
    
    if os.path.exists(frames_dir):
        image_files = [f for f in os.listdir(frames_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            source_path = os.path.join(frames_dir, img_file)
            train_images.append((img_file, source_path))
    
    print(f"Found {len(train_images)} training images")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(train_images)
    
    total_images = len(train_images)
    train_count = int(train_ratio * total_images)
    valid_count = int(val_ratio * total_images)

    splits = {
        'train': train_images[:train_count],
        'valid': train_images[train_count:train_count + valid_count],
        'test': train_images[train_count + valid_count:]
    }
    
    print(f"Split: {len(splits['train'])} train, {len(splits['valid'])} valid, {len(splits['test'])} test")
    
    # Process each split
    for split_name, split_images in splits.items():
        if len(split_images) == 0:
            print(f"Creating empty {split_name} split...")
            coco_data = {
                'info': {'description': f'Drone Tank Detection - {split_name} (empty)'},
                'categories': [{'id': 0, 'name': 'tank', 'supercategory': 'vehicle'}],
                'images': [],
                'annotations': []
            }
            ann_file = os.path.join(target_dir, split_name, '_annotations.coco.json')
            with open(ann_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            continue
            
        print(f"\nProcessing {split_name} split...")
        
        # Create COCO structure (0-based category IDs for RF-DETR)
        coco_data = {
            'info': {'description': f'Drone Tank Detection - {split_name}'},
            'categories': [{'id': 0, 'name': 'tank', 'supercategory': 'vehicle'}],
            'images': [],
            'annotations': []
        }
        
        image_id = 1
        ann_id = 1
        
        # Process each image
        for img_file, source_path in tqdm(split_images, desc=f"Processing {split_name}"):
            target_path = os.path.join(target_dir, split_name, img_file)
            
            with Image.open(source_path) as img:
                original_width, original_height = img.size
                
                # Resize image to standard size
                resized_img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
                resized_img.save(target_path, quality=95)
                
                # Calculate scale factors
                scale_x = TARGET_WIDTH / original_width
                scale_y = TARGET_HEIGHT / original_height
            
            # Add image to COCO data
            coco_data['images'].append({
                'id': image_id,
                'file_name': img_file,
                'width': TARGET_WIDTH,
                'height': TARGET_HEIGHT
            })
            
            # Find and process annotations
            img_basename = os.path.splitext(img_file)[0]
            
            # Try different annotation key formats
            annotation_key = None
            if img_basename in all_annotations:
                annotation_key = img_basename
            else:
                frame_num = ''.join(filter(str.isdigit, img_basename))
                if frame_num in all_annotations:
                    annotation_key = frame_num
            
            if annotation_key and annotation_key in all_annotations:
                img_annotations = all_annotations[annotation_key]
                
                if isinstance(img_annotations, list):
                    for ann in img_annotations:
                        if isinstance(ann, list) and len(ann) >= 3:
                            flat_points = []
                            for point in ann:
                                if isinstance(point, list) and len(point) == 2:
                                    scaled_x = point[0] * scale_x
                                    scaled_y = point[1] * scale_y
                                    flat_points.extend([scaled_x, scaled_y])
                            
                            if len(flat_points) >= 6:
                                bbox = polygon_to_bbox(flat_points)
                                area = bbox[2] * bbox[3]
                                
                                coco_data['annotations'].append({
                                    'id': ann_id,
                                    'image_id': image_id,
                                    'category_id': 0,
                                    'bbox': bbox,
                                    'area': area,
                                    'segmentation': [flat_points],
                                    'iscrowd': 0
                                })
                                ann_id += 1
            
            image_id += 1
        
        # Save COCO annotations
        ann_file = os.path.join(target_dir, split_name, '_annotations.coco.json')
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"{split_name}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    # Verify image sizes
    print("\n✅ Dataset creation completed!")
    print("\nVerifying image sizes:")
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(target_dir, split)
        image_files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
        if image_files:
            img_path = os.path.join(split_dir, image_files[0])
            with Image.open(img_path) as img:
                print(f"  {split}: {img.size[0]}x{img.size[1]}")


def main():
    parser = argparse.ArgumentParser(description='Create COCO-format dataset for tank detection')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to source dataset directory')
    parser.add_argument('--target_dir', type=str, required=True,
                        help='Path to output dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    create_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
