#!/usr/bin/env python3
"""
COCO to YOLO Label Conversion Script

This script converts COCO-format annotations to YOLO format and creates
the dataset.yaml configuration file for YOLO training.

Usage:
    python create_yolo_labels.py --dataset_path /path/to/coco/dataset
"""

import os
import json
import glob
import argparse


def convert_coco_to_yolo(coco_ann_file, output_label_dir):
    """
    Convert COCO annotations to YOLO format.
    
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    
    Args:
        coco_ann_file: Path to COCO annotation JSON file
        output_label_dir: Output directory for YOLO label files
    """
    os.makedirs(output_label_dir, exist_ok=True)
    
    with open(coco_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image_id to image info
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    converted_count = 0
    
    # Process each image
    for image_id, anns in image_annotations.items():
        image_info = image_id_to_info[image_id]
        filename = image_info['file_name']
        
        img_width = image_info['width']
        img_height = image_info['height']
        
        # Create label file name
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(output_label_dir, label_filename)
        
        # Convert each annotation
        yolo_lines = []
        for ann in anns:
            # COCO bbox format: [x, y, width, height] (top-left corner)
            x, y, w, h = ann['bbox']
            
            # Convert to YOLO format: [center_x, center_y, width, height] (normalized)
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # Clamp values to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            # YOLO format: class_id center_x center_y width height
            class_id = 0  # Single class: tank
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_lines.append(yolo_line)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted_count += 1
    
    print(f"✓ Converted {converted_count} images with annotations")
    return converted_count


def organize_dataset_structure(dataset_path):
    """
    Organize dataset structure for YOLO format.
    Creates images/ and labels/ subdirectories.
    """
    print("📁 Organizing dataset structure...")
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_path, split)
        images_dir = os.path.join(split_dir, 'images')
        
        os.makedirs(images_dir, exist_ok=True)
        
        # Move image files to images/ subdirectory
        image_files = glob.glob(os.path.join(split_dir, "*.jpg")) + \
                     glob.glob(os.path.join(split_dir, "*.png"))
        
        moved_count = 0
        for img_file in image_files:
            filename = os.path.basename(img_file)
            new_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(new_path):
                os.rename(img_file, new_path)
                moved_count += 1
        
        print(f"  {split}: {moved_count} images moved to images/ directory")


def create_yolo_dataset_yaml(dataset_path, output_yaml_path):
    """
    Create dataset.yaml file for YOLO training.
    
    Args:
        dataset_path: Path to the processed dataset
        output_yaml_path: Path to save the yaml file
    """
    # Convert COCO annotations to YOLO format
    print("🔄 Converting COCO annotations to YOLO format...")
    
    for split in ['train', 'valid', 'test']:
        coco_file = os.path.join(dataset_path, split, '_annotations.coco.json')
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(coco_file):
            print(f"Converting {split} annotations...")
            convert_coco_to_yolo(coco_file, labels_dir)
        else:
            print(f"  {split}: No COCO file found at {coco_file}")
    
    # Create YAML configuration
    yaml_content = f"""# YOLO Tank Detection Dataset
# Created for DroneVision Challenge

path: {dataset_path}  # dataset root dir
train: train/images   # train images (relative to 'path') 
val: valid/images     # val images (relative to 'path')
test: test/images     # test images (optional)

# Classes
names:
  0: tank  # class 0 = tank

# Dataset info
nc: 1  # number of classes
"""
    
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created dataset.yaml at {output_yaml_path}")
    return output_yaml_path


def verify_dataset(dataset_path):
    """Verify the final dataset structure."""
    print("\n📊 Final dataset structure verification:")
    
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(dataset_path, split, 'images')
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(images_dir):
            image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                         glob.glob(os.path.join(images_dir, "*.png"))
            print(f"  {split}/images: {len(image_files)} image files")
        else:
            print(f"  {split}/images: Directory not found")
        
        if os.path.exists(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
            print(f"  {split}/labels: {len(label_files)} label files")
        else:
            print(f"  {split}/labels: Directory not found")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to COCO-format dataset')
    parser.add_argument('--output_yaml', type=str, default=None,
                        help='Path to save dataset.yaml (default: dataset_path/dataset.yaml)')
    
    args = parser.parse_args()
    
    if args.output_yaml is None:
        args.output_yaml = os.path.join(args.dataset_path, 'dataset.yaml')
    
    # Organize dataset structure
    organize_dataset_structure(args.dataset_path)
    
    # Create YOLO labels and yaml
    create_yolo_dataset_yaml(args.dataset_path, args.output_yaml)
    
    # Verify
    verify_dataset(args.dataset_path)
    
    print(f"\n✅ Dataset converted to YOLO format!")
    print(f"🎯 Ready for YOLO training with: {args.output_yaml}")


if __name__ == "__main__":
    main()
