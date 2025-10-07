#!/usr/bin/env python3
"""
Generate segmentation masks from COCO RLE format JSON file
Creates mask_cnos folders for each scene compatible with FoundationPose
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from collections import defaultdict
import argparse
    

def rle_to_mask(rle_counts, height, width):
    """Convert RLE counts to binary mask"""
    total_pixels = height * width
    mask = np.zeros(total_pixels, dtype=np.uint8)
    
    current_pos = 0
    for i, run_length in enumerate(rle_counts):
        if current_pos >= total_pixels:
            print(f"Warning: RLE data exceeds image size. Stopping at position {current_pos}")
            break
            
        end_pos = min(current_pos + run_length, total_pixels)
        
        if i % 2 == 1:  # Odd indices represent object pixels (1s)
            mask[current_pos:end_pos] = 1
        
        current_pos = end_pos
        
        if current_pos >= total_pixels:
            break
    
    # Reshape to image dimensions (COCO uses Fortran-style ordering)
    return mask.reshape(width, height).T


def generate_instance_mask(detections, height=480, width=640):
    """
    Generate instance segmentation mask from COCO detection data
    
    Args:
        detections: List of detection dictionaries with RLE segmentation
        height: Image height
        width: Image width
    
    Returns:
        mask: numpy array where each pixel value corresponds to category_id
    """
    # Initialize mask with zeros (background)
    instance_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Process each detection
    for detection in detections:
        category_id = detection['category_id']
        rle_counts = detection['segmentation']['counts']
        rle_size = detection['segmentation']['size']  # [height, width]
        
        # Use the size from the RLE data if available
        if rle_size and len(rle_size) == 2:
            rle_height, rle_width = rle_size
        else:
            rle_height, rle_width = height, width
        
        # Convert RLE to binary mask
        binary_mask = rle_to_mask(rle_counts, rle_height, rle_width)
        
        # Resize if dimensions don't match
        if binary_mask.shape != (height, width):
            binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Set pixels to category_id where object is present
        instance_mask[binary_mask > 0] = category_id
    
    return instance_mask


def process_json_to_masks(json_path, output_base_dir, height=480, width=640):
    """
    Process JSON file and generate mask_cnos folders for each scene
    
    Args:
        json_path: Path to JSON file with detections
        output_base_dir: Base directory for output (e.g., ycbv_test_bop19/test)
        height: Image height
        width: Image width
    """
    # Load JSON data
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group detections by scene_id and image_id
    detections_by_scene = defaultdict(lambda: defaultdict(list))
    
    for detection in data:
        scene_id = detection['scene_id']
        image_id = detection['image_id']
        detections_by_scene[scene_id][image_id].append(detection)
    
    print(f"\nFound {len(detections_by_scene)} scenes")
    
    # Process each scene
    for scene_id, images in detections_by_scene.items():
        # Create scene folder path (format: 000048)
        scene_folder = os.path.join(output_base_dir, f"{scene_id:06d}")
        mask_cnos_folder = os.path.join(scene_folder, "mask_sam6d_fine_sam")
        
        # Create mask_cnos directory
        os.makedirs(mask_cnos_folder, exist_ok=True)
        
        print(f"\nProcessing scene {scene_id:06d} ({len(images)} images)")
        print(f"  Output folder: {mask_cnos_folder}")
        
        # Process each image in the scene
        for image_id, detections in sorted(images.items()):
            # Generate mask for this image
            mask = generate_instance_mask(detections, height, width)
            
            # Save mask with same naming as rgb images (format: 000000.png)
            mask_filename = f"{image_id:06d}.png"
            mask_path = os.path.join(mask_cnos_folder, mask_filename)
            cv2.imwrite(mask_path, mask)
            
            # Print statistics
            unique_values = np.unique(mask)
            non_zero = unique_values[unique_values > 0]
            print(f"  Image {image_id:06d}: {len(non_zero)} objects detected (categories: {non_zero.tolist()})")
    
    print("\n✓ Mask generation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate CNOS-format masks from RLE JSON')
    parser.add_argument('--json', type=str, 
                       default='./ycbv_test_bop19/sam6d_fine_ycbv-test.json',
                       help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='./ycbv_test_bop19/test',
                       help='Output base directory (test folder)')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    
    args = parser.parse_args()
    
    # Verify JSON file exists
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        return
    
    # Verify output directory exists
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        print(f"Creating directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process JSON and generate masks
    process_json_to_masks(args.json, args.output_dir, args.height, args.width)


if __name__ == "__main__":
    main()