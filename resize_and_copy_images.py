#!/usr/bin/env python3
"""
Script to recursively search for images, resize them to max 512x512 pixels,
and copy them to a training folder.
"""

import os
from pathlib import Path
from PIL import Image
import shutil

# Configuration
SOURCE_DIR = r"D:\delete2025\20251205_patent_abnormal\20260106_pics"
TARGET_DIR = r".\training\step1_pics"
MAX_SIZE = 512

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def resize_image(image_path, output_path, max_size=512):
    """
    Resize image to fit within max_size x max_size while maintaining aspect ratio.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the resized image
        max_size: Maximum dimension (width or height) in pixels
    """
    try:
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary (for PNG with transparency)
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            elif img.mode not in ('RGB', 'L'):  # L is grayscale
                img = img.convert('RGB')
            
            # Get current dimensions
            width, height = img.size
            
            # Check if resizing is needed
            if width <= max_size and height <= max_size:
                # No resizing needed, just save
                img.save(output_path, quality=95, optimize=True)
                return True, width, height, width, height
            
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            
            # Resize using high-quality Lanczos resampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image
            resized_img.save(output_path, quality=95, optimize=True)
            
            return True, width, height, new_width, new_height
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False, 0, 0, 0, 0


def process_images(source_dir, target_dir, max_size=512):
    """
    Recursively find all images in source_dir, resize them, and copy to target_dir.
    
    Args:
        source_dir: Source directory to search for images
        target_dir: Target directory to save resized images
        max_size: Maximum dimension for resized images
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files recursively
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(source_path.rglob(f"*{ext}"))
        image_files.extend(source_path.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates (in case of case-insensitive filesystem)
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"No images found in {source_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Source: {source_dir}")
    print(f"Target: {target_path.absolute()}")
    print(f"Max size: {max_size}x{max_size} pixels")
    print("-" * 80)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, img_path in enumerate(image_files, 1):
        # Generate output filename
        # Keep original filename, but if there are duplicates from different folders,
        # add parent folder name as prefix
        relative_path = img_path.relative_to(source_path)
        
        # If image is in a subfolder, include parent folder name to avoid conflicts
        if len(relative_path.parts) > 1:
            # Get parent folder name and combine with filename
            parent_folder = relative_path.parts[-2]
            output_filename = f"{parent_folder}_{img_path.name}"
        else:
            output_filename = img_path.name
        
        output_path = target_path / output_filename
        
        # Check if output already exists
        if output_path.exists():
            print(f"[{idx}/{len(image_files)}] Skipping (exists): {output_filename}")
            skipped += 1
            continue
        
        # Resize and save
        success, orig_w, orig_h, new_w, new_h = resize_image(
            img_path, output_path, max_size
        )
        
        if success:
            if orig_w == new_w and orig_h == new_h:
                print(f"[{idx}/{len(image_files)}] Copied: {output_filename} ({orig_w}x{orig_h})")
            else:
                print(f"[{idx}/{len(image_files)}] Resized: {output_filename} ({orig_w}x{orig_h} → {new_w}x{new_h})")
            processed += 1
        else:
            errors += 1
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {len(image_files)}")
    print(f"\nOutput directory: {target_path.absolute()}")


if __name__ == "__main__":
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory does not exist: {SOURCE_DIR}")
        exit(1)
    
    # Process images
    process_images(SOURCE_DIR, TARGET_DIR, MAX_SIZE)
