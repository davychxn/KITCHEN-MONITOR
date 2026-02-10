"""
Dataset utilities for temporal sequence classification
Handles grouping of sequential images and data preparation
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
from collections import defaultdict
from tqdm import tqdm
import cv2
import numpy as np
import json


def merge_bboxes_with_outlier_removal(bboxes, threshold=2.0):
    """
    Merge multiple bounding boxes into one, removing outliers based on size
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        threshold: Standard deviation threshold for outlier detection (default: 2.0)
        
    Returns:
        Merged bounding box [x1, y1, x2, y2] or None if no valid boxes
    """
    if len(bboxes) == 0:
        return None
    
    if len(bboxes) == 1:
        return bboxes[0]
    
    # Calculate area for each bbox
    areas = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    
    areas = np.array(areas)
    
    # Remove outliers based on area (using z-score method)
    if len(areas) >= 3:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        if std_area > 0:
            z_scores = np.abs((areas - mean_area) / std_area)
            valid_indices = z_scores < threshold
            
            # Keep at least 1 bbox
            if np.sum(valid_indices) > 0:
                bboxes = [bbox for i, bbox in enumerate(bboxes) if valid_indices[i]]
    
    # Merge remaining bboxes by taking min/max coordinates
    x1_min = min(bbox[0] for bbox in bboxes)
    y1_min = min(bbox[1] for bbox in bboxes)
    x2_max = max(bbox[2] for bbox in bboxes)
    y2_max = max(bbox[3] for bbox in bboxes)
    
    return [x1_min, y1_min, x2_max, y2_max]


def save_sequence_bboxes(bbox_dict, save_path):
    """
    Save bounding boxes for sequences to JSON file
    
    Args:
        bbox_dict: Dictionary mapping (state, group_id) -> bbox [x1, y1, x2, y2]
        save_path: Path to save JSON file
    """
    # Convert tuple keys to strings for JSON serialization
    json_dict = {}
    for (state, group_id), bbox in bbox_dict.items():
        key = f"{state}_{group_id}"
        json_dict[key] = bbox
    
    with open(save_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


def load_sequence_bboxes(load_path):
    """
    Load bounding boxes for sequences from JSON file
    
    Args:
        load_path: Path to JSON file
        
    Returns:
        Dictionary mapping (state, group_id) -> bbox [x1, y1, x2, y2]
    """
    if not Path(load_path).exists():
        return {}
    
    with open(load_path, 'r') as f:
        json_dict = json.load(f)
    
    # Convert string keys back to tuples
    bbox_dict = {}
    for key, bbox in json_dict.items():
        parts = key.split('_')
        if len(parts) >= 2:
            # Handle state names with underscores (like 'on_fire')
            # Try to find the group_id (last numeric part)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].isdigit():
                    state = '_'.join(parts[:i])
                    group_id = int(parts[i])
                    bbox_dict[(state, group_id)] = bbox
                    break
    
    return bbox_dict


def parse_filename(filename):
    """
    Parse filename to extract label and group info
    
    Supports multiple formats:
    - Format 1: object_state_groupId_frameId.jpg (e.g., cooking-pot_boiling_1_1.jpg)
    - Format 2: object_state_*.jpg (e.g., cooking-pot_normal_Clipboard_*.jpg)
    
    Args:
        filename: Image filename or path
        
    Returns:
        Tuple of (state, group_id, frame_id) or (None, None, None) if parsing fails
    """
    stem = Path(filename).stem
    
    # Try temporal format first
    match = re.match(r'(.+?)_(boiling|normal|on-fire|smoking)_(\d+)_(\d+)', stem)
    if match:
        state = match.group(2)
        group_id = int(match.group(3))
        frame_id = int(match.group(4))
        if state == 'on-fire':
            state = 'on_fire'
        return state, group_id, frame_id
    
    # Try single image format
    match = re.match(r'(.+?)_(boiling|normal|on-fire|smoking)_(.+)', stem)
    if match:
        state = match.group(2)
        if state == 'on-fire':
            state = 'on_fire'
        # Use filename as unique group ID for single images
        return state, hash(stem) % 10000, 1
    
    return None, None, None


def group_images_by_sequence(image_files, num_frames=3, include_labels=True):
    """
    Group images into temporal sequences based on filename patterns
    
    Args:
        image_files: List of image file paths
        num_frames: Expected number of frames per sequence (default: 3)
        include_labels: Whether to extract labels from filenames
        
    Returns:
        If include_labels=True:
            Tuple of (sequence_groups, labels) where:
            - sequence_groups: List of lists, each containing paths to sequential frames
            - labels: List of label strings (e.g., 'boiling', 'normal', etc.)
        If include_labels=False:
            Dictionary mapping group_key -> [frame_paths] sorted by frame_id
    """
    # Group images by (state, group_id) or just group_id
    sequence_dict = defaultdict(list)
    
    for img_path in image_files:
        state, group_id, frame_id = parse_filename(img_path.name if hasattr(img_path, 'name') else img_path)
        
        if state is not None or not include_labels:
            if include_labels:
                key = (state, group_id)
            else:
                # For verification without labels, try to extract group info differently
                # Pattern: anything_groupId_frameId.jpg
                stem = Path(img_path).stem
                match = re.match(r'(.+?)_(\d+)_(\d+)', stem)
                if match:
                    prefix = match.group(1)
                    group_id = int(match.group(2))
                    frame_id = int(match.group(3))
                    key = (prefix, group_id)
                else:
                    # Fallback: use filename without extension as key
                    key = (stem, 0)
            
            sequence_dict[key].append((frame_id, str(img_path)))
    
    # Sort frames within each group and create final sequences
    if include_labels:
        final_sequences = []
        final_labels = []
        
        for (state, group_id), frames in sequence_dict.items():
            frames.sort(key=lambda x: x[0])
            frame_paths = [path for _, path in frames]
            final_sequences.append(frame_paths)
            final_labels.append(state)
        
        return final_sequences, final_labels
    else:
        # For verification, return dictionary with sorted frames
        grouped = {}
        for key, frames in sequence_dict.items():
            frames.sort(key=lambda x: x[0])
            frame_paths = [path for _, path in frames]
            grouped[key] = frame_paths
        return grouped


def get_standard_transforms(augment=False):
    """
    Get standard image transforms for training/validation
    
    Args:
        augment: Whether to include augmentation transforms
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    # Base transform for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Augmentation for training
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
    else:
        train_transform = val_transform
        aug_transform = None
    
    return {
        'train': train_transform,
        'val': val_transform,
        'augment': aug_transform
    }


class TemporalSequenceDataset(Dataset):
    """
    Dataset for temporal sequence classification
    Always uses 3-frame sequences for consistency between training and verification
    """
    
    def __init__(self, sequence_groups, labels, transform=None, augment_transform=None, 
                 num_frames=3, class_to_idx=None):
        """
        Args:
            sequence_groups: List of image path sequences (each group is a list of paths)
            labels: List of labels (one per group) - can be strings or indices
            transform: Transform to apply to each frame
            augment_transform: Optional augmentation transform to apply before main transform
            num_frames: Number of frames per sequence (default: 3)
            class_to_idx: Optional dictionary mapping class names to indices
        """
        self.sequence_groups = sequence_groups
        self.labels = labels
        self.transform = transform
        self.augment_transform = augment_transform
        self.num_frames = num_frames
        self.class_to_idx = class_to_idx
        
        # Convert string labels to indices if needed
        if class_to_idx is not None and labels and isinstance(labels[0], str):
            self.labels = [class_to_idx[label] for label in labels]
    
    def __len__(self):
        return len(self.sequence_groups)
    
    def __getitem__(self, idx):
        image_paths = self.sequence_groups[idx]
        label = self.labels[idx]
        
        # Load all frames in the sequence
        frames = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            
            # Apply augmentation if provided (same for all frames in sequence)
            if self.augment_transform:
                image = self.augment_transform(image)
            
            # Apply main transform
            if self.transform:
                image = self.transform(image)
            
            frames.append(image)
        
        # Pad or repeat frames to get exactly num_frames
        while len(frames) < self.num_frames:
            # Repeat last frame if we have fewer than num_frames
            frames.append(frames[-1].clone() if len(frames) > 0 else frames[0].clone())
        
        # Take only first num_frames if we have more
        frames = frames[:self.num_frames]
        
        # Stack frames along channel dimension (C*T, H, W)
        # For 3 frames of RGB: [3, 224, 224] -> [9, 224, 224]
        stacked = torch.cat(frames, dim=0)
        
        return stacked, label


def load_temporal_dataset(data_dir, detector=None, num_frames=3, verbose=True):
    """
    Load dataset and organize into temporal sequences
    Merges bounding boxes across frames for consistent cropping
    
    Args:
        data_dir: Directory containing images or pre-cropped images
        detector: Optional KitchenObjectDetector for YOLO cropping
        num_frames: Number of frames per sequence (default: 3)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (sequence_groups, labels, label_counts)
        - sequence_groups: List of lists, each containing paths to sequential frames
        - labels: List of label strings
        - label_counts: Dictionary of label -> count
    """
    data_dir = Path(data_dir)
    bbox_file = data_dir / 'sequence_bboxes.json'
    
    # Check if pre-cropped images and bboxes exist
    crop_dir = data_dir / 'yolo_crops'
    if crop_dir.exists() and bbox_file.exists():
        crop_files = list(crop_dir.glob('*.jpg')) + list(crop_dir.glob('*.jpeg')) + list(crop_dir.glob('*.png'))
        if len(crop_files) > 0:
            if verbose:
                print(f"\n✓ Using existing YOLO crops from: {crop_dir}")
                print(f"✓ Found {len(crop_files)} pre-cropped images")
                print(f"✓ Loading saved bounding boxes from: {bbox_file.name}")
            use_existing_crops = True
            image_files = crop_files
        else:
            use_existing_crops = False
            image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg')) + list(data_dir.glob('*.png'))
    else:
        use_existing_crops = False
        image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg')) + list(data_dir.glob('*.png'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    if not use_existing_crops and detector is not None:
        if verbose:
            print(f"\nFound {len(image_files)} images")
            print("\n=== YOLO-based Detection & Merging ===")
            print("Detecting objects, merging bboxes with outlier removal...\n")
    
    # First pass: Group images and detect bboxes for each frame
    sequence_dict = defaultdict(list)
    sequence_bboxes = defaultdict(list)  # Store all bboxes for each sequence
    failed_detections = []
    
    if not use_existing_crops and detector is not None:
        # Detect bboxes for all frames
        for img_path in tqdm(image_files, desc="Detecting bboxes", disable=not verbose):
            state, group_id, frame_id = parse_filename(img_path.name)
            
            if state:
                img = cv2.imread(str(img_path))
                if img is not None:
                    detections = detector.get_detections_with_crops(img, apply_padding=False)
                    
                    if len(detections) > 0:
                        key = (state, group_id)
                        sequence_dict[key].append((frame_id, str(img_path), img))
                        
                        # Store all bboxes for this frame (without padding)
                        for det in detections:
                            sequence_bboxes[key].append(det['bbox'])
                    else:
                        failed_detections.append(img_path.name)
        
        # Second pass: Merge bboxes for each sequence and crop
        crop_dir = data_dir / 'yolo_crops'
        crop_dir.mkdir(exist_ok=True)
        
        merged_bboxes = {}  # Store merged bbox for each sequence
        
        for key in tqdm(sequence_dict.keys(), desc="Merging & cropping", disable=not verbose):
            bboxes = sequence_bboxes[key]
            
            # Merge bboxes with outlier removal
            merged_bbox = merge_bboxes_with_outlier_removal(bboxes)
            
            if merged_bbox is not None:
                merged_bboxes[key] = merged_bbox
                
                # Apply padding to merged bbox
                x1, y1, x2, y2 = merged_bbox
                h, w = sequence_dict[key][0][2].shape[:2]  # Get image dimensions
                
                box_w = x2 - x1
                box_h = y2 - y1
                pad_w = int(box_w * detector.padding_ratio)
                pad_h = int(box_h * detector.padding_ratio)
                
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(w, x2 + pad_w)
                y2_pad = min(h, y2 + pad_h)
                
                # Crop all frames in this sequence with the same merged bbox
                frames = sequence_dict[key]
                cropped_frames = []
                
                for frame_id, img_path, img in frames:
                    crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                    crop_path = crop_dir / Path(img_path).name
                    cv2.imwrite(str(crop_path), crop)
                    cropped_frames.append((frame_id, str(crop_path)))
                
                # Update sequence_dict with cropped paths
                sequence_dict[key] = cropped_frames
        
        # Save merged bboxes for future use
        save_sequence_bboxes(merged_bboxes, bbox_file)
        if verbose:
            print(f"\n✓ Saved merged bounding boxes to: {bbox_file.name}")
    
    elif use_existing_crops:
        # Group existing crops
        for img_path in image_files:
            state, group_id, frame_id = parse_filename(img_path.name)
            if state:
                key = (state, group_id)
                sequence_dict[key].append((frame_id, str(img_path)))
    else:
        # No detector, use full images
        for img_path in image_files:
            state, group_id, frame_id = parse_filename(img_path.name)
            if state:
                key = (state, group_id)
                sequence_dict[key].append((frame_id, str(img_path)))
    
    # Sort frames within each group and create final sequences
    final_sequences = []
    final_labels = []
    label_counts = defaultdict(int)
    
    for (state, group_id), frames in sequence_dict.items():
        # Sort by frame_id
        frames.sort(key=lambda x: x[0])
        # Extract just the paths
        frame_paths = [path if isinstance(path, str) else path[1] for _, path in frames] if isinstance(frames[0], tuple) and len(frames[0]) > 1 else [path for _, path in frames]
        
        final_sequences.append(frame_paths)
        final_labels.append(state)
        label_counts[state] += 1
    
    if detector is not None and failed_detections and verbose:
        print(f"\n⚠ Warning: YOLO failed to detect objects in {len(failed_detections)} images")
    
    if verbose:
        print(f"\nDataset distribution (by {num_frames}-frame sequence groups):")
        for state, count in sorted(label_counts.items()):
            frames_per_group = sum(len(seq) for seq, lbl in zip(final_sequences, final_labels) 
                                 if lbl == state) / max(count, 1)
            print(f"  {state}: {count} groups ({frames_per_group:.1f} frames/group avg)")
    
    return final_sequences, final_labels, dict(label_counts)


def load_verification_sequences(input_dir, detector=None, num_frames=3, verbose=True):
    """
    Load and group images for verification with merged bounding boxes
    Uses saved bounding boxes or detects and merges new ones
    
    Args:
        input_dir: Directory containing sequential images
        detector: Optional KitchenObjectDetector for YOLO detection
        num_frames: Number of frames per sequence (default: 3)
        verbose: Whether to print progress information
        
    Returns:
        Dictionary mapping group_key -> dict with:
            - 'frame_paths': List of frame paths (or crop paths if detector used)
            - 'label': Ground truth label (if parseable from filename)
            - 'bbox': Merged bounding box used for cropping (if available)
    """
    input_dir = Path(input_dir)
    bbox_file = input_dir / 'sequence_bboxes.json'
    crop_dir = input_dir / 'yolo_crops'
    
    # Check if we have existing crops and bboxes
    use_existing = crop_dir.exists() and bbox_file.exists()
    
    if use_existing:
        crop_files = list(crop_dir.glob('*.jpg')) + list(crop_dir.glob('*.jpeg')) + list(crop_dir.glob('*.png'))
        if len(crop_files) > 0:
            if verbose:
                print(f"\n✓ Using existing crops from: {crop_dir}")
                print(f"✓ Loading saved bounding boxes from: {bbox_file.name}")
            image_files = crop_files
            saved_bboxes = load_sequence_bboxes(bbox_file)
        else:
            use_existing = False
    
    if not use_existing:
        image_files = sorted(input_dir.glob("*.jpg"))
        if len(image_files) == 0:
            image_files = sorted(input_dir.glob("*.jpeg")) + sorted(input_dir.glob("*.png"))
        saved_bboxes = {}
    
    grouped_dict = defaultdict(list)
    
    # Group images by sequence
    for img_path in image_files:
        state, group_id, frame_id = parse_filename(img_path.name)
        
        # Group by state and group_id if available, otherwise by filename pattern
        if state and group_id is not None:
            key = f"{state}_{group_id}"
            grouped_dict[key].append({
                'frame_id': frame_id,
                'path': img_path,
                'label': state,
                'state': state,
                'group_id': group_id
            })
        else:
            # Try to extract any grouping pattern
            stem = Path(img_path).stem
            match = re.match(r'(.+?)_(\d+)_(\d+)', stem)
            if match:
                prefix = match.group(1)
                group_id = int(match.group(2))
                frame_id = int(match.group(3))
                key = f"{prefix}_{group_id}"
                grouped_dict[key].append({
                    'frame_id': frame_id,
                    'path': img_path,
                    'label': None,
                    'state': None,
                    'group_id': group_id
                })
    
    # If detector provided and no existing crops, detect and merge bboxes
    if detector is not None and not use_existing:
        if verbose:
            print(f"\n=== YOLO Detection & Merging for Verification ===")
            print("Detecting objects and merging bboxes with outlier removal...\n")
        
        crop_dir.mkdir(exist_ok=True)
        sequence_bboxes = defaultdict(list)
        merged_bboxes = {}
        
        # First pass: detect all bboxes
        for key, frames in tqdm(grouped_dict.items(), desc="Detecting bboxes", disable=not verbose):
            for frame_info in frames:
                img = cv2.imread(str(frame_info['path']))
                if img is not None:
                    detections = detector.get_detections_with_crops(img, apply_padding=False)
                    for det in detections:
                        sequence_bboxes[key].append(det['bbox'])
        
        # Second pass: merge and crop
        for key in tqdm(grouped_dict.keys(), desc="Merging & cropping", disable=not verbose):
            bboxes = sequence_bboxes[key]
            
            if len(bboxes) > 0:
                merged_bbox = merge_bboxes_with_outlier_removal(bboxes)
                
                if merged_bbox is not None:
                    # Store for JSON
                    if grouped_dict[key][0]['state'] and grouped_dict[key][0]['group_id'] is not None:
                        bbox_key = (grouped_dict[key][0]['state'], grouped_dict[key][0]['group_id'])
                        merged_bboxes[bbox_key] = merged_bbox
                    
                    # Apply padding
                    x1, y1, x2, y2 = merged_bbox
                    img = cv2.imread(str(grouped_dict[key][0]['path']))
                    h, w = img.shape[:2]
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    pad_w = int(box_w * detector.padding_ratio)
                    pad_h = int(box_h * detector.padding_ratio)
                    
                    x1_pad = max(0, x1 - pad_w)
                    y1_pad = max(0, y1 - pad_h)
                    x2_pad = min(w, x2 + pad_w)
                    y2_pad = min(h, y2 + pad_h)
                    
                    # Crop all frames with merged bbox
                    for frame_info in grouped_dict[key]:
                        img = cv2.imread(str(frame_info['path']))
                        crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                        crop_path = crop_dir / frame_info['path'].name
                        cv2.imwrite(str(crop_path), crop)
                        frame_info['path'] = crop_path
                        frame_info['bbox'] = [x1_pad, y1_pad, x2_pad, y2_pad]
        
        # Save merged bboxes
        if len(merged_bboxes) > 0:
            save_sequence_bboxes(merged_bboxes, bbox_file)
            if verbose:
                print(f"\n✓ Saved merged bounding boxes to: {bbox_file.name}")
    
    # Sort frames within each group and build result
    result = {}
    for key, frames in grouped_dict.items():
        frames.sort(key=lambda x: x['frame_id'])
        
        # Get bbox if available
        bbox = None
        if frames[0]['state'] and frames[0]['group_id'] is not None:
            bbox_key = (frames[0]['state'], frames[0]['group_id'])
            bbox = saved_bboxes.get(bbox_key)
        
        result[key] = {
            'frame_paths': [f['path'] for f in frames[:num_frames]],
            'label': frames[0]['label'] if frames else None,
            'bbox': bbox
        }
    
    return result
