"""
Verify temporal classifier and draw on ORIGINAL-SIZED images (not cropped)
Groups 3 consecutive frames as one sample (as trained)
Draws bounding boxes and predictions on full-resolution original images
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import json
from datetime import datetime
import torchvision.transforms as transforms
from train_classifier_temporal import TemporalStateClassifier
from collections import defaultdict
from modules.yolo_detector import KitchenObjectDetector
from modules.dataset import parse_filename
import re


def merge_bboxes_with_outlier_removal(bboxes, threshold_percentile=20):
    """
    Merge multiple bounding boxes using median with outlier removal
    """
    if not bboxes:
        return None
    
    if len(bboxes) == 1:
        return bboxes[0]
    
    # Convert to numpy for easier manipulation
    boxes_array = np.array(bboxes)  # Shape: [N, 4] where each row is [x1, y1, x2, y2]
    
    # Calculate box areas and remove extreme outliers
    widths = boxes_array[:, 2] - boxes_array[:, 0]
    heights = boxes_array[:, 3] - boxes_array[:, 1]
    areas = widths * heights
    
    # Remove boxes with extreme areas (keep middle 80%)
    area_threshold_low = np.percentile(areas, threshold_percentile)
    area_threshold_high = np.percentile(areas, 100 - threshold_percentile)
    valid_mask = (areas >= area_threshold_low) & (areas <= area_threshold_high)
    
    if not np.any(valid_mask):
        valid_mask = np.ones(len(bboxes), dtype=bool)
    
    filtered_boxes = boxes_array[valid_mask]
    
    # Use median for robustness
    merged = np.median(filtered_boxes, axis=0).astype(int)
    
    return tuple(merged.tolist())


def run_verification_on_originals(input_dir, 
                                   output_dir,
                                   model_path='./yolo_training/pan_pot_classifier_temporal.pth',
                                   yolo_model_path='./yolo_training/kitchenware_detector2/weights/best.pt',
                                   backbone='mobilenet_v2',
                                   conf_threshold=0.3):
    """
    Run verification with bounding boxes drawn on ORIGINAL images
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading models...")
    print(f"Using device: {device}")
    
    # Load temporal classifier
    model = TemporalStateClassifier(backbone=backbone)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded temporal classifier: {model_path}")
    
    # Load YOLO detector
    detector = KitchenObjectDetector(model_path=yolo_model_path, conf_threshold=conf_threshold)
    detector.load_model()
    print(f"✓ Loaded YOLO: {yolo_model_path}")
    
    # Find all images and group them
    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg')) + list(input_dir.glob('*.png')))
    print(f"\nFound {len(image_files)} images")
    
    # Group images by sequence
    grouped_dict = defaultdict(list)
    for img_path in image_files:
        state, group_id, frame_id = parse_filename(img_path.name)
        if state and group_id is not None:
            key = f"{state}_{group_id}"
            grouped_dict[key].append({
                'frame_id': frame_id,
                'path': img_path,
                'label': state
            })
    
    print(f"\nFound {len(grouped_dict)} temporal groups to process")
    
    # Print grouping summary
    print("\nGrouping Summary:")
    label_counts = defaultdict(int)
    for group_key, frames in grouped_dict.items():
        if frames and frames[0]['label']:
            label_counts[frames[0]['label']] += 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} groups")
    
    # Transform for classifier input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Class mappings
    class_to_idx = {'boiling': 0, 'normal': 1, 'on_fire': 2, 'smoking': 3}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Process each group
    results_data = []
    correct_count = 0
    total_count = 0
    
    for group_idx, (group_key, frames_list) in enumerate(sorted(grouped_dict.items()), 1):
        # Sort frames by frame_id
        frames_list.sort(key=lambda x: x['frame_id'])
        frame_paths = [f['path'] for f in frames_list]
        true_label = frames_list[0]['label'] if frames_list else None
        
        print(f"\n[{group_idx}/{len(grouped_dict)}] Processing group: {group_key}")
        print(f"  Frames: {len(frame_paths)}")
        
        # Handle different sequence lengths
        if len(frame_paths) < 3:
            print(f"  ⚠ Warning: Only {len(frame_paths)} frames, triplicating last frame")
            while len(frame_paths) < 3:
                frame_paths.append(frame_paths[-1])
        elif len(frame_paths) > 3:
            print(f"  ⚠ Warning: {len(frame_paths)} frames, using first 3")
            frame_paths = frame_paths[:3]
        
        # Load ORIGINAL images (not cropped)
        original_frames = []
        frame_valid = True
        for fpath in frame_paths[:3]:
            img = cv2.imread(str(fpath))
            if img is None:
                print(f"  ⚠ Failed to load {fpath.name}")
                frame_valid = False
                break
            original_frames.append(img)
        
        if not frame_valid:
            continue
        
        # Detect and collect bboxes from all 3 frames
        all_bboxes = []
        cropped_frames = []
        
        for idx, orig_img in enumerate(original_frames):
            detections = detector.get_detections_with_crops(orig_img, apply_padding=True)
            
            if detections:
                # Take the first (most confident) detection
                detection = detections[0]
                all_bboxes.append(detection['bbox'])
                cropped_frames.append(detection['crop'])
            else:
                print(f"  ⚠ No detection in frame {idx+1}, using full image")
                # Use full image as fallback
                all_bboxes.append((0, 0, orig_img.shape[1], orig_img.shape[0]))
                cropped_frames.append(orig_img)
        
        if len(cropped_frames) < 3:
            print(f"  ⚠ Failed to get 3 crops, skipping")
            continue
        
        # Merge bboxes across the 3 frames for consistent region
        merged_bbox = merge_bboxes_with_outlier_removal(all_bboxes)
        print(f"  Merged bbox: {merged_bbox}")
        
        # Create temporal input from 3 cropped regions
        crop_tensors = []
        for crop_img in cropped_frames:
            crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            crop_tensor = transform(crop_pil)
            crop_tensors.append(crop_tensor)
        
        # Concatenate 3 frames into 9-channel input
        temporal_input = torch.cat(crop_tensors, dim=0)  # Shape: [9, 224, 224]
        temporal_input = temporal_input.unsqueeze(0).to(device)  # Shape: [1, 9, 224, 224]
        
        # Predict
        with torch.no_grad():
            outputs = model(temporal_input)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100
        
        predicted_class = idx_to_class[predicted_idx]
        is_correct = true_label.lower() == predicted_class.lower() if true_label else False
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Color: Green if correct, Red if wrong
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        marker = "✓" if is_correct else "✗"
        
        print(f"    Prediction: {predicted_class.upper()} ({confidence:.1f}%) {marker}")
        if true_label:
            print(f"      Ground truth: {true_label.upper()}")
        
        # Draw on ORIGINAL first frame
        img_display = original_frames[0].copy()
        
        # Draw merged bounding box on original image
        x1, y1, x2, y2 = merged_bbox
        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
        
        # Add prediction text
        label_text = f"{predicted_class.upper()} {confidence:.1f}%"
        # Add background for better readability
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(img_display, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(img_display, label_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Add ground truth if available
        if true_label:
            gt_text = f"GT: {true_label.upper()}"
            cv2.putText(img_display, gt_text, (x1, y2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        output_path = output_dir / f"{group_key}_marked_original.jpg"
        cv2.imwrite(str(output_path), img_display)
        print(f"  ✓ Saved marked image: {output_path.name}")
        
        # Store results
        results_data.append({
            "group_key": group_key,
            "ground_truth": true_label,
            "frame_files": [p.name for p in frame_paths[:3]],
            "merged_bbox": merged_bbox,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "correct": is_correct,
            "all_probabilities": {
                "boiling": float(probabilities[0]),
                "normal": float(probabilities[1]),
                "on_fire": float(probabilities[2]),
                "smoking": float(probabilities[3])
            }
        })
    
    # Calculate accuracy
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # Save results
    results_file = output_dir / "verification_results_on_originals.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model_type": "temporal_grouped_on_originals",
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "model_path": model_path,
            "yolo_model_path": yolo_model_path,
            "total_groups": len(grouped_dict),
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy,
            "results": results_data
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Verification Complete!")
    print(f"  Processed: {total_count} temporal groups")
    print(f"  Correct: {correct_count}/{total_count}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Marked images saved to: {output_dir}")
    print(f"  Results saved to: {results_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    # Configuration with command line support
    INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "./verification_pics"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./verification_results_on_originals"
    MODEL_PATH = "./yolo_training/pan_pot_classifier_temporal.pth"
    YOLO_MODEL_PATH = "./yolo_training/kitchenware_detector2/weights/best.pt"
    BACKBONE = "mobilenet_v2"
    
    print("="*60)
    print("  Temporal Verification on ORIGINAL Images")
    print("  Draws bounding boxes on full-resolution originals")
    print("="*60)
    print(f"\nInput: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Temporal Model: {MODEL_PATH}")
    print(f"YOLO Detector: {YOLO_MODEL_PATH}")
    print()
    
    run_verification_on_originals(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        yolo_model_path=YOLO_MODEL_PATH,
        backbone=BACKBONE,
        conf_threshold=0.3
    )
