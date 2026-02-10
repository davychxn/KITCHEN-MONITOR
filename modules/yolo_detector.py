"""
Step 1: YOLO Detection Module
Reusable module for detecting and cropping kitchen objects (pans/pots)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class KitchenObjectDetector:
    """YOLO-based detector for kitchen objects with cropping functionality"""
    
    def __init__(self, model_path='./yolo_training/kitchenware_detector/weights/best.pt', 
                 conf_threshold=0.3, 
                 padding_ratio=0.1):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained YOLO model weights
            conf_threshold: Minimum confidence threshold for detections
            padding_ratio: Padding ratio to add around detected bounding boxes (e.g., 0.1 = 10%)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.padding_ratio = padding_ratio
        self.model = None
        
    def load_model(self):
        """Load YOLO model"""
        if self.model is None:
            self.model = YOLO(self.model_path)
        return self.model
    
    def detect(self, image, conf_threshold=None):
        """
        Run YOLO detection on image
        
        Args:
            image: Input image (numpy array in BGR format)
            conf_threshold: Optional override for confidence threshold
            
        Returns:
            List of detection results from YOLO
        """
        if self.model is None:
            self.load_model()
        
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model(image, conf=conf, verbose=False)
        return results
    
    def get_detections_with_crops(self, image, conf_threshold=None, apply_padding=True):
        """
        Detect objects and extract cropped regions
        
        Args:
            image: Input image (numpy array in BGR format)
            conf_threshold: Optional override for confidence threshold
            apply_padding: Whether to apply padding around bounding boxes
            
        Returns:
            List of dictionaries containing:
                - bbox: [x1, y1, x2, y2] coordinates
                - bbox_padded: [x1_pad, y1_pad, x2_pad, y2_pad] with padding applied
                - confidence: YOLO detection confidence
                - crop: Cropped image region (numpy array)
        """
        results = self.detect(image, conf_threshold)
        h, w = image.shape[:2]
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence
                conf = float(box.conf[0])
                
                # Apply padding if requested
                if apply_padding:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    pad_w = int(box_w * self.padding_ratio)
                    pad_h = int(box_h * self.padding_ratio)
                    
                    x1_pad = max(0, x1 - pad_w)
                    y1_pad = max(0, y1 - pad_h)
                    x2_pad = min(w, x2 + pad_w)
                    y2_pad = min(h, y2 + pad_h)
                else:
                    x1_pad, y1_pad, x2_pad, y2_pad = x1, y1, x2, y2
                
                # Extract crop
                crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'bbox_padded': [x1_pad, y1_pad, x2_pad, y2_pad],
                    'confidence': conf,
                    'crop': crop
                })
        
        return detections
    
    def save_crops(self, image_path, output_dir, prefix='crop'):
        """
        Detect objects in image and save crops to disk
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save crops
            prefix: Prefix for saved crop filenames
            
        Returns:
            List of saved crop paths
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # Get detections with crops
        detections = self.get_detections_with_crops(image)
        
        # Save crops
        saved_paths = []
        base_name = image_path.stem
        
        for idx, det in enumerate(detections, 1):
            crop_name = f"{prefix}_{base_name}_{idx}.jpg"
            crop_path = output_dir / crop_name
            cv2.imwrite(str(crop_path), det['crop'])
            saved_paths.append(crop_path)
        
        return saved_paths
    
    def batch_process_directory(self, input_dir, output_dir, pattern='*.jpg'):
        """
        Process all images in a directory and save crops
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save crops
            pattern: Glob pattern for matching images
            
        Returns:
            Dictionary mapping input image paths to list of crop paths
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        image_files = sorted(input_dir.glob(pattern))
        
        for img_path in image_files:
            crop_paths = self.save_crops(img_path, output_dir)
            results[img_path] = crop_paths
        
        return results


def draw_detections(image, detections, labels=None, colors=None):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image (will be copied, not modified)
        detections: List of detection dictionaries (from get_detections_with_crops)
        labels: Optional list of labels to display (one per detection)
        colors: Optional list of colors (BGR tuples) for each detection
        
    Returns:
        Image with drawn bounding boxes
    """
    img_display = image.copy()
    
    if colors is None:
        colors = [(0, 255, 0)] * len(detections)  # Green by default
    
    if labels is None:
        labels = [f"Obj {i+1}" for i in range(len(detections))]
    
    for idx, det in enumerate(detections):
        bbox = det['bbox_padded']
        x1, y1, x2, y2 = bbox
        
        color = colors[idx] if idx < len(colors) else (0, 255, 0)
        label = labels[idx] if idx < len(labels) else f"Obj {idx+1}"
        
        # Draw rectangle
        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_display, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img_display, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_display
