"""
Train a temporal-aware classifier for pan/pot states
Uses sequential frames from video to capture motion patterns
Format: object_state_groupId_frameId.jpg (e.g., cooking-pot_boiling_1_1.jpg)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import re
from collections import defaultdict
from modules.yolo_detector import KitchenObjectDetector
from modules.dataset import (
    TemporalSequenceDataset,
    load_temporal_dataset,
    get_standard_transforms
)


class TemporalPanPotDataset(Dataset):
    """Dataset for pan/pot state classification using temporal sequences"""
    
    def __init__(self, sequence_groups, labels, transform=None, augment=False, use_temporal=True):
        """
        Args:
            sequence_groups: List of image path sequences (each group is a list of paths)
            labels: List of labels (one per group)
            transform: Transform to apply to each frame
            augment: Whether to apply augmentation
            use_temporal: If True, use all frames; if False, use only first frame
        """
        self.sequence_groups = sequence_groups
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.use_temporal = use_temporal
        
        # Augmentation transforms (applied to each frame)
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
    
    def __len__(self):
        return len(self.sequence_groups)
    
    def __getitem__(self, idx):
        image_paths = self.sequence_groups[idx]
        label = self.labels[idx]
        
        # Load all frames in the sequence
        frames = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            
            # Apply augmentation if enabled (same for all frames in sequence)
            if self.augment and self.aug_transform:
                image = self.aug_transform(image)
            
            # Apply main transform
            if self.transform:
                image = self.transform(image)
            
            frames.append(image)
        
        if self.use_temporal:
            # Pad or repeat frames to get exactly 3 frames
            while len(frames) < 3:
                # Repeat last frame if we have fewer than 3
                frames.append(frames[-1].clone())
            
            # Take only first 3 frames if we have more
            frames = frames[:3]
            
            # Stack frames along channel dimension (C*T, H, W)
            stacked = torch.cat(frames, dim=0)
            return stacked, label
        else:
            # Use only first frame
            return frames[0], label


class TemporalStateClassifier(nn.Module):
    """
    Temporal-aware classifier for pan/pot states
    Processes multiple frames to capture motion patterns
    """
    
    def __init__(self, num_classes=4, backbone='mobilenet_v2', pretrained=True, 
                 num_frames=3, use_temporal=True):
        super(TemporalStateClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_frames = num_frames
        self.use_temporal = use_temporal
        
        # Load backbone
        if backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            num_features = base_model.classifier[1].in_features
            
            # Modify first conv layer to accept multiple frames
            if use_temporal and num_frames > 1:
                # Original conv: (3, 32, 3x3)
                # New conv: (3*num_frames, 32, 3x3)
                original_conv = base_model.features[0][0]
                self.backbone = base_model
                self.backbone.features[0][0] = nn.Conv2d(
                    3 * num_frames, 32, 
                    kernel_size=3, stride=2, padding=1, bias=False
                )
                # Initialize new weights (average over temporal dimension)
                with torch.no_grad():
                    weight = original_conv.weight.repeat(1, num_frames, 1, 1) / num_frames
                    self.backbone.features[0][0].weight = nn.Parameter(weight)
            else:
                self.backbone = base_model
            
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            num_features = base_model.fc.in_features
            
            if use_temporal and num_frames > 1:
                original_conv = base_model.conv1
                self.backbone = base_model
                self.backbone.conv1 = nn.Conv2d(
                    3 * num_frames, 64,
                    kernel_size=7, stride=2, padding=3, bias=False
                )
                with torch.no_grad():
                    weight = original_conv.weight.repeat(1, num_frames, 1, 1) / num_frames
                    self.backbone.conv1.weight = nn.Parameter(weight)
            else:
                self.backbone = base_model
            
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Enhanced classifier with temporal awareness
        if use_temporal and num_frames > 1:
            # Larger capacity for temporal features
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            # Standard classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class TemporalTrainer:
    """Trainer for temporal-aware state classifier"""
    
    def __init__(self, num_classes=4, backbone='mobilenet_v2', device=None, 
                 yolo_model='yolov8n.pt', num_frames=3, use_temporal=True):
        self.num_classes = num_classes
        self.backbone = backbone
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frames = num_frames
        self.use_temporal = use_temporal
        
        # Class names
        self.class_names = ['boiling', 'normal', 'on_fire', 'smoking']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # YOLO detector for cropping (using reusable module)
        self.detector = KitchenObjectDetector(model_path=yolo_model)
        self.detector.load_model()
        print(f"Loaded YOLO model: {yolo_model}")
        
        # Model
        self.model = None
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Temporal mode: {'ENABLED' if use_temporal else 'DISABLED'} ({num_frames} frames per sequence)")
    
    def parse_filename(self, filename):
        """
        Parse filename to extract label and group info
        Format 1: object_state_groupId_frameId.jpg (e.g., cooking-pot_boiling_1_1.jpg)
        Format 2: object_state_*.jpg (e.g., cooking-pot_normal_Clipboard_*.jpg)
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
    
    def crop_with_yolo(self, image_path: str) -> Optional[np.ndarray]:
        """Detect and crop pan/pot using YOLO - uses reusable detector module"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        detections = self.detector.get_detections_with_crops(img)
        
        if len(detections) > 0:
            # Return first detection's crop
            return detections[0]['crop']
        
        return None
    
    def load_dataset(self, data_dir, test_size=0.2, use_yolo_crops=True):
        """Load dataset and organize into temporal sequences - uses reusable module"""
        # Use the reusable dataset loading module
        detector = self.detector if use_yolo_crops else None
        
        final_sequences, label_strings, label_counts = load_temporal_dataset(
            data_dir=data_dir,
            detector=detector,
            num_frames=self.num_frames,
            verbose=True
        )
        
        # Convert string labels to indices
        final_labels = [self.class_to_idx[label] for label in label_strings]
        
        # Split into train and validation
        if len(final_sequences) >= 20:
            train_seqs, val_seqs, train_labels, val_labels = train_test_split(
                final_sequences, final_labels, test_size=test_size, 
                stratify=final_labels, random_state=42
            )
        else:
            print(f"\nSmall dataset ({len(final_sequences)} groups). Using all for both train and val.")
            train_seqs, val_seqs = final_sequences, final_sequences
            train_labels, val_labels = final_labels, final_labels
        
        print(f"\nTrain set: {len(train_seqs)} sequence groups")
        print(f"Val set: {len(val_seqs)} sequence groups")
        
        return train_seqs, val_seqs, train_labels, val_labels
    
    def create_dataloaders(self, train_seqs, val_seqs, train_labels, val_labels, 
                          batch_size=8, augment=True):
        """Create data loaders for temporal sequences - uses reusable module"""
        
        # Get augmentation transform if needed
        aug_transform = None
        if augment:
            aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        
        train_dataset = TemporalSequenceDataset(
            train_seqs, train_labels, 
            transform=self.train_transform,
            augment_transform=aug_transform,
            num_frames=self.num_frames,
            class_to_idx=self.class_to_idx
        )
        
        val_dataset = TemporalSequenceDataset(
            val_seqs, val_labels,
            transform=self.val_transform,
            augment_transform=None,
            num_frames=self.num_frames,
            class_to_idx=self.class_to_idx
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, drop_last=False
        )
        
        return train_loader, val_loader
    
    def train(self, data_dir, epochs=200, batch_size=8, learning_rate=0.001, 
             save_path='temporal_classifier.pth', augment=True, use_yolo_crops=True):
        """Train the temporal classifier"""
        
        # Load dataset
        train_seqs, val_seqs, train_labels, val_labels = self.load_dataset(
            data_dir, use_yolo_crops=use_yolo_crops
        )
        
        # Create data loaders
        train_loader, val_loader = self.create_dataloaders(
            train_seqs, val_seqs, train_labels, val_labels,
            batch_size=batch_size, augment=augment
        )
        
        # Create model
        self.model = TemporalStateClassifier(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=True,
            num_frames=self.num_frames,
            use_temporal=self.use_temporal
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        print(f"\nStarting training for {epochs} epochs...")
        print("="*60)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  ✓ Best model saved (Val Acc: {best_val_acc:.2f}%)")
        
        print("="*60)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Plot history
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('temporal_training_history.png', dpi=150)
        print(f"\nTraining history plot saved to: temporal_training_history.png")


def main():
    """Main training function"""
    
    # Configuration
    DATA_DIR = r'D:\delete2025\20251205_patent_abnormal\KITCHEN-ASSIST-V2\training\step1_pics'
    EPOCHS = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    BACKBONE = 'mobilenet_v2'
    USE_YOLO_CROPS = True
    YOLO_MODEL = './yolo_training/kitchenware_detector2/weights/best.pt'
    NUM_FRAMES = 3
    USE_TEMPORAL = True
    
    print("="*70)
    print("  Temporal-Aware Pan/Pot State Classifier Training")
    print("  Motion Pattern Learning from Video Sequences")
    print("="*70)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Backbone: {BACKBONE}")
    print(f"YOLO Model: {YOLO_MODEL}")
    print(f"Training config: {EPOCHS} epochs, batch size {BATCH_SIZE}, LR {LEARNING_RATE}")
    print(f"YOLO Cropping: {'ENABLED' if USE_YOLO_CROPS else 'DISABLED'}")
    print(f"Temporal Learning: {'ENABLED' if USE_TEMPORAL else 'DISABLED'} ({NUM_FRAMES} frames)")
    print("\nThis approach uses sequential frames to capture motion patterns:")
    print("  - Boiling: Bubbling motion")
    print("  - Smoking: Rising smoke patterns")
    print("  - On-fire: Flame movement")
    print("  - Normal: Static or slow changes")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = TemporalTrainer(
        num_classes=4,
        backbone=BACKBONE,
        yolo_model=YOLO_MODEL,
        num_frames=NUM_FRAMES,
        use_temporal=USE_TEMPORAL
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path='pan_pot_classifier_temporal.pth',
        augment=True,
        use_yolo_crops=USE_YOLO_CROPS
    )
    
    print("\nTraining completed!")
    print("Model saved to: pan_pot_classifier_temporal.pth")


if __name__ == "__main__":
    main()
