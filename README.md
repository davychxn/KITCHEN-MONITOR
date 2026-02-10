# KITCHEN-MONITOR

[English](./README.md) | [中文](./README_CN.md)

## Project Overview

Kitchen Monitor is a computer vision project for kitchen cookware detection and cooking state classification, featuring YOLO-based object detection, temporal classification models, and real-time monitoring with audio alerts on Raspberry Pi 5.

## Project Structure

### Documentation
- **`README.md`** - Project documentation (English)
- **`README_CN.md`** - Project documentation (Chinese)
- **`FINE_TUNING_JOURNEY.md`** - Model fine-tuning process documentation (English)
- **`FINE_TUNING_JOURNEY_CN.md`** - Model fine-tuning process documentation (Chinese)

### Models
- **`yolo_training/kitchenware_detector2/weights/best.pt`** - **Stage 1**: YOLOv8n object detection model for detecting pans and pots
  - 497 training images, 2 classes (cooking-pot, frying-pan)
  - Performance: 99.9% precision, 100% recall
- **`yolo_training/pan_pot_classifier_temporal.pth`** - **Stage 2**: MobileNet v2 temporal classification model for cooking state detection
  - 143 temporal groups (3 frames each), 4 classes (boiling, normal, on_fire, smoking)
  - Performance: 93.10% validation accuracy, 92.9% test accuracy

### Scripts
- **`start_monitoring.py`** - Live monitoring script for Raspberry Pi 5 with camera and audio alerts
- **`test_audio.py`** - Audio playback test script (plays all MP3s in assets/sound/ continuously)
- **`train_classifier_temporal.py`** - Temporal classifier training script
- **`verify_yolo.py`** - YOLO model verification script
- **`verify_temporal_on_originals.py`** - Temporal model verification on original images
- **`resize_and_copy_images.py`** - Image preprocessing utility

### Directories

#### `modules/`
Core Python modules for the monitoring system.
- **`yolo_detector.py`** - YOLO-based kitchenware object detector
- **`dataset.py`** - Dataset loading and preprocessing
- **`__init__.py`** - Package initialization

#### `assets/sound/`
Audio alert files for the live monitoring system.
- **`kitchenware_0.mp3` ~ `kitchenware_6.mp3`** - Kitchenware count announcements (0-6 items)
- **`kitchenware_x.mp3`** - Kitchenware count announcement (more than 6 items)
- **`statues_normal.mp3`** - Normal cooking state alert
- **`statues_boiling.mp3`** - Boiling state alert
- **`statues_smoking.mp3`** - Smoking state alert
- **`statues_on-fire.mp3`** - Fire state alert

#### `yolo_training/`
YOLO model training results and artifacts.
- **`kitchenware_detector/`** - First training run results
- **`kitchenware_detector2/`** - Second training run (improved model with more data)

#### `verification_results_on_originals/`
Model verification outputs on original images.
- Marked/annotated images showing detection results
- `verification_results_on_originals.json` - Verification metrics and results

## Key Features

- **Real-time Monitoring** - Live camera feed processing on Raspberry Pi 5
- **YOLO Object Detection** - Real-time kitchenware detection using fine-tuned YOLOv8n
- **Temporal Classification** - 3-frame temporal model for cooking state detection
- **Audio Alerts** - Voice announcements via Bluetooth speaker for detected states
- **Multi-class Detection** - Detects kitchenware count and classifies cooking states (boiling, normal, on_fire, smoking)
- **Edge Deployment** - Optimized for Raspberry Pi 5 with MobileNet v2 backbone (3.5M parameters)

## Model Performance

### Stage 1: YOLO Detector (Object Detection)
- Model: `yolo_training/kitchenware_detector2/weights/best.pt`
- Classes: cooking-pot, frying-pan
- Performance: 99.9% precision, 100% recall

### Stage 2: Temporal Classifier (State Classification)
- Model: `yolo_training/pan_pot_classifier_temporal.pth`
- Classes: boiling, normal, on_fire, smoking
- Performance: 93.10% validation accuracy, 92.9% test accuracy

## Usage

### Live Monitoring (Raspberry Pi 5)

```bash
python start_monitoring.py
```

This starts the live monitoring loop:
1. Captures 3 consecutive frames from the Pi Camera at 30 FPS
2. Resizes frames to max 512x512 pixels
3. Runs YOLO detection on each frame
4. Feeds cropped regions to the temporal classifier
5. Plays audio alerts via speaker (kitchenware count and cooking state)

### Test Audio Playback

```bash
python test_audio.py
```

Plays all MP3 files in `assets/sound/` continuously. Useful for verifying audio output is working.

### Verify YOLO Model
```bash
python verify_yolo.py
```

### Verify Temporal Model
```bash
python verify_temporal_on_originals.py
```

## Requirements

### PC (Windows/Linux x86_64)

```bash
# Optional: create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Raspberry Pi 5 (ARM64)

```bash
# Create venv with system site-packages (required for Picamera2 and libcamera)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install PyTorch wheels (ARM64)
pip3 install https://github.com/KumaTea/pytorch-aarch64/releases/download/v2.3.0/torch-2.3.0a0+gitc8f7e6d-cp311-cp311-linux_aarch64.whl
pip3 install https://github.com/KumaTea/pytorch-aarch64/releases/download/v2.3.0/torchvision-0.18.0a0+gitc8f7e6d-cp311-cp311-linux_aarch64.whl

# OpenCV from apt (more reliable on Pi)
sudo apt update
sudo apt install -y python3-opencv

# Install Picamera2 (for Pi Camera)
sudo apt install -y python3-picamera2 python3-libcamera

# Install audio player (for audio alerts)
sudo apt install -y mpg123
# or
sudo apt install -y ffmpeg

# Install the rest
pip3 install -r requirements.txt
```

### Raspberry Pi 5 Audio Setup (Bluetooth Speaker)

```bash
# Install PipeWire Bluetooth audio support
sudo apt install -y libspa-0.2-bluetooth pipewire-audio

# Restart PipeWire
systemctl --user restart pipewire pipewire-pulse

# Connect Bluetooth speaker
bluetoothctl connect <MAC_ADDRESS>

# Verify Bluetooth sink appears
pactl list sinks short

# Set Bluetooth speaker as default output
pactl set-default-sink bluez_output.<MAC_ADDRESS>.1

# Test audio
mpg123 ./assets/sound/kitchenware_1.mp3
```

## Documentation

For detailed information about the fine-tuning process:
- English: [FINE_TUNING_JOURNEY.md](FINE_TUNING_JOURNEY.md)
- Chinese: [FINE_TUNING_JOURNEY_CN.md](FINE_TUNING_JOURNEY_CN.md)

## Repository Contents

This repository contains:
- Trained models (PyTorch weights)
- Training results and metrics
- Live monitoring script for Raspberry Pi 5
- Audio alert assets
- Verification scripts and results
- Documentation (English & Chinese)

**Note**: Training data, annotation tools (X-AnyLabeling, ChatRex), and utility scripts are not included in version control.

## License

See individual component licenses for details.
