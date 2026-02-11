#!/usr/bin/env python3
"""
Live monitoring loop for RPi camera input.
Reads 3 continuous frames, runs detector + temporal classifier, and plays voice prompts.
"""

from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import queue
import shutil
import subprocess
import threading
import time
import sys
from enum import Enum
from pathlib import Path
from picamera2 import Picamera2
from PIL import Image

import torch
import torchvision.transforms as transforms

from modules.yolo_detector import KitchenObjectDetector
from train_classifier_temporal import TemporalStateClassifier

class WareDetect(Enum):
    WARE_0 = 0
    WARE_1 = 1
    WARE_2 = 2
    WARE_3 = 3
    WARE_4 = 4
    WARE_5 = 5
    WARE_6 = 6
    WARE_X = 7
    
class WareSatus(Enum):
    NORMAL = 1
    BOILING = 2
    SMOKING = 3
    ON_FIRE = 4

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


class AudioPlayer:
    _DETECT_TO_FILE = {
        WareDetect.WARE_0: "kitchenware_0.mp3",
        WareDetect.WARE_1: "kitchenware_1.mp3",
        WareDetect.WARE_2: "kitchenware_2.mp3",
        WareDetect.WARE_3: "kitchenware_3.mp3",
        WareDetect.WARE_4: "kitchenware_4.mp3",
        WareDetect.WARE_5: "kitchenware_5.mp3",
        WareDetect.WARE_6: "kitchenware_6.mp3",
        WareDetect.WARE_X: "kitchenware_x.mp3",
    }

    _STATUS_TO_FILE = {
        WareSatus.NORMAL: "statues_normal.mp3",
        WareSatus.BOILING: "statues_boiling.mp3",
        WareSatus.SMOKING: "statues_smoking.mp3",
        WareSatus.ON_FIRE: "statues_on-fire.mp3",
    }
    
    _WARE_DETECT_LAST = WareDetect.WARE_0
    _WARE_DETECT_COUNT = 0
    _WARE_STATUS_LAST = WareSatus.NORMAL
    _WARE_STATUS_COUNT = 0
    
    _WARE_INFO_THRESHOLD = 0

    def __init__(self, sound_dir: Path):
        self._sound_dir = sound_dir
        self._queue = queue.Queue(maxsize=1)
        self._backend = self._select_backend()
        self._last_enqueued = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _select_backend(self):
        try:
            from playsound import playsound  # type: ignore
            return ("playsound", playsound)
        except Exception:
            if shutil.which("ffplay"):
                return ("ffplay", None)
            if shutil.which("mpg123"):
                return ("mpg123", None)
        return (None, None)

    def _resolve_path(self, result: WareDetect | WareSatus) -> Path | None:
        if isinstance(result, WareDetect):
            filename = self._DETECT_TO_FILE.get(result)
            #if result == self._WARE_DETECT_LAST:
            #    self._WARE_DETECT_COUNT += 1
            #    if self._WARE_DETECT_COUNT >= self._WARE_INFO_THRESHOLD:
            #        filename = self._DETECT_TO_FILE.get(result)
            #    else:
            #        return None
            #else:
            #    self._WARE_DETECT_LAST = result
            #    self._WARE_DETECT_COUNT = 0
            #    return None
        elif isinstance(result, WareSatus):
            filename = self._STATUS_TO_FILE.get(result)
            #if result == self._WARE_STATUS_LAST:
            #    self._WARE_STATUS_COUNT += 1
            #    if self._WARE_STATUS_COUNT >= self._WARE_INFO_THRESHOLD:
            #        filename = self._STATUS_TO_FILE.get(result)
            #    else:
            #        return None
            #else:
            #    self._WARE_STATUS_LAST = result
            #    self._WARE_STATUS_COUNT = 0
            #    return None
        else:
            #self._WARE_DETECT_LAST = WareDetect.WARE_0
            #self._WARE_DETECT_COUNT = 0
            #self._WARE_STATUS_LAST = WareSatus.NORMAL
            #self._WARE_STATUS_COUNT = 0
            return None
        if filename is None:
            return None
        return self._sound_dir / filename

    def _play(self, result: WareDetect | WareSatus):
        audio_path = self._resolve_path(result)
        if audio_path is None or not audio_path.exists():
            return
        print(f"voice path: {audio_path} to play")
        backend, handler = self._backend
        if backend is None:
            return
        if backend == "playsound":
            handler(str(audio_path))
            return
        if backend == "ffplay":
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
                check=False,
            )
            return
        if backend == "mpg123":
            subprocess.run(["mpg123", "-q", str(audio_path)], check=False)
            return

    def enqueue(self, result: WareDetect | WareSatus):
        if self._backend[0] is None:
            return
        if self._last_enqueued == result:
            return
        self._last_enqueued = result
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(result)

    def _worker(self):
        while True:
            result = self._queue.get()
            try:
                self._play(result)
            finally:
                self._queue.task_done()


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def resize_max(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    scale = min(max_size / width, max_size / height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def load_temporal_classifier(model_path, backbone, device):
    model = TemporalStateClassifier(backbone=backbone)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def read_continuous_frames(
    capture,
    num_frames,
    fps,
    retries=8,
    retry_delay=0.02,
    rotate_180=True,
    max_size=None,
):
    frames = []
    frame_delay = 1.0 / fps if fps > 0 else 0.0

    # Check if using Picamera2 or OpenCV
    is_picamera2 = hasattr(capture, 'capture_array')

    for _ in range(num_frames):
        frame = None

        start_time = time.monotonic()
        for _ in range(retries):
            if is_picamera2:
                # Picamera2: capture from main stream
                frame = capture.capture_array("main")
            else:
                # OpenCV VideoCapture
                ret, frame = capture.read()
                if not ret:
                    frame = None

            if frame is not None:
                print("frame read successfully")
                # Picamera2 returns RGB, OpenCV returns BGR
                if is_picamera2:
                    image = Image.fromarray(frame)
                else:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if rotate_180:
                    image = image.transpose(Image.ROTATE_180)
                if max_size is not None:
                    image = resize_max(image, max_size)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                print("frame read and prepare successfully")
                break
            time.sleep(retry_delay)

        if frame is None:
            print("frame read None")
            return []

        frames.append(frame)
        print("frame append successfully")

        elapsed = time.monotonic() - start_time
        remaining = frame_delay - elapsed
        if remaining > 0:
            print("frame read interval sleeping")
            time.sleep(remaining)

    print("frame read done")
    return frames


def open_camera(camera_index, capture_fps):
    if sys.platform.startswith("linux"):
        try:
            if Picamera2 is None:
                raise ImportError("Picamera2 not available")
            picam2 = Picamera2()
            # Use video configuration for continuous capture, not still
            config = picam2.create_video_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                controls={"FrameRate": capture_fps}
            )
            picam2.configure(config)
            picam2.start()
            print(f"picamera2 opened successfully at {capture_fps} FPS")
            return picam2
        except (ImportError, RuntimeError) as e:
            print(f"Picamera2 initialization failed: {e}, falling back to OpenCV")
            # Fall through to OpenCV fallback

    # OpenCV fallback (for non-RPi Linux or if Picamera2 fails)
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        print("cv2 camera opened failed")
        return None

    print("cv2 camera opened successfully")
    capture.set(cv2.CAP_PROP_FPS, float(capture_fps))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Flush initial frames
    for _ in range(5):
        capture.read()
        time.sleep(0.03)

    print("cv2 camera opened return successfully")
    return capture


def build_temporal_input(frames, detector, transform):
    all_bboxes = []
    cropped_frames = []
    detection_counts = []

    for frame in frames:
        detections = detector.get_detections_with_crops(frame, apply_padding=True)
        detection_counts.append(len(detections))
        if detections:
            detection = detections[0]
            all_bboxes.append(detection['bbox'])
            cropped_frames.append(detection['crop'])
        else:
            all_bboxes.append((0, 0, frame.shape[1], frame.shape[0]))
            cropped_frames.append(frame)

    merged_bbox = merge_bboxes_with_outlier_removal(all_bboxes)

    crop_tensors = []
    for crop_img in cropped_frames[:3]:
        crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        crop_tensors.append(transform(crop_pil))

    temporal_input = torch.cat(crop_tensors, dim=0).unsqueeze(0)
    kitchenware_count = max(detection_counts) if detection_counts else 0

    return temporal_input, merged_bbox, kitchenware_count


def count_to_ware_detect(count: int) -> WareDetect:
    if count <= 6:
        return WareDetect(count)
    return WareDetect.WARE_X


def run_live_monitoring(
    camera_index=0,
    model_path="./yolo_training/pan_pot_classifier_temporal.pth",
    yolo_model_path="./yolo_training/kitchenware_detector2/weights/best.pt",
    backbone="mobilenet_v2",
    conf_threshold=0.3,
    loop_fps=10,
    capture_fps=30,
    frames_per_sample=3,
    sound_dir="./assets/sound",
    max_size=512,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sound_dir = Path(sound_dir)

    print("Loading models...")
    print(f"Using device: {device}")

    model = load_temporal_classifier(model_path, backbone, device)
    detector = KitchenObjectDetector(model_path=yolo_model_path, conf_threshold=conf_threshold)
    detector.load_model()

    transform = build_transform()
    idx_to_status = {0: WareSatus.BOILING, 1: WareSatus.NORMAL, 2: WareSatus.ON_FIRE, 3: WareSatus.SMOKING}

    capture = open_camera(camera_index, capture_fps)
    if capture is None:
        raise RuntimeError("Failed to open camera")

    audio_player = AudioPlayer(sound_dir)
    print(f"Audio backend selected: {audio_player._backend[0]}")
    last_kitchenware_count = None

    loop_interval = 1.0 / loop_fps if loop_fps > 0 else 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            frames = read_continuous_frames(capture, frames_per_sample, capture_fps, max_size=max_size)
            if len(frames) < frames_per_sample:
                print("{} frames read, retrying...".format(frames))
                continue

            temporal_input, _, kitchenware_count = build_temporal_input(
                frames, detector, transform
            )
            temporal_input = temporal_input.to(device)

            with torch.no_grad():
                outputs = model(temporal_input)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                predicted_status = idx_to_status[predicted_idx]

            result_to_play = None
            if last_kitchenware_count is None or kitchenware_count != last_kitchenware_count:
                result_to_play = count_to_ware_detect(kitchenware_count)
                last_kitchenware_count = kitchenware_count
            else:
                result_to_play = predicted_status

            if result_to_play is not None:
                audio_player.enqueue(result_to_play)

            elapsed = time.perf_counter() - loop_start
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if hasattr(capture, "stop"):
            capture.stop()
            print("picamera2 stopped")
        else:
            capture.release()
            print("cv2 camera released")


if __name__ == "__main__":
    MODEL_PATH = "./yolo_training/pan_pot_classifier_temporal.pth"
    YOLO_MODEL_PATH = "./yolo_training/kitchenware_detector2/weights/best.pt"
    BACKBONE = "mobilenet_v2"
    SOUND_DIR = "./assets/sound"

    run_live_monitoring(
        camera_index=0,
        model_path=MODEL_PATH,
        yolo_model_path=YOLO_MODEL_PATH,
        backbone=BACKBONE,
        conf_threshold=0.3,
        loop_fps=10,
        capture_fps=30,
        frames_per_sample=3,
        sound_dir=SOUND_DIR,
        max_size=512,
    )
