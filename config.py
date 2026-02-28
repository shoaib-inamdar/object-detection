"""
Configuration settings for the Real-Time Object Detection system.
Adjust these parameters to fine-tune detection performance.
"""

import os

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# YOLOv4-tiny (fast, good accuracy)
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
YOLO_CONFIG = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
COCO_NAMES = os.path.join(MODEL_DIR, "coco.names")

# ──────────────────────────────────────────────
#  Detection Parameters
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25      # Minimum confidence to accept a detection
NMS_THRESHOLD = 0.45             # Non-Maximum Suppression threshold
INPUT_SIZE = (416, 416)          # Network input size (width, height)

# ──────────────────────────────────────────────
#  Camera / Video
# ──────────────────────────────────────────────
CAMERA_INDEX = 0                 # Default webcam
VIDEO_WIDTH = 1280               # Requested capture width
VIDEO_HEIGHT = 720               # Requested capture height

# ──────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────
WINDOW_NAME = "Real-Time Object Detection | YOLOv4-tiny"
SHOW_FPS = True
SHOW_CONFIDENCE = True
SHOW_CLASS_LABEL = True

# ──────────────────────────────────────────────
#  Voice Announcements (pyttsx3)
# ──────────────────────────────────────────────
VOICE_ENABLED = True             # Set False to disable voice entirely
VOICE_CONFIDENCE_THRESHOLD = 0.60  # Only speak if confidence > 60%
VOICE_COOLDOWN = 3.0             # Seconds before repeating the same label
VOICE_CENTER_FRACTION = 0.5      # Centre region: middle 50% of the frame
VOICE_RATE = 175                 # Speech speed (words per minute)
VOICE_VOLUME = 1.0               # Volume 0.0 – 1.0

# ──────────────────────────────────────────────
#  Backend (OpenCV DNN)
# ──────────────────────────────────────────────
# Set to True to use CUDA GPU acceleration (requires OpenCV built with CUDA)
USE_GPU = False
