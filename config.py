"""
Configuration file for the YOLO defect detection system.
Configure camera index, model paths, and detection parameters here.
"""

import os

# ====================
# CAMERA CONFIGURATION
# ====================
# IMPORTANT: Configure your external USB webcam index here
# Default camera indices: 0 = built-in/laptop camera, 1+ = external USB webcams
# To find your external webcam index, run: python -c "import cv2; [print(f'Index {i}: {cv2.VideoCapture(i).read()[0]}') for i in range(5)]"
CAMERA_INDEX = 1  # Change this to your external USB webcam index (typically 1, 2, 3, etc.)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Enable this to test available cameras at startup
# Set to False for faster startup (skips camera scanning)
TEST_CAMERAS_ON_STARTUP = False

# ====================
# YOLO MODEL CONFIGURATION
# ====================
# YOLOv8 model path - can be a pretrained model or custom trained model
# Options:
#   - "yolov8n.pt" (nano, fastest)
#   - "yolov8s.pt" (small)
#   - "yolov8m.pt" (medium)
#   - "yolov8l.pt" (large)
#   - "yolov8x.pt" (extra large, most accurate)
#   - Or path to your custom trained model: "path/to/best.pt"
MODEL_PATH = "yolov8n.pt"  # Will download automatically on first use

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# IoU threshold for NMS (Non-Maximum Suppression)
IOU_THRESHOLD = 0.45

# ====================
# DEFECT CLASS MAPPING
# ====================
# Map YOLO class IDs to defect types
# NOTE: For production, you'll need a custom trained YOLO model with these classes
# This is a demo mapping - adjust based on your trained model
DEFECT_CLASSES = {
    0: "crack",
    1: "scratch",
    2: "missing_label",
    3: "wrong_label",
    4: "missing_cap",
    5: "wrong_cap_color"
}

# Reverse mapping for display
DEFECT_CLASS_NAMES = {
    "crack": "Crack",
    "scratch": "Scratch/Mark",
    "missing_label": "Missing Label",
    "wrong_label": "Wrong Label",
    "missing_cap": "Missing Cap",
    "wrong_cap_color": "Wrong Cap Color"
}

# ====================
# DATABASE CONFIGURATION
# ====================
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "bottle_defect_detection"
COLLECTION_NAME = "defects"

# ====================
# DEFECT LOGGING CONFIGURATION
# ====================
# Cooldown period in seconds to avoid duplicate logging
DEFECT_LOGGING_COOLDOWN = 2.0  # Log at most once every 2 seconds per defect type

# Maximum number of defects to log per frame
MAX_DEFECTS_PER_FRAME = 5

# ====================
# APPLICATION CONFIGURATION
# ====================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# Video streaming configuration
STREAM_FPS = 15  # FPS for MJPEG stream (lower = less bandwidth)

# Application directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
