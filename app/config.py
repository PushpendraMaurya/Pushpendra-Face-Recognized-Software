import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
DB_PATH = DATA_DIR / "app.db"
FRONTEND_DIR = BASE_DIR / "frontend"

YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Detector parameters
DETECTOR_SCORE_THRESH = 0.60
DETECTOR_NMS_THRESH = 0.30
DETECTOR_TOP_K = 5000

# Recognition parameters
# Recommended threshold for SFace cosine similarity is 0.363
RECOGNITION_COSINE_THRESH = 0.363

# Ensure storage directories exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
