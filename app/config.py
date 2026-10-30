import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Detect Vercel or serverless environment
IS_VERCEL = bool(os.environ.get("VERCEL"))

# Paths: Use /tmp for writable state in serverless environments
if IS_VERCEL:
    DATA_DIR = Path("/tmp/data")
    # If models exist in repo use them, otherwise download to /tmp/models
    if (BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx").exists():
        MODELS_DIR = BASE_DIR / "models"
    else:
        MODELS_DIR = Path("/tmp/models")
else:
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"

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
