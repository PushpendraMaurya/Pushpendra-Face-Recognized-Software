#!/usr/bin/env bash
set -e

# Change directory to the script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "   👤 Pushpendra Face Recognition Software (FaceGuard AI) "
echo "=========================================================="

# Check Python environment
if [ -d "venv" ]; then
    echo "⚡ Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "⚡ Activating parent virtual environment..."
    source ../venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Ensure model files are present
if [ ! -f "models/face_detection_yunet_2023mar.onnx" ] || [ ! -f "models/face_recognition_sface_2021dec.onnx" ]; then
    echo "📥 Downloading required deep learning models..."
    mkdir -p models
    curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
fi

echo ""
echo "🚀 Starting FastAPI Server on http://127.0.0.1:8000"
echo "🌐 Open your web browser at: http://127.0.0.1:8000"
echo "📚 API Documentation available at: http://127.0.0.1:8000/docs"
echo "=========================================================="
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
