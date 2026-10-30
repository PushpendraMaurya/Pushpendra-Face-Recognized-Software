# 👤 Pushpendra Face Recognized Software (FaceGuard AI)

A modern, high-performance, real-time Deep Learning Face Recognition application built with **Python**, **FastAPI**, **OpenCV (YuNet + SFace)**, and an intuitive **Web Frontend** designed for anyone to use with zero setup hassle.

---

## 🌟 Key Highlights & Features

- **🚀 Ultra-Fast Real-Time Recognition**:
  - Uses OpenCV Zoo's official **YuNet** (Deep CNN face detector) and **SFace** (128-dimensional deep feature extractor).
  - Processes frames in **~14 ms** (~70 FPS on CPU) with zero C++ compilation or heavy framework overhead.
  - Robust to diverse lighting conditions, angles, and facial expressions.
- **🌐 Intuitive & Responsive Web Interface**:
  - **Live Webcam Scanner**: Real-time bounding box tracking, glowing identification cards, confidence scores, and landmark tracking dots.
  - **Single-Click Person Enrollment**: Register anyone instantly using either a webcam snap (with guided face oval) or photo upload.
  - **Multi-Sample Support**: Add multiple photos or angles per person (e.g. smile, slight tilt, glasses) to enhance real-world recognition accuracy.
  - **Photo Face Scanner**: Drag-and-drop or upload static photos to detect and recognize multiple faces simultaneously.
  - **People Directory**: Manage enrolled identities, view cropped face thumbnails, and delete records with a click.
  - **Attendance & Activity History**: Records every recognition event with timestamps, confidence scores, and face snapshots.
  - **1-Click CSV Export**: Download attendance logs directly to Excel or CSV.
  - **Audio Feedback**: Pleasant synthesized chime cues on positive recognition via HTML5 Web Audio API (no external audio files needed).
  - **Theme Toggle**: Switch between Dark and Light modes.
- **🔒 Private & Local**:
  - Runs completely on your local machine. No external cloud API keys or subscriptions required.
  - Face embeddings and metadata are stored securely in a local SQLite database (`data/app.db`).

---

## 🏗️ System Architecture

```
Pushpendra-Face-Recognized-Software/
├── app/
│   ├── __init__.py
│   ├── config.py             # Model paths, thresholds, and configuration
│   ├── database.py           # SQLite database for users, embeddings & logs
│   ├── face_engine.py        # YuNet detector + SFace recognizer engine
│   ├── main.py               # FastAPI application with REST endpoints
│   └── schemas.py            # Pydantic validation schemas
├── frontend/
│   ├── index.html            # Single-page interactive frontend
│   ├── css/
│   │   └── style.css         # Responsive modern styling & dark theme
│   └── js/
│       └── app.js            # Live webcam, canvas overlay, audio & API logic
├── models/
│   ├── face_detection_yunet_2023mar.onnx    # Deep CNN face detector (227 KB)
│   └── face_recognition_sface_2021dec.onnx   # 128D deep feature extractor (37 MB)
├── data/
│   ├── app.db                # SQLite database (auto-created)
│   ├── faces/                # Enrolled face thumbnails
│   └── snapshots/            # Activity log snapshots
├── requirements.txt          # Python dependencies
├── run.sh                    # 1-click launch script
├── test_system.py            # Unit & engine tests
└── test_api.py               # API route integration tests
```

---

## ⚡ Quick Start Guide

### 1. Requirements
- Python 3.9+ (Installed on your system)
- A modern web browser (Chrome, Edge, Safari, Firefox)
- Built-in or external USB Webcam

### 2. Run with One Command
From your terminal, simply run the startup script:

```bash
cd Pushpendra-Face-Recognized-Software
./run.sh
```

The script will automatically:
1. Activate the Python virtual environment (`venv`).
2. Verify all dependencies and deep learning models.
3. Start the FastAPI server on `http://127.0.0.1:8000`.

### 3. Open in Your Browser
Visit:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🛠️ Manual Installation (Alternative)

If you prefer to run step-by-step:

```bash
# 1. Navigate to the project directory
cd Pushpendra-Face-Recognized-Software

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📖 How to Use the Software

### 1. 📷 Live Webcam Scanner
1. Click **"Start Camera"** and allow browser camera permissions.
2. Position your face in front of the camera.
3. The software will instantly draw a bounding box:
   - 🟢 **Green Box**: Recognized person with Name and Confidence % (e.g. `Pushpendra • 96.4%`).
   - 🔴 **Red Box**: `Unknown` person (not yet registered).
4. Click **"Log Recognition Event"** to manually record a snapshot to the activity log.

### 2. ➕ Enroll a New Person
1. Click the **"Enroll Person"** tab.
2. Enter the person's **Full Name** and optional **Role / Department** (e.g. `Admin`, `Staff`, `Student`).
3. Choose either:
   - **Capture with Webcam**: Click *Start Camera*, align face in the guide oval, and click *Take Photo*.
   - **Upload Image File**: Select or drag-and-drop a photo from your computer.
4. Click **"Complete Enrollment"**. The face is aligned, cropped, embedded into a 128D vector, and saved!

### 3. 👥 Managing Enrolled People
- View all enrolled users in the **"People Directory"** tab.
- Click **"+ Sample"** on any person card to add extra photos/angles (e.g., side profile, different lighting, wearing glasses) for superior recognition under all conditions.
- Click **"Delete"** to remove a person from the database.

### 4. 🖼️ Photo Face Scanner
- Click the **"Photo Scanner"** tab.
- Upload any image containing one or more people.
- The engine detects all faces simultaneously, draws bounding boxes, and displays match details with cosine similarity scores.

### 5. 📋 Activity & Attendance Logs
- Switch to the **"Activity Logs"** tab to view all past recognition events.
- Search logs by person name in real-time.
- Click **"Export CSV"** to download the complete log into an Excel-ready spreadsheet.

### 6. ⚙️ Sensitivity & Settings
- **Cosine Similarity Threshold**: Adjust between `0.25` (lenient) and `0.55` (strict). The OpenCV SFace standard default is `0.363`.
- **Audio Chime**: Toggle on/off audio feedback.
- **Theme**: Toggle Dark / Light mode in the top right.

---

## 📡 REST API Endpoints

FastAPI includes automatic interactive Swagger documentation accessible at **`http://127.0.0.1:8000/docs`**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the Web UI |
| `GET` | `/api/status` | Model status, active users, and system health |
| `GET` | `/api/users` | List all registered users |
| `POST` | `/api/users/register` | Register new user via JSON Base64 image |
| `POST` | `/api/users/register-file` | Register new user via multipart file upload |
| `POST` | `/api/users/add-sample` | Add extra face sample to existing user |
| `DELETE` | `/api/users/{id}` | Delete user and their embeddings |
| `POST` | `/api/recognize` | Detect & recognize faces from Base64 frame |
| `POST` | `/api/recognize-file` | Detect & recognize faces from uploaded file |
| `GET` | `/api/logs` | Fetch recognition activity logs |
| `DELETE` | `/api/logs` | Clear activity logs |
| `GET` | `/api/logs/export` | Download logs as CSV |

---

## 🧪 Running Tests

To verify that the engine and API are operating properly:

```bash
# Test the database, YuNet detector, SFace recognizer, and vector math
python test_system.py

# Test FastAPI route handlers in-process
python test_api.py
```

---

## 👤 Author & License
Developed by Pushpendra Kumar. Open-source and free to customize for educational, personal, or commercial attendance and security applications.
