import os
import io
import csv
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from app.config import (
    BASE_DIR,
    DB_PATH,
    FACES_DIR,
    SNAPSHOTS_DIR,
    FRONTEND_DIR,
    RECOGNITION_COSINE_THRESH
)
from app.database import Database
from app.face_engine import FaceEngine
from app.schemas import RegisterRequest, RecognizeRequest, AddSampleRequest

app = FastAPI(
    title="Pushpendra Face Recognition System",
    description="High-performance Deep Learning Face Recognition Software with YuNet & SFace",
    version="1.0.0"
)

# Enable CORS for easy cross-origin development if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and face recognition engine
db = Database(str(DB_PATH))
face_engine = FaceEngine(db)

# Mount static asset directories
app.mount("/static/faces", StaticFiles(directory=str(FACES_DIR)), name="faces")
app.mount("/static/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_assets")

@app.get("/")
def serve_index():
    """Serves the intuitive web application frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))

@app.get("/api/status")
def get_system_status():
    """Returns engine health, model status, and user count."""
    users = db.get_all_users()
    return {
        "status": "online",
        "detector_model": "YuNet (ONNX)",
        "recognizer_model": "SFace (ONNX Deep Embeddings)",
        "default_threshold": RECOGNITION_COSINE_THRESH,
        "enrolled_users_count": len(users),
        "total_embeddings_loaded": len(face_engine.cached_user_ids)
    }

@app.get("/api/users")
def list_users():
    """Retrieves all registered users."""
    return {"users": db.get_all_users()}

@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    """Retrieves single user details."""
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/api/users/register")
def register_user(payload: RegisterRequest):
    """
    Registers a new person via base64 encoded photo or webcam capture.
    """
    try:
        res = face_engine.enroll_user(
            name=payload.name,
            role=payload.role or "Member",
            image_input=payload.image
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment error: {str(e)}")

@app.post("/api/users/register-file")
async def register_user_file(
    name: str = Form(...),
    role: str = Form("Member"),
    file: UploadFile = File(...)
):
    """Registers a person by uploading an image file."""
    try:
        contents = await file.read()
        res = face_engine.enroll_user(
            name=name,
            role=role,
            image_input=contents
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment error: {str(e)}")

@app.post("/api/users/add-sample")
def add_sample(payload: AddSampleRequest):
    """Adds an additional angle or sample to an existing person."""
    try:
        res = face_engine.add_sample(
            user_id=payload.user_id,
            image_input=payload.image
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample error: {str(e)}")

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    """Deletes an enrolled person from the database."""
    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    face_engine.reload_cache()
    return {"success": True, "message": f"User {user_id} deleted successfully"}

@app.post("/api/recognize")
def recognize_face(payload: RecognizeRequest):
    """
    Recognizes faces from base64 image (from live webcam feed or uploaded picture).
    """
    try:
        res = face_engine.process_frame(
            image_input=payload.image,
            threshold=payload.threshold,
            save_log=payload.save_log or False
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

@app.post("/api/recognize-file")
async def recognize_face_file(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    save_log: Optional[bool] = Form(False)
):
    """Recognizes faces from uploaded image file."""
    try:
        contents = await file.read()
        res = face_engine.process_frame(
            image_input=contents,
            threshold=threshold,
            save_log=save_log
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

@app.get("/api/logs")
def get_logs(limit: int = 100):
    """Returns the most recent recognition history and attendance logs."""
    return {"logs": db.get_logs(limit=limit)}

@app.delete("/api/logs")
def clear_logs():
    """Clears the recognition history logs."""
    db.clear_logs()
    return {"success": True, "message": "Recognition history cleared"}

@app.get("/api/logs/export")
def export_logs_csv():
    """Exports all recognition logs as a downloadable CSV file."""
    logs = db.get_logs(limit=10000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Log ID", "User ID", "Person Name", "Confidence (%)", "Cosine Similarity", "Timestamp", "Snapshot Path"])

    for item in logs:
        writer.writerow([
            item.get("id"),
            item.get("user_id") or "N/A",
            item.get("user_name"),
            item.get("confidence"),
            item.get("similarity_score"),
            item.get("created_at"),
            item.get("snapshot_path") or "N/A"
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=face_recognition_logs.csv"}
    )
