import cv2
import numpy as np
import base64
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import (
    app,
    serve_index,
    get_system_status,
    list_users,
    recognize_face,
    get_logs,
    export_logs_csv
)
from app.schemas import RecognizeRequest

def create_synthetic_face_jpg_base64():
    img = np.full((320, 320, 3), 220, dtype=np.uint8)
    cv2.ellipse(img, (160, 160), (70, 95), 0, 0, 360, (180, 200, 230), -1)
    cv2.circle(img, (135, 135), 10, (50, 50, 50), -1)
    cv2.circle(img, (185, 135), 10, (50, 50, 50), -1)
    cv2.circle(img, (135, 135), 4, (255, 255, 255), -1)
    cv2.circle(img, (185, 135), 4, (255, 255, 255), -1)
    cv2.line(img, (160, 145), (160, 175), (80, 80, 80), 2)
    cv2.ellipse(img, (160, 195), (25, 12), 0, 0, 180, (60, 60, 160), 3)
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def main():
    print("==================================================")
    print("Testing FastAPI Route Functions In-Process")
    print("==================================================")

    # 1. Test serve_index
    index_res = serve_index()
    assert Path(index_res.path).exists()
    print("✓ serve_index() resolved index file:", index_res.path)

    # 2. Test get_system_status
    status = get_system_status()
    assert status["status"] == "online"
    assert "YuNet" in status["detector_model"]
    assert "SFace" in status["recognizer_model"]
    print("✓ get_system_status() returned online status:", status)

    # 3. Test list_users
    users_data = list_users()
    assert "users" in users_data
    print("✓ list_users() returned successfully, count:", len(users_data["users"]))

    # 4. Test recognize_face
    b64_img = create_synthetic_face_jpg_base64()
    req = RecognizeRequest(image=b64_img, save_log=False)
    recog_data = recognize_face(req)
    assert "faces_detected" in recog_data
    print(f"✓ recognize_face() processed frame in {recog_data['processing_time_ms']} ms, detected {recog_data['faces_detected']} faces")

    # 5. Test get_logs
    logs_data = get_logs()
    assert "logs" in logs_data
    print("✓ get_logs() returned successfully, count:", len(logs_data["logs"]))

    # 6. Test export_logs_csv
    csv_stream = export_logs_csv()
    assert "attachment; filename=face_recognition_logs.csv" in csv_stream.headers["Content-Disposition"]
    print("✓ export_logs_csv() generated downloadable streaming response.")

    print("\n🎉 ALL FASTAPI ROUTE HANDLERS PASSED TESTS!")

if __name__ == "__main__":
    main()
