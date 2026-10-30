import os
import cv2
import numpy as np
import base64
import time
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from app.config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    DETECTOR_SCORE_THRESH,
    DETECTOR_NMS_THRESH,
    DETECTOR_TOP_K,
    RECOGNITION_COSINE_THRESH,
    FACES_DIR,
    SNAPSHOTS_DIR
)
from app.database import Database

class FaceEngine:
    def __init__(self, db: Database):
        self.db = db
        self.detector = None
        self.recognizer = None
        self.cached_user_ids = []
        self.cached_user_names = []
        self.cached_embeddings = None  # np.ndarray of shape (N, 128)
        self.load_models()
        self.reload_cache()

    def load_models(self):
        if not os.path.exists(YUNET_MODEL_PATH):
            raise FileNotFoundError(f"YuNet model not found at {YUNET_MODEL_PATH}")
        if not os.path.exists(SFACE_MODEL_PATH):
            raise FileNotFoundError(f"SFace model not found at {SFACE_MODEL_PATH}")

        # Initialize detector with default resolution (will be updated dynamically per image)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 320),
            score_threshold=DETECTOR_SCORE_THRESH,
            nms_threshold=DETECTOR_NMS_THRESH,
            top_k=DETECTOR_TOP_K
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(SFACE_MODEL_PATH),
            config=""
        )

    def reload_cache(self):
        """Loads all enrolled embeddings from the database into a contiguous NumPy matrix"""
        records = self.db.get_all_embeddings()
        if not records:
            self.cached_user_ids = []
            self.cached_user_names = []
            self.cached_embeddings = None
            return

        user_ids = []
        user_names = []
        embeddings = []

        for uid, name, emb in records:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            user_ids.append(uid)
            user_names.append(name)
            embeddings.append(emb)

        self.cached_user_ids = user_ids
        self.cached_user_names = user_names
        self.cached_embeddings = np.vstack(embeddings).astype(np.float32)

    @staticmethod
    def decode_image(image_input: Any) -> np.ndarray:
        """Decodes raw bytes, base64 string, or filepath into a BGR OpenCV image."""
        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, str):
            # Check if base64 data URI
            if image_input.startswith("data:image"):
                image_input = image_input.split(",", 1)[1]
            # Try decoding base64
            try:
                raw_bytes = base64.b64decode(image_input)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass
            # Try reading as file path
            if os.path.exists(image_input):
                img = cv2.imread(image_input)
                if img is not None:
                    return img

        if isinstance(image_input, (bytes, bytearray)):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img

        raise ValueError("Could not decode image from the provided input.")

    def detect_faces(self, image: np.ndarray, score_threshold: Optional[float] = None) -> List[Dict]:
        """Detect faces in an image using YuNet."""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        if score_threshold is not None:
            self.detector.setScoreThreshold(score_threshold)
        else:
            self.detector.setScoreThreshold(DETECTOR_SCORE_THRESH)

        ret, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp bounding box inside image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)

            landmarks = [
                {"x": int(face[4]), "y": int(face[5])},    # Right eye
                {"x": int(face[6]), "y": int(face[7])},    # Left eye
                {"x": int(face[8]), "y": int(face[9])},    # Nose tip
                {"x": int(face[10]), "y": int(face[11])},  # Right mouth
                {"x": int(face[12]), "y": int(face[13])}   # Left mouth
            ]
            score = float(face[14])

            results.append({
                "box": [x1, y1, box_w, box_h],
                "landmarks": landmarks,
                "score": round(score, 3),
                "raw_face": face
            })
        return results

    def extract_features(self, image: np.ndarray, raw_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aligns face and computes 128D normalized feature vector."""
        aligned_face = self.recognizer.alignCrop(image, raw_face)
        feature = self.recognizer.feature(aligned_face).reshape(128)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return aligned_face, feature

    @staticmethod
    def similarity_to_confidence(sim: float, threshold: float = RECOGNITION_COSINE_THRESH) -> float:
        """Converts cosine similarity (-1.0 to 1.0) to an intuitive confidence percentage (1-99%)."""
        if sim >= threshold:
            ratio = (sim - threshold) / max(0.85 - threshold, 0.01)
            conf = 70.0 + min(ratio, 1.0) * 29.5
        else:
            ratio = max(0.0, sim) / max(threshold, 0.01)
            conf = 10.0 + ratio * 59.9
        return round(float(np.clip(conf, 1.0, 99.9)), 1)

    def match_feature(self, query_feature: np.ndarray, threshold: Optional[float] = None) -> Dict:
        """Matches a single 128D query feature against all cached enrolled embeddings."""
        if threshold is None:
            threshold = RECOGNITION_COSINE_THRESH

        if self.cached_embeddings is None or len(self.cached_embeddings) == 0:
            return {
                "user_id": None,
                "name": "Unknown",
                "similarity": 0.0,
                "confidence": 0.0,
                "is_match": False
            }

        # Vectorized dot product for cosine similarity with all enrolled faces
        sims = np.dot(self.cached_embeddings, query_feature)

        # Aggregate by user_id (take maximum similarity for users with multiple samples)
        user_best = {}
        for uid, name, s in zip(self.cached_user_ids, self.cached_user_names, sims):
            score = float(s)
            if uid not in user_best or score > user_best[uid]["similarity"]:
                user_best[uid] = {
                    "user_id": uid,
                    "name": name,
                    "similarity": score
                }

        # Find best matching user
        best_match = max(user_best.values(), key=lambda item: item["similarity"])
        best_sim = best_match["similarity"]
        confidence = self.similarity_to_confidence(best_sim, threshold)

        if best_sim >= threshold:
            return {
                "user_id": best_match["user_id"],
                "name": best_match["name"],
                "similarity": round(best_sim, 4),
                "confidence": confidence,
                "is_match": True
            }
        else:
            return {
                "user_id": None,
                "name": "Unknown",
                "similarity": round(best_sim, 4),
                "confidence": confidence,
                "is_match": False
            }

    def process_frame(
        self,
        image_input: Any,
        threshold: Optional[float] = None,
        save_log: bool = False
    ) -> Dict:
        """
        Main pipeline: Takes an image, detects all faces, recognizes each one,
        and optionally records recognition events.
        """
        t0 = time.time()
        image = self.decode_image(image_input)
        detections = self.detect_faces(image)

        recognized_faces = []
        for det in detections:
            aligned_face, feature = self.extract_features(image, det["raw_face"])
            match_res = self.match_feature(feature, threshold=threshold)

            # Optional face snapshot logging for recognized persons or unknowns
            snapshot_rel = None
            if save_log:
                # Save small snapshot of the face crop
                snap_id = f"{int(time.time()*1000)}_{np.random.randint(100,999)}.jpg"
                snap_path = SNAPSHOTS_DIR / snap_id
                cv2.imwrite(str(snap_path), aligned_face)
                snapshot_rel = f"/static/snapshots/{snap_id}"

                self.db.add_log(
                    user_id=match_res["user_id"],
                    user_name=match_res["name"],
                    confidence=match_res["confidence"],
                    similarity_score=match_res["similarity"],
                    snapshot_path=snapshot_rel
                )

            face_dict = {
                "box": det["box"],
                "landmarks": det["landmarks"],
                "detection_score": det["score"],
                "user_id": match_res["user_id"],
                "name": match_res["name"],
                "confidence": match_res["confidence"],
                "similarity": match_res["similarity"],
                "is_match": match_res["is_match"],
                "snapshot_url": snapshot_rel
            }
            recognized_faces.append(face_dict)

        proc_time_ms = round((time.time() - t0) * 1000, 1)

        return {
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "faces_detected": len(recognized_faces),
            "faces": recognized_faces,
            "processing_time_ms": proc_time_ms
        }

    def enroll_user(
        self,
        name: str,
        role: str,
        image_input: Any
    ) -> Dict:
        """
        Registers a new user:
        1. Detects face in image.
        2. Validates face presence & quality.
        3. Aligns & crops face.
        4. Saves face thumbnail.
        5. Saves user and embedding to database.
        6. Updates memory cache.
        """
        image = self.decode_image(image_input)
        detections = self.detect_faces(image)

        if len(detections) == 0:
            raise ValueError("No face detected in the image. Please ensure your face is well-lit and facing the camera.")

        # Pick the largest face if multiple detected
        best_face = max(detections, key=lambda d: d["box"][2] * d["box"][3])

        aligned_face, feature = self.extract_features(image, best_face["raw_face"])

        # Insert user to database
        user_id = self.db.add_user(name=name, role=role)

        # Save thumbnail
        thumb_name = f"user_{user_id}_{int(time.time())}.jpg"
        thumb_path = FACES_DIR / thumb_name
        cv2.imwrite(str(thumb_path), aligned_face)
        rel_thumb_url = f"/static/faces/{thumb_name}"

        # Update thumbnail in DB and add sample
        self.db.update_user_thumbnail(user_id, rel_thumb_url)
        self.db.add_face_sample(user_id, feature, photo_path=rel_thumb_url)

        # Refresh in-memory cache
        self.reload_cache()

        return {
            "user_id": user_id,
            "name": name,
            "role": role,
            "thumbnail_url": rel_thumb_url,
            "message": f"Successfully enrolled {name}!"
        }

    def add_sample(self, user_id: int, image_input: Any) -> Dict:
        """Adds an additional angle or photo sample to an existing enrolled user."""
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} does not exist.")

        image = self.decode_image(image_input)
        detections = self.detect_faces(image)
        if len(detections) == 0:
            raise ValueError("No face detected in the sample image.")

        best_face = max(detections, key=lambda d: d["box"][2] * d["box"][3])
        aligned_face, feature = self.extract_features(image, best_face["raw_face"])

        sample_name = f"user_{user_id}_sample_{int(time.time()*1000)}.jpg"
        sample_path = FACES_DIR / sample_name
        cv2.imwrite(str(sample_path), aligned_face)
        rel_url = f"/static/faces/{sample_name}"

        sample_id = self.db.add_face_sample(user_id, feature, photo_path=rel_url)
        self.reload_cache()

        return {
            "sample_id": sample_id,
            "user_id": user_id,
            "name": user["name"],
            "sample_url": rel_url,
            "message": f"Added new sample for {user['name']}."
        }
