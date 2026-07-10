"""
Video I/O Utilities
===================

Provides :class:`VideoProcessor` which handles video loading, frame
iteration, real-time display, and output writing.

This class owns no detection or tracking logic — it simply provides
frames and writes the annotated output.
"""

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# --- FIREBASE IMPORTS ---
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase only once to prevent crashes
if not firebase_admin._apps:
    # Make sure this JSON file is in your root folder!
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ------------------------


class VideoProcessor:
    """Handles video loading, display, and output writing.

    Attributes:
        video_path: Absolute path to the input video.
        output_dir: Directory where output files are saved.
        cap: The OpenCV :class:`cv2.VideoCapture` instance.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = "outputs",
        camera_id: Optional[int] = None,
    ) -> None:
        
        # --- Validate input path ---
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: '{video_path}'")

        self.video_path = video_path
        self.output_dir = output_dir
        self._camera_id = camera_id
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # --- Open video capture ---
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: '{self.video_path}'")

        # --- Extract video metadata ---
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Initialise video writer ---
        self.output_video_path = os.path.join(self.output_dir, "tracked_video.mp4")
        # avc1 is the H.264 codec that web browsers love!
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.writer = cv2.VideoWriter(
            self.output_video_path, fourcc, self.fps, (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to initialise video writer at '{self.output_video_path}'")

        # Storage for tracking results
        self._tracking_results: List[dict] = []

    @property
    def frame_count(self) -> int:
        return self.total_frames

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def write_frame(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def display_frame(self, frame: np.ndarray, window_name: str = "Person Tracking") -> bool:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27) 

    def add_tracking_result(
        self,
        frame_number: int,
        track_id,
        bbox: List[int],
        confidence: float,
        global_id: Optional[int] = None,
        reid_similarity: Optional[float] = None,
    ) -> None:
        timestamp = round(frame_number / self.fps, 4)
        entry: dict = {
            "frame": frame_number,
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        if self._camera_id is not None:
            entry = {"camera_id": self._camera_id, **entry}
        if global_id is not None:
            entry["global_id"] = global_id
        if reid_similarity is not None:
            entry["reid_similarity"] = round(reid_similarity, 4)
            
        self._tracking_results.append(entry)

    def save_tracking_results(self) -> str:
        self.output_json_path = os.path.join(self.output_dir, "tracking_results.json")
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(self._tracking_results, f, indent=2)
        return self.output_json_path

    # --- FIREBASE & AUTO-PLAY INTEGRATIONS ---
    def upload_to_firebase(self):
        """Pushes a lightweight summary to the Firestore Database (Free Tier)."""
        print(f"[Camera {self._camera_id}] Uploading tracking data to Firebase Firestore...")
        try:
            doc_ref = db.collection('processing_runs').document()
            doc_ref.set({
                'camera_id': self._camera_id,
                'total_frames_processed': self.total_frames,
                'total_detections': len(self._tracking_results),
                'timestamp': firestore.SERVER_TIMESTAMP,
                # Link directly to the saved video file for the web server
                'video_path': self.output_video_path.replace("\\", "/") 
            })
            print(f"[Camera {self._camera_id}] Run summary logged to Firestore successfully!")
        except Exception as e:
            print(f"[Camera {self._camera_id}] Failed to upload to Firebase: {e}")

    def play_output_video(self) -> None:
        """Opens the final processed video in the system's default media player."""
        print(f"[Camera {self._camera_id}] Opening processed video: {self.output_video_path}")
        try:
            if platform.system() == 'Windows':
                os.startfile(self.output_video_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', self.output_video_path))
            else:  # Linux
                subprocess.call(('xdg-open', self.output_video_path))
        except Exception as e:
            print(f"[Camera {self._camera_id}] Could not open video automatically: {e}")
    # ----------------------------------------

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()