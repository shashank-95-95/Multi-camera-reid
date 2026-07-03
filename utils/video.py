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
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


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
        """Initialise the video processor.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory for output video and JSON.  Created
                automatically if it doesn't exist.
            camera_id: Optional camera identifier.  When set, every
                tracking-result JSON entry includes a ``camera_id``
                field (required for Phase 2 multi-camera output).

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If the video cannot be opened.
        """
        # --- Validate input path ---
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Video file not found: '{video_path}'"
            )

        self.video_path = video_path
        self.output_dir = output_dir
        self._camera_id = camera_id
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # --- Open video capture ---
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video: '{self.video_path}'"
            )

        # --- Extract video metadata ---
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # --- Initialise video writer ---
        output_video_path = os.path.join(self.output_dir, "tracked_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_video_path, fourcc, self.fps, (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Failed to initialise video writer at "
                f"'{output_video_path}'"
            )

        # Storage for tracking results (written to JSON at the end)
        self._tracking_results: List[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Return the total number of frames in the source video."""
        return self.total_frames

    # ------------------------------------------------------------------
    # Frame I/O
    # ------------------------------------------------------------------

    def read_frame(self):
        """Read the next frame from the video.

        Returns:
            A tuple ``(success, frame)`` where *success* is a bool and
            *frame* is a NumPy BGR array (or ``None`` on failure).
        """
        ret, frame = self.cap.read()
        return ret, frame

    def write_frame(self, frame: np.ndarray) -> None:
        """Write an annotated frame to the output video.

        Args:
            frame: The annotated BGR frame.
        """
        self.writer.write(frame)

    def display_frame(
        self, frame: np.ndarray, window_name: str = "Person Tracking"
    ) -> bool:
        """Show the frame in a window and handle quit keys.

        Args:
            frame: BGR image to display.
            window_name: Title of the display window.

        Returns:
            ``True`` if the user pressed **q** or **Esc** to quit,
            ``False`` otherwise.
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)  # q or Esc

    # ------------------------------------------------------------------
    # Tracking-result persistence
    # ------------------------------------------------------------------

    def add_tracking_result(
        self,
        frame_number: int,
        track_id,
        bbox: List[int],
        confidence: float,
        global_id: Optional[int] = None,
        reid_similarity: Optional[float] = None,
    ) -> None:
        """Append a single tracking record.

        Args:
            frame_number: 0-indexed frame number.
            track_id: Unique ID assigned by the tracker.
            bbox: ``[x1, y1, x2, y2]`` bounding box.
            confidence: Detection confidence score.
            global_id: Optional global identity ID assigned by the
                ReID engine (Phase 3).
            reid_similarity: Optional cosine similarity score from
                the ReID match (Phase 3).
        """
        timestamp = round(frame_number / self.fps, 4)
        entry: dict = {
            "frame": frame_number,
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        # Prepend camera_id when operating in multi-camera mode
        if self._camera_id is not None:
            entry = {"camera_id": self._camera_id, **entry}
        # Append ReID fields when available (Phase 3)
        if global_id is not None:
            entry["global_id"] = global_id
        if reid_similarity is not None:
            entry["reid_similarity"] = round(reid_similarity, 4)
        self._tracking_results.append(entry)

    def save_tracking_results(self) -> str:
        """Write all accumulated tracking results to a JSON file.

        Returns:
            The absolute path to the saved JSON file.
        """
        output_path = os.path.join(
            self.output_dir, "tracking_results.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._tracking_results, f, indent=2)
        return output_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release all OpenCV resources."""
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
