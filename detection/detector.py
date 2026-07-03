"""
Person Detection Module
=======================

Provides the PersonDetector class for detecting people in video frames
using YOLOv8s from Ultralytics.

This module is intentionally decoupled from tracking logic so it can be
reused independently in later phases (multi-camera, re-identification).
"""

from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


# COCO class index for "person"
_PERSON_CLASS_ID = 0


class PersonDetector:
    """Detects people in video frames using YOLOv8s.

    Attributes:
        model: The loaded YOLO model instance.
        confidence_threshold: Minimum confidence to accept a detection.
        nms_iou_threshold: IoU threshold for Non-Maximum Suppression.
        min_bbox_width: Minimum detection width in pixels.
        min_bbox_height: Minimum detection height in pixels.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.6,
        device: str = "",
        nms_iou_threshold: float = 0.45,
        min_bbox_width: int = 40,
        min_bbox_height: int = 60,
    ) -> None:
        """Initialise the detector.

        Args:
            model_path: Path to YOLO weights.  Defaults to ``yolov8s.pt``
                which offers significantly better accuracy than YOLOv8n
                for person detection in surveillance scenarios.
            confidence_threshold: Minimum confidence score for a detection
                to be considered valid.  Raised to 0.6 to suppress
                marginal false positives that create noisy tracks.
            device: Compute device (``"cpu"``, ``"cuda"``, ``"mps"``, or
                ``""`` for auto-selection).
            nms_iou_threshold: IoU threshold for Non-Maximum Suppression.
                Lower values suppress more overlapping boxes, reducing
                duplicate detections in crowded scenes.
            min_bbox_width: Minimum bounding-box width in pixels.
                Detections narrower than this are discarded to filter
                out distant background pedestrians.
            min_bbox_height: Minimum bounding-box height in pixels.
                Detections shorter than this are discarded.

        Raises:
            FileNotFoundError: If a custom weight file is specified but
                does not exist on disk.
            RuntimeError: If the model cannot be loaded.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_bbox_width = min_bbox_width
        self.min_bbox_height = min_bbox_height

        try:
            self.model = YOLO(model_path)
            # Move to the requested device (Ultralytics handles this
            # transparently when we pass `device` during inference).
            self._device = device if device else None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[List[int], float]]:
        """Run person detection on a single frame.

        Args:
            frame: BGR image as a NumPy array (H × W × 3).

        Returns:
            A list of ``([x1, y1, x2, y2], confidence)`` tuples, one
            per detected person.  Coordinates are in pixel space.
        """
        # Run inference with explicit NMS IoU threshold.
        # verbose=False suppresses per-frame console logs.
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=[_PERSON_CLASS_ID],
            device=self._device,
            verbose=False,
        )

        detections: List[Tuple[List[int], float]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # xyxy gives [x1, y1, x2, y2] as a tensor
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())

                # --- Filter out tiny detections ---
                # Small bounding boxes correspond to distant background
                # pedestrians that generate noisy, short-lived tracks.
                # Discarding them here prevents them from ever reaching
                # the tracker.
                x1, y1, x2, y2 = coords
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w < self.min_bbox_width or bbox_h < self.min_bbox_height:
                    continue

                detections.append((coords, conf))

        return detections
