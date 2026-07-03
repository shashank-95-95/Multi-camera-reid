"""
Person Tracking Module
======================

Provides the DeepSortTracker class that wraps the ``deep-sort-realtime``
library to maintain persistent identity across frames.

This module only handles tracking logic.  Detection is performed
separately by :class:`detection.detector.PersonDetector`.
"""

from typing import List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """Maintains persistent person IDs across frames using DeepSORT.

    Each call to :meth:`update` ingests new detections and returns the
    currently-active tracks with their assigned IDs.

    Attributes:
        tracker: The underlying ``DeepSort`` instance.
    """

    def __init__(
        self,
        max_age: int = 50,
        n_init: int = 3,
        max_iou_distance: float = 0.5,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        max_time_since_update: int = 1,
        min_output_confidence: float = 0.35,
    ) -> None:
        """Initialise the tracker.

        Args:
            max_age: Maximum number of frames a track is kept alive
                without a matching detection.  Increased to 50 (~1.7 s
                at 30 FPS) so tracks survive brief occlusions instead
                of being prematurely deleted and re-assigned new IDs.
            n_init: Number of consecutive detections required before a
                track is confirmed.  Kept at 3 to suppress transient
                false-positive tracks while confirming real ones quickly.
            max_iou_distance: Maximum IoU distance for data association.
                Lowered to 0.5 for stricter spatial gating — only
                detections with significant overlap are associated,
                reducing cross-track ID switches in crowds.
            max_cosine_distance: Maximum cosine distance for the
                appearance metric.  Slightly relaxed to 0.3 to tolerate
                lighting and pose changes while remaining discriminative.
            nn_budget: Maximum size of the appearance-feature gallery
                per track.  Caps memory usage and keeps the gallery
                focused on recent appearances.
            max_time_since_update: Only emit tracks that received a
                matching detection within this many frames.  Set to 1
                to suppress "ghost" tracks coasting on prediction alone
                — the primary mechanism that eliminates ID flicker.
            min_output_confidence: Minimum detection confidence for a
                track to be included in the output.  Filters out
                low-quality detections that slip past the detector.
        """
        self._max_time_since_update = max_time_since_update
        self._min_output_confidence = min_output_confidence

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[List[int], float]],
        frame,
    ) -> List[dict]:
        """Feed new detections into the tracker and retrieve active tracks.

        Args:
            detections: List of ``([x1, y1, x2, y2], confidence)``
                tuples produced by :class:`PersonDetector`.
            frame: The current BGR frame (used internally by
                ``deep-sort-realtime`` for appearance feature extraction).

        Returns:
            A list of dicts, each containing::

                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                }
        """
        # deep-sort-realtime expects detections as a list of
        # ([left, top, width, height], confidence, class_name) tuples.
        raw_detections = []
        for (bbox, conf) in detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([x1, y1, w, h], conf, "person"))

        # Update the tracker state
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        # Collect confirmed, recently-updated tracks that meet the
        # minimum confidence bar.  This three-layer filter is the
        # primary defence against ID flicker:
        #   1. is_confirmed()       — requires n_init consecutive hits
        #   2. time_since_update    — suppresses coasting ghost tracks
        #   3. min_output_confidence — removes low-quality stragglers
        active_tracks: List[dict] = []
        for track in tracks:
            # --- Gate 1: only confirmed tracks ---
            if not track.is_confirmed():
                continue

            # --- Gate 2: only recently matched tracks ---
            # Tracks that haven't been matched to a detection for more
            # than max_time_since_update frames are coasting on Kalman
            # prediction alone.  Emitting them causes the bounding box
            # and ID to visually flicker.
            if track.time_since_update > self._max_time_since_update:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            bbox = [int(v) for v in ltrb]

            # Retrieve the original detection confidence if available;
            # fall back to 0.0 for tracks carried over without a fresh
            # detection this frame.
            confidence = 0.0
            if hasattr(track, "det_conf") and track.det_conf is not None:
                confidence = float(track.det_conf)
            elif hasattr(track, "conf") and track.conf is not None:
                confidence = float(track.conf)

            # --- Gate 3: minimum confidence ---
            if confidence < self._min_output_confidence:
                continue

            active_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": round(confidence, 4),
                }
            )

        return active_tracks

    def reset(self) -> None:
        """Reset all tracker state.

        Useful when switching between video sources or cameras.
        """
        self.tracker.delete_all_tracks()
