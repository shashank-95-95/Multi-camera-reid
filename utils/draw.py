"""
Drawing Utilities
=================

Provides functions for annotating video frames with bounding boxes,
track IDs, and confidence scores.

All drawing functions accept and return NumPy arrays so they remain
framework-agnostic.
"""

from typing import List, Tuple

import cv2
import numpy as np


# ── Colour palette (BGR) ──────────────────────────────────────────────
# A visually distinct palette for assigning consistent colours to track
# IDs.  Colours cycle when there are more tracks than entries.
_PALETTE: List[Tuple[int, int, int]] = [
    (255, 76, 76),     # coral-red
    (76, 255, 76),     # lime-green
    (76, 76, 255),     # bright-blue
    (255, 200, 0),     # amber
    (0, 220, 255),     # cyan
    (255, 0, 200),     # magenta
    (128, 255, 0),     # chartreuse
    (255, 128, 0),     # orange
    (0, 128, 255),     # azure
    (200, 0, 255),     # purple
]

# ── Font settings ─────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 2
_BOX_THICKNESS = 2


def _colour_for_id(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic colour for a given track ID."""
    # Use hash so string IDs also map consistently
    idx = hash(track_id) % len(_PALETTE)
    return _PALETTE[idx]


def draw_tracks(
    frame: np.ndarray,
    tracks: List[dict],
) -> np.ndarray:
    """Draw bounding boxes, track IDs, and confidence on the frame.

    Args:
        frame: BGR image to annotate (modified **in-place**).
        tracks: List of track dicts as returned by
            :meth:`DeepSortTracker.update`.  Each dict must contain
            ``track_id``, ``bbox`` (``[x1, y1, x2, y2]``), and
            ``confidence``.

    Returns:
        The annotated frame (same object that was passed in).
    """
    for track in tracks:
        track_id = track["track_id"]
        x1, y1, x2, y2 = track["bbox"]
        conf = track["confidence"]
        colour = _colour_for_id(track_id)

        # --- Bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)

        # --- Label background ---
        if "global_id" in track:
            label = f"T:{track_id} G:{track['global_id']}  {conf:.2f}"
        else:
            label = f"ID:{track_id}  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        label_y1 = max(y1 - th - baseline - 6, 0)
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + tw + 4, y1),
            colour,
            cv2.FILLED,
        )

        # --- Label text ---
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            _FONT,
            _FONT_SCALE,
            (0, 0, 0),          # black text on coloured background
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return frame


def draw_frame_info(
    frame: np.ndarray,
    frame_number: int,
    total_frames: int,
    active_tracks: int,
    fps: float = 0.0,
) -> np.ndarray:
    """Overlay frame-level HUD information.

    Args:
        frame: BGR image to annotate (modified **in-place**).
        frame_number: Current 0-indexed frame number.
        total_frames: Total frames in the video.
        active_tracks: Number of currently-active tracks.
        fps: Processing speed in frames per second.

    Returns:
        The annotated frame.
    """
    info_lines = [
        f"Frame: {frame_number}/{total_frames}",
        f"Tracks: {active_tracks}",
    ]
    if fps > 0:
        info_lines.append(f"FPS: {fps:.1f}")

    y_offset = 30
    for line in info_lines:
        cv2.putText(
            frame,
            line,
            (10, y_offset),
            _FONT,
            _FONT_SCALE,
            (0, 255, 0),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y_offset += 28

    return frame
