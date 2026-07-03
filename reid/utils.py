"""
ReID Utility Functions
======================

Provides helper functions for person cropping and validation
used by the re-identification pipeline.
"""

from typing import List, Optional

import numpy as np


def crop_person(
    frame: np.ndarray,
    bbox: List[int],
    margin: float = 0.0,
) -> Optional[np.ndarray]:
    """Crop a person from the frame using the bounding box.

    Args:
        frame: BGR image as a NumPy array (H × W × 3).
        bbox: ``[x1, y1, x2, y2]`` bounding box in pixel coordinates.
        margin: Optional fractional margin to add around the crop
            (e.g., 0.1 adds 10% padding on each side).

    Returns:
        Cropped BGR image, or ``None`` if the crop is invalid.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply optional margin
    if margin > 0:
        bw = x2 - x1
        bh = y2 - y1
        x1 = int(x1 - bw * margin)
        y1 = int(y1 - bh * margin)
        x2 = int(x2 + bw * margin)
        y2 = int(y2 + bh * margin)

    # Clamp to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Validate dimensions
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def is_valid_crop(
    crop: Optional[np.ndarray],
    min_height: int = 20,
    min_width: int = 10,
) -> bool:
    """Check whether a crop is large enough for feature extraction.

    Args:
        crop: Cropped image or ``None``.
        min_height: Minimum crop height in pixels.
        min_width: Minimum crop width in pixels.

    Returns:
        ``True`` if the crop meets the minimum size requirements.
    """
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return h >= min_height and w >= min_width
