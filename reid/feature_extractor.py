"""
Feature Extractor Module
========================

Provides :class:`FeatureExtractor` which uses a pretrained OSNet model
to extract appearance embeddings from person crops.

The extractor handles:

- BGR → RGB conversion
- Resize to OSNet input size (256 × 128)
- ImageNet normalisation
- Forward pass through OSNet
- L2 normalisation of the output embedding
- Batch processing for GPU efficiency
"""

from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from reid.osnet_loader import load_osnet


# ── Preprocessing pipeline ────────────────────────────────────────────
# OSNet expects 256×128 (H×W), RGB, normalised with ImageNet stats.
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                     # HWC uint8 → CHW [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],            # ImageNet mean
        std=[0.229, 0.224, 0.225],             # ImageNet std
    ),
])

# OSNet input size: (height, width)
_INPUT_SIZE = (256, 128)


class FeatureExtractor:
    """Extracts L2-normalised appearance embeddings from person crops.

    Uses a pretrained OSNet model loaded via :func:`load_osnet`.
    The model is loaded **once** at construction and reused for all
    subsequent calls.

    Attributes:
        device: The compute device the model resides on.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "",
    ) -> None:
        """Initialise the feature extractor.

        Args:
            model_name: TorchReID model name (default: ``osnet_x1_0``).
            device: Compute device (``"cpu"``, ``"cuda"``, or ``""``
                for auto-selection).
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = load_osnet(
            model_name=model_name,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess a single person crop for OSNet.

        Args:
            crop: BGR image as a NumPy array (H × W × 3).

        Returns:
            Preprocessed tensor of shape ``(3, 256, 128)``.
        """
        # BGR → RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Resize to (height=256, width=128)
        resized = cv2.resize(
            rgb,
            (_INPUT_SIZE[1], _INPUT_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return _TRANSFORM(resized)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract a single L2-normalised embedding.

        Args:
            crop: BGR person crop as a NumPy array.

        Returns:
            1-D NumPy array (512-d for ``osnet_x1_0``).
        """
        tensor = self._preprocess(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model(tensor)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()

    def extract_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract L2-normalised embeddings for a batch of crops.

        Batched inference is significantly faster on GPU than
        sequential single-crop extraction.

        Args:
            crops: List of BGR person crops.

        Returns:
            List of 1-D NumPy arrays, one per crop.
        """
        if not crops:
            return []

        tensors = [self._preprocess(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self._model(batch)
        features = F.normalize(features, p=2, dim=1)

        return [f.cpu().numpy().flatten() for f in features]
