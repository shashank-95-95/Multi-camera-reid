"""
Embedding Store
===============

Provides :class:`EmbeddingStore`, a fixed-capacity FIFO buffer for
appearance embeddings.

Each :class:`Identity` owns one store.  The store automatically
computes a representative embedding (L2-normalised mean) used for
cross-camera similarity comparisons.
"""

from typing import List, Optional

import numpy as np


class EmbeddingStore:
    """Fixed-capacity FIFO buffer for appearance embeddings.

    When the store reaches capacity, the oldest embedding is evicted
    to make room for the newest.  This keeps the representative
    embedding biased toward the person's recent appearance.

    Attributes:
        max_embeddings: Maximum number of stored embeddings.
    """

    def __init__(self, max_embeddings: int = 10) -> None:
        """Initialise the embedding store.

        Args:
            max_embeddings: Maximum number of embeddings to retain.
                Older entries are evicted in FIFO order.
        """
        self.max_embeddings = max_embeddings
        self._embeddings: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray) -> None:
        """Add an embedding to the store.

        If at capacity, the oldest embedding is removed first.

        Args:
            embedding: 1-D feature vector (should be L2-normalised).
        """
        if len(self._embeddings) >= self.max_embeddings:
            self._embeddings.pop(0)
        self._embeddings.append(embedding.copy())

    @property
    def representative(self) -> Optional[np.ndarray]:
        """Compute the representative embedding.

        Returns the L2-normalised mean of all stored embeddings.
        Returns ``None`` if the store is empty.
        """
        if not self._embeddings:
            return None

        mean = np.mean(self._embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean

    @property
    def count(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._embeddings)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if no embeddings are stored."""
        return len(self._embeddings) == 0

    @property
    def all_embeddings(self) -> List[np.ndarray]:
        """Return all stored embeddings (read-only copy).

        Used by :func:`find_best_match_with_constraints` to compute
        the maximum cosine similarity across all stored appearances,
        which is more robust than representative-only matching.
        """
        return list(self._embeddings)
