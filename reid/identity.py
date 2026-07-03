"""
Identity Module
===============

Provides the :class:`Identity` class representing a single
globally-identified person across all cameras.

Each identity maintains an :class:`EmbeddingStore` with the latest
N appearance embeddings, enabling robust cross-camera matching
as the person's appearance changes over time.
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from reid.embedding_store import EmbeddingStore


class Identity:
    """A globally-identified person.

    Attributes:
        global_id: Unique global identifier.
        embedding_store: Stores the latest N appearance embeddings.
        last_seen_timestamp: Unix timestamp of the last observation.
        last_seen_camera: Camera ID where the person was last seen.
        track_history: List of ``(camera_id, track_id)`` observations.
        similarity_history: List of similarity scores for matches.
    """

    def __init__(
        self,
        global_id: int,
        embedding: np.ndarray,
        camera_id: int,
        track_id: Optional[str] = None,
        max_embeddings: int = 10,
    ) -> None:
        """Initialise a new identity.

        Args:
            global_id: Unique global identifier for this person.
            embedding: Initial appearance embedding.
            camera_id: Camera where this person was first seen.
            track_id: Optional local track ID from the camera.
            max_embeddings: Maximum embeddings to retain in store.
        """
        self.global_id = global_id
        self.embedding_store = EmbeddingStore(
            max_embeddings=max_embeddings
        )
        self.embedding_store.add(embedding)

        self.last_seen_timestamp: float = time.time()
        self.last_seen_camera: int = camera_id
        self.track_history: List[Tuple[int, Optional[str]]] = [
            (camera_id, track_id)
        ]
        self.similarity_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add a new appearance embedding to the store.

        Args:
            embedding: 1-D L2-normalised feature vector.
        """
        self.embedding_store.add(embedding)

    def update_camera(self, camera_id: int) -> None:
        """Update the last-seen camera.

        Args:
            camera_id: Camera where the person was just observed.
        """
        self.last_seen_camera = camera_id

    def update_timestamp(self) -> None:
        """Update the last-seen timestamp to now."""
        self.last_seen_timestamp = time.time()

    def record_observation(
        self,
        camera_id: int,
        track_id: Optional[str] = None,
        similarity: Optional[float] = None,
    ) -> None:
        """Record a full observation event.

        Convenience method that updates camera, timestamp, track
        history, and similarity history in one call.

        Args:
            camera_id: Camera where the person was observed.
            track_id: Local track ID from the camera.
            similarity: Cosine similarity score for this match.
        """
        self.update_camera(camera_id)
        self.update_timestamp()
        self.track_history.append((camera_id, track_id))
        if similarity is not None:
            self.similarity_history.append(similarity)

    @property
    def representative_embedding(self) -> Optional[np.ndarray]:
        """Return the representative embedding (L2-normalised mean).

        Delegates to the underlying :class:`EmbeddingStore`.
        """
        return self.embedding_store.representative

    @property
    def embedding_count(self) -> int:
        """Return the number of stored embeddings."""
        return self.embedding_store.count
