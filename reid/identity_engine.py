"""
Identity Engine
===============

Provides :class:`IdentityEngine`, the central matching engine for
cross-camera person re-identification.

**Critical invariant**: Two simultaneously-active tracks within the
same camera must NEVER share a global ID.  This is enforced by the
active-identity registry, which tracks which global IDs are
currently in use per camera.

The engine:

1. Receives an appearance embedding (from :class:`FeatureExtractor`).
2. Computes **max** cosine similarity against all stored embeddings
   per identity (not just the representative).
3. Skips candidates whose global ID is already active in the same
   camera by a different track.
4. Either matches the best valid identity or creates a new one.
5. Returns the assigned global ID and similarity score.

**Important**: The engine is shared across all camera pipelines so
that identities persist across cameras.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from reid.identity import Identity
from reid.similarity import find_best_match_with_constraints


class IdentityEngine:
    """Central engine for cross-camera person re-identification.

    Owns all :class:`Identity` instances, the active-identity
    registry, and handles the match-or-create decision.

    Attributes:
        similarity_threshold: Minimum cosine similarity for a match.
        max_embeddings: Maximum embeddings stored per identity.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_embeddings: int = 10,
    ) -> None:
        """Initialise the identity engine.

        Args:
            similarity_threshold: Minimum cosine similarity required
                to consider two embeddings as the same person.
            max_embeddings: Maximum embeddings retained per identity.
        """
        self.similarity_threshold = similarity_threshold
        self.max_embeddings = max_embeddings
        self._identities: Dict[int, Identity] = {}
        self._next_global_id: int = 1

        # Active-identity registry: camera_id → {track_id → global_id}
        # Enforces the constraint that no two active tracks in the
        # same camera share a global ID.
        self._active_assignments: Dict[int, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Active-identity registry
    # ------------------------------------------------------------------

    def register_active(
        self, camera_id: int, track_id: str, global_id: int
    ) -> None:
        """Register an active track → global ID assignment.

        If the track was previously assigned a different global ID,
        the old assignment is silently overwritten.

        Args:
            camera_id: Camera where the track is active.
            track_id: Local track ID from DeepSORT.
            global_id: Assigned global identity.
        """
        if camera_id not in self._active_assignments:
            self._active_assignments[camera_id] = {}
        self._active_assignments[camera_id][track_id] = global_id

    def deregister_track(
        self, camera_id: int, track_id: str
    ) -> None:
        """Remove a disappeared track from the active registry.

        Once deregistered, the global ID may be matched again by
        future tracks in the same camera.

        Args:
            camera_id: Camera the track belonged to.
            track_id: The disappeared track's ID.
        """
        if camera_id in self._active_assignments:
            self._active_assignments[camera_id].pop(track_id, None)

    def get_excluded_ids(
        self, camera_id: int, track_id: Optional[str] = None
    ) -> Set[int]:
        """Get global IDs active in this camera by *other* tracks.

        The current track's own global ID is excluded from the result
        to allow self-matching during re-extraction.

        Args:
            camera_id: Camera to check.
            track_id: Current track (its assignment is not excluded).

        Returns:
            Set of global IDs that must NOT be assigned to *track_id*.
        """
        if camera_id not in self._active_assignments:
            return set()

        excluded: Set[int] = set()
        for tid, gid in self._active_assignments[camera_id].items():
            if tid != track_id:
                excluded.add(gid)
        return excluded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_identity(
        self,
        embedding: np.ndarray,
        camera_id: int,
        track_id: Optional[str] = None,
        debug: bool = False,
    ) -> Tuple[int, float]:
        """Match an embedding to an existing identity or create one.

        **Active-identity constraint**: If the best candidate is
        already assigned to another active track in the same camera,
        it is skipped.  The next-best candidate is evaluated, and so
        on.  If no valid candidate exists, a new identity is created.

        Args:
            embedding: L2-normalised appearance embedding.
            camera_id: Camera that produced this detection.
            track_id: Local track ID from DeepSORT.
            debug: If ``True``, print matching decisions.

        Returns:
            ``(global_id, similarity_score)`` — similarity is 1.0 for
            newly-created identities.
        """
        # ── Build exclusion set ───────────────────────────────────────
        excluded = self.get_excluded_ids(camera_id, track_id)

        # ── Build gallery with ALL stored embeddings per identity ─────
        gallery: List[Tuple[int, List[np.ndarray]]] = []
        for gid, identity in self._identities.items():
            all_embs = identity.embedding_store.all_embeddings
            if all_embs:
                gallery.append((gid, all_embs))

        # ── Constrained matching ──────────────────────────────────────
        result = find_best_match_with_constraints(
            query=embedding,
            gallery=gallery,
            threshold=self.similarity_threshold,
            excluded_ids=excluded,
            debug=debug,
            track_id=track_id,
            camera_id=camera_id,
        )

        if result.is_match and result.matched_global_id is not None:
            # ── Update existing identity ──────────────────────────────
            global_id = result.matched_global_id
            identity = self._identities[global_id]
            identity.add_embedding(embedding)
            identity.record_observation(
                camera_id=camera_id,
                track_id=track_id,
                similarity=result.score,
            )
            self.register_active(camera_id, track_id, global_id)
            return global_id, result.score

        else:
            # ── Create new identity ───────────────────────────────────
            global_id = self._next_global_id
            self._next_global_id += 1

            identity = Identity(
                global_id=global_id,
                embedding=embedding,
                camera_id=camera_id,
                track_id=track_id,
                max_embeddings=self.max_embeddings,
            )
            self._identities[global_id] = identity
            self.register_active(camera_id, track_id, global_id)

            if debug:
                print(
                    f"    [ReID] Track {track_id}: "
                    f"Created new G:{global_id}"
                )
            return global_id, 1.0

    def update_identity(
        self,
        global_id: int,
        embedding: np.ndarray,
        camera_id: int,
        track_id: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Update an existing identity with a fresh embedding.

        Unlike :meth:`assign_identity`, this method **never** changes
        the global ID.  It is used during periodic re-extraction of
        already-matched tracks to keep the embedding store current
        without risking identity reassignment.

        Args:
            global_id: The identity to update.
            embedding: Fresh L2-normalised appearance embedding.
            camera_id: Camera where the track is active.
            track_id: Local track ID (for observation history).
            debug: If ``True``, log the update.
        """
        identity = self._identities.get(global_id)
        if identity is None:
            return

        identity.add_embedding(embedding)
        identity.record_observation(
            camera_id=camera_id,
            track_id=track_id,
        )

        if debug:
            print(
                f"    [ReID] Track {track_id}: "
                f"Updated G:{global_id} (embedding refresh)"
            )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_identity(self, global_id: int) -> Optional[Identity]:
        """Retrieve an identity by global ID."""
        return self._identities.get(global_id)

    @property
    def identity_count(self) -> int:
        """Return the total number of known identities."""
        return len(self._identities)

    @property
    def all_identities(self) -> Dict[int, Identity]:
        """Return the full identity registry (read-only copy)."""
        return dict(self._identities)

    def summary(self) -> str:
        """Return a human-readable summary of the engine state."""
        lines = [
            f"IdentityEngine: {self.identity_count} identities, "
            f"threshold={self.similarity_threshold}"
        ]
        for gid, identity in self._identities.items():
            lines.append(
                f"  G:{gid} | embeddings={identity.embedding_count}"
                f" | last_cam={identity.last_seen_camera}"
                f" | observations={len(identity.track_history)}"
            )
        return "\n".join(lines)
