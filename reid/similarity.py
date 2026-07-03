"""
Similarity Module
=================

Provides cosine similarity computation and matching utilities
for person re-identification.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np


@dataclass
class SimilarityResult:
    """Result of a similarity comparison.

    Attributes:
        score: Best cosine similarity score (0.0 – 1.0).
        matched_global_id: Global ID of the best match, or ``None``
            if no match exceeds the threshold.
        is_match: Whether the score exceeds the threshold.
    """

    score: float
    matched_global_id: Optional[int]
    is_match: bool


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Compute cosine similarity between two embeddings.

    Both inputs should be 1-D vectors.  If either has zero norm,
    returns 0.0 to avoid division errors.

    Args:
        embedding_a: First feature vector.
        embedding_b: Second feature vector.

    Returns:
        Cosine similarity in the range [-1.0, 1.0].
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def find_best_match(
    query: np.ndarray,
    gallery: List[Tuple[int, np.ndarray]],
    threshold: float = 0.75,
) -> SimilarityResult:
    """Find the best matching identity for a query embedding.

    Args:
        query: Query feature vector (1-D, L2-normalised).
        gallery: List of ``(global_id, representative_embedding)``
            tuples to compare against.
        threshold: Minimum similarity score for a positive match.

    Returns:
        A :class:`SimilarityResult` with the best match details.
    """
    if not gallery:
        return SimilarityResult(
            score=0.0, matched_global_id=None, is_match=False
        )

    best_score = -1.0
    best_id: Optional[int] = None

    for global_id, rep_embedding in gallery:
        score = cosine_similarity(query, rep_embedding)
        if score > best_score:
            best_score = score
            best_id = global_id

    is_match = best_score >= threshold

    return SimilarityResult(
        score=best_score,
        matched_global_id=best_id if is_match else None,
        is_match=is_match,
    )


def find_best_match_with_constraints(
    query: np.ndarray,
    gallery: List[Tuple[int, List[np.ndarray]]],
    threshold: float = 0.75,
    excluded_ids: Optional[Set[int]] = None,
    debug: bool = False,
    track_id: Optional[str] = None,
    camera_id: Optional[int] = None,
) -> SimilarityResult:
    """Find the best match respecting active-identity constraints.

    Unlike :func:`find_best_match`, this function:

    1. Accepts a gallery of ``(global_id, [embeddings])`` and computes
       the **maximum** cosine similarity across *all* stored
       embeddings per identity (more robust than representative-only).
    2. Supports an *exclusion set* to prevent two active tracks in
       the same camera from sharing one global ID.
    3. Iterates candidates in descending similarity order, skipping
       excluded IDs until a valid match is found.

    Args:
        query: L2-normalised query embedding.
        gallery: List of ``(global_id, embeddings_list)`` tuples.
        threshold: Minimum cosine similarity for a positive match.
        excluded_ids: Global IDs to skip (already active in the
            same camera by a different track).
        debug: If ``True``, print matching decisions to stdout.
        track_id: For debug logging only.
        camera_id: For debug logging only.

    Returns:
        A :class:`SimilarityResult` with the best valid match.
    """
    if not gallery:
        if debug:
            print(
                f"    [ReID] Track {track_id}: "
                f"Empty gallery, creating new identity"
            )
        return SimilarityResult(
            score=0.0, matched_global_id=None, is_match=False
        )

    excluded = excluded_ids or set()

    # Compute max similarity per identity across all stored embeddings
    candidates: List[Tuple[int, float]] = []
    for gid, embeddings in gallery:
        if not embeddings:
            continue
        max_sim = max(
            cosine_similarity(query, emb) for emb in embeddings
        )
        candidates.append((gid, max_sim))

    # Sort descending by similarity
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Find first valid candidate above threshold
    for gid, score in candidates:
        if score < threshold:
            break  # All remaining are below threshold

        if gid in excluded:
            if debug:
                print(
                    f"    [ReID] Track {track_id}: "
                    f"Candidate G:{gid} sim={score:.4f} "
                    f"REJECTED (active in Camera {camera_id})"
                )
            continue

        # Valid match found
        if debug:
            print(
                f"    [ReID] Track {track_id}: "
                f"Matched G:{gid} sim={score:.4f}"
            )
        return SimilarityResult(
            score=score, matched_global_id=gid, is_match=True
        )

    # No valid match
    if debug:
        best = candidates[0][1] if candidates else 0.0
        print(
            f"    [ReID] Track {track_id}: "
            f"No valid match (best={best:.4f}, "
            f"threshold={threshold}), creating new identity"
        )
    return SimilarityResult(
        score=candidates[0][1] if candidates else 0.0,
        matched_global_id=None,
        is_match=False,
    )
