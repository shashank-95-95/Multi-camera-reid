"""
Camera Pipeline Module
======================

Provides :class:`CameraPipeline`, the self-contained processing unit
for a single camera stream.  Each pipeline owns its own detector,
tracker, and video I/O — making it safe to run multiple instances
concurrently without shared mutable state.

Phase 3 adds ReID integration with:

- **Track cache**: avoids running OSNet every frame for already-
  matched tracks (configurable interval, default 15 frames).
- **Active-track lifecycle**: deregisters disappeared tracks so
  their global IDs can be reused by future detections.
- **Debug logging**: optional trace of every match decision.
"""

import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from detection.detector import PersonDetector
from tracking.tracker import DeepSortTracker
from utils.video import VideoProcessor
from utils.draw import draw_tracks, draw_frame_info

if TYPE_CHECKING:
    from reid.feature_extractor import FeatureExtractor
    from reid.identity_engine import IdentityEngine


class CameraPipeline:
    """End-to-end detection → tracking → ReID pipeline for one camera.

    Each instance is **completely independent**: it creates its own
    :class:`PersonDetector`, :class:`DeepSortTracker`, and
    :class:`VideoProcessor`, and writes outputs to a dedicated
    per-camera directory.

    When *feature_extractor* and *identity_engine* are supplied
    (Phase 3), the pipeline additionally crops detected persons,
    extracts appearance embeddings, and assigns persistent global
    IDs via the shared :class:`IdentityEngine`.

    **Track cache** (improvement #5): Already-matched tracks reuse
    their cached global ID instead of re-running OSNet every frame.
    Fresh extraction occurs only for new tracks or every
    *reid_interval* frames.

    Attributes:
        camera_id: Integer identifier for this camera (1-indexed).
        video_path: Path to the input video file.
        output_dir: Per-camera output directory.
    """

    def __init__(
        self,
        camera_id: int,
        video_path: str,
        output_dir: str = "outputs",
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.6,
        device: str = "",
        min_bbox_width: int = 40,
        min_bbox_height: int = 60,
        display: bool = True,
        should_stop: Optional[Callable[[], bool]] = None,
        feature_extractor: Optional["FeatureExtractor"] = None,
        identity_engine: Optional["IdentityEngine"] = None,
        reid_interval: int = 15,
        debug: bool = False,
    ) -> None:
        """Initialise a camera pipeline.

        Args:
            camera_id: Unique integer identifier for this camera.
            video_path: Path to the input video file.
            output_dir: Base output directory.
            model_path: Path to YOLO weights file.
            confidence_threshold: Minimum detection confidence.
            device: Compute device for inference.
            min_bbox_width: Minimum detection bbox width in pixels.
            min_bbox_height: Minimum detection bbox height in pixels.
            display: If ``True``, show a live OpenCV window.
            should_stop: Optional callable returning ``True`` when the
                pipeline should terminate early.
            feature_extractor: Optional shared :class:`FeatureExtractor`
                for ReID embedding extraction (Phase 3).
            identity_engine: Optional shared :class:`IdentityEngine`
                for global identity assignment (Phase 3).
            reid_interval: Number of frames between re-extractions
                for cached tracks.  Lower values are more accurate
                but slower.  Default: 15.
            debug: If ``True``, print detailed ReID matching decisions.
        """
        self.camera_id = camera_id
        self.video_path = video_path
        self.output_dir = f"{output_dir}/camera_{camera_id}"

        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._min_bbox_width = min_bbox_width
        self._min_bbox_height = min_bbox_height
        self._display = display
        self._should_stop = should_stop or (lambda: False)
        self._window_name = f"Camera {camera_id}"
        
        # ── NEW: Expose the VideoProcessor so the Manager can access it
        self.processor = None

        # Phase 3 — ReID components (optional, shared across cameras)
        self._feature_extractor = feature_extractor
        self._identity_engine = identity_engine
        self._reid_enabled = (
            feature_extractor is not None
            and identity_engine is not None
        )
        self._reid_interval = reid_interval
        self._debug = debug

        # Track cache: track_id → {global_id, similarity, last_frame}
        self._track_cache: Dict[str, Dict] = {}
        # Last frame each track was seen (for grace-period expiry)
        self._track_last_seen: Dict[str, int] = {}
        # Grace period before evicting a track from cache.
        # Matches DeepSORT max_age so a temporarily-undetected track
        # keeps its global ID across brief occlusions.
        self._cache_max_age: int = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Execute the full detection → tracking → ReID pipeline.

        Returns:
            A result dict with camera_id, frames_processed, etc.
        """
        tag = f"[Camera {self.camera_id}]"

        result: Dict = {
            "camera_id": self.camera_id,
            "video_path": self.video_path,
            "frames_processed": 0,
            "output_dir": self.output_dir,
            "quit_requested": False,
            "error": None,
        }

        # ── 1. Initialise components ──────────────────────────────────
        print(f"{tag} Initialising detector …")
        detector = PersonDetector(
            model_path=self._model_path,
            confidence_threshold=self._confidence_threshold,
            device=self._device,
            min_bbox_width=self._min_bbox_width,
            min_bbox_height=self._min_bbox_height,
        )

        print(f"{tag} Initialising tracker …")
        tracker = DeepSortTracker()

        print(f"{tag} Opening video: {self.video_path}")
        # ── CHANGED: Save the VideoProcessor to self.processor
        self.processor = VideoProcessor(
            video_path=self.video_path,
            output_dir=self.output_dir,
            camera_id=self.camera_id,
        )

        reid_label = " + ReID" if self._reid_enabled else ""
        print(
            f"{tag} Video — "
            f"{self.processor.width}×{self.processor.height} @ {self.processor.fps:.1f} FPS, "
            f"{self.processor.total_frames} frames{reid_label}"
        )

        # ── 2. Frame loop ─────────────────────────────────────────────
        frame_number = 0
        prev_time = time.time()
        processing_fps = 0.0
        quit_requested = False

        try:
            while not self._should_stop():
                ret, frame = self.processor.read_frame()
                if not ret:
                    break

                # --- Detect persons ---
                detections = detector.detect(frame)

                # --- Update tracker ---
                tracks = tracker.update(detections, frame)

                # --- Phase 3: Person Re-Identification ────────────────
                if self._reid_enabled:
                    self._assign_global_ids(frame, tracks, frame_number)

                # --- Record tracking results ---
                for track in tracks:
                    self.processor.add_tracking_result(
                        frame_number=frame_number,
                        track_id=track["track_id"],
                        bbox=track["bbox"],
                        confidence=track["confidence"],
                        global_id=track.get("global_id"),
                        reid_similarity=track.get("similarity"),
                    )

                # --- Annotate frame ---
                draw_tracks(frame, tracks)
                draw_frame_info(
                    frame,
                    frame_number=frame_number,
                    total_frames=self.processor.total_frames,
                    active_tracks=len(tracks),
                    fps=processing_fps,
                )

                # --- Write output ---
                self.processor.write_frame(frame)

                # --- Display (unless headless) ---
                if self._display:
                    if self.processor.display_frame(frame, self._window_name):
                        print(f"{tag} Quit requested by user.")
                        quit_requested = True
                        break

                # --- Calculate processing FPS ---
                current_time = time.time()
                elapsed = current_time - prev_time
                if elapsed > 0:
                    processing_fps = 1.0 / elapsed
                prev_time = current_time

                frame_number += 1

                # --- Progress logging (every 100 frames) ---
                if frame_number % 100 == 0:
                    print(
                        f"{tag} Processed "
                        f"{frame_number}/{self.processor.total_frames} frames "
                        f"({processing_fps:.1f} FPS)"
                    )

        except KeyboardInterrupt:
            print(f"\n{tag} Interrupted by user.")
            quit_requested = True

        finally:
            # ── CHANGED: Use self.processor
            json_path = self.processor.save_tracking_results()
            self.processor.release()

            result["frames_processed"] = frame_number
            result["quit_requested"] = quit_requested

            print(f"\n{'─' * 55}")
            print(f"  Camera {self.camera_id} — Processing complete")
            print(f"  Frames processed : {frame_number}")
            print(f"  Output video     : {self.output_dir}/tracked_video.mp4")
            print(f"  Tracking JSON    : {json_path}")
            print(f"{'─' * 55}")

        return result

    # ------------------------------------------------------------------
    # Phase 3 — ReID integration with track cache & lifecycle
    # ------------------------------------------------------------------

    def _assign_global_ids(
        self,
        frame,
        tracks: List[Dict],
        frame_number: int,
    ) -> None:
        """Crop tracked persons, extract embeddings, assign global IDs.

        Implements three critical mechanisms:

        1. **Track lifecycle**: Deregisters disappeared tracks from the
           active registry so their global IDs can be reused.
        2. **Track cache**: Reuses cached global IDs for tracks that
           were recently extracted, avoiding redundant OSNet calls.
        3. **Batch extraction**: Extracts embeddings for all tracks
           needing fresh extraction in a single forward pass.

        Args:
            frame: Current BGR video frame.
            tracks: List of track dicts from DeepSortTracker.
            frame_number: Current 0-indexed frame number.
        """
        from reid.utils import crop_person, is_valid_crop

        current_track_ids = {t["track_id"] for t in tracks}

        # ── 1. Update last-seen timestamps & expire stale tracks ──────
        # We do NOT evict a track the instant it is missing from one
        # frame.  DeepSORT may keep the track alive internally for up
        # to max_age frames.  Only after _cache_max_age frames of
        # absence do we consider the track truly gone.
        for track in tracks:
            self._track_last_seen[track["track_id"]] = frame_number

        stale_tids = [
            tid
            for tid, last_frame in self._track_last_seen.items()
            if frame_number - last_frame > self._cache_max_age
        ]
        for tid in stale_tids:
            self._identity_engine.deregister_track(
                self.camera_id, tid
            )
            self._track_cache.pop(tid, None)
            del self._track_last_seen[tid]
            if self._debug:
                print(
                    f"    [ReID] Camera {self.camera_id}: "
                    f"Track {tid} expired after "
                    f"{self._cache_max_age} frames unseen"
                )

        # ── 2. Apply cached values & identify tracks needing extraction
        extract_indices: List[int] = []

        for i, track in enumerate(tracks):
            tid = track["track_id"]
            cached = self._track_cache.get(tid)

            if cached is not None:
                frames_since = frame_number - cached["last_extract_frame"]
                if frames_since < self._reid_interval:
                    # Reuse cached assignment
                    track["global_id"] = cached["global_id"]
                    track["similarity"] = cached["similarity"]
                    continue

            # Needs fresh extraction (new track or interval elapsed)
            extract_indices.append(i)

        if not extract_indices:
            return

        # ── 3. Batch crop & extract ───────────────────────────────────
        crops: List = []
        crop_indices: List[int] = []

        for i in extract_indices:
            crop = crop_person(frame, tracks[i]["bbox"])
            if is_valid_crop(crop):
                crops.append(crop)
                crop_indices.append(i)
            else:
                # Invalid crop — fall back to cache if available
                tid = tracks[i]["track_id"]
                cached = self._track_cache.get(tid)
                if cached is not None:
                    tracks[i]["global_id"] = cached["global_id"]
                    tracks[i]["similarity"] = cached["similarity"]

        if not crops:
            return

        embeddings = self._feature_extractor.extract_batch(crops)

        # ── 4. Assign or update identities ────────────────────────────
        for j, embedding in enumerate(embeddings):
            idx = crop_indices[j]
            track = tracks[idx]
            tid = track["track_id"]
            cached = self._track_cache.get(tid)

            if cached is not None:
                # ── RE-EXTRACTION: update identity, keep global ID ────
                # The track already has a global ID. Only refresh the
                # embedding store — never call assign_identity().
                self._identity_engine.update_identity(
                    global_id=cached["global_id"],
                    embedding=embedding,
                    camera_id=self.camera_id,
                    track_id=tid,
                    debug=self._debug,
                )
                track["global_id"] = cached["global_id"]
                track["similarity"] = cached["similarity"]
                # Only update the extraction timestamp
                cached["last_extract_frame"] = frame_number

            else:
                # ── NEW TRACK: full identity assignment ───────────────
                global_id, similarity = (
                    self._identity_engine.assign_identity(
                        embedding=embedding,
                        camera_id=self.camera_id,
                        track_id=tid,
                        debug=self._debug,
                    )
                )
                sim_rounded = round(similarity, 4)
                track["global_id"] = global_id
                track["similarity"] = sim_rounded

                self._track_cache[tid] = {
                    "global_id": global_id,
                    "similarity": sim_rounded,
                    "last_extract_frame": frame_number,
                }