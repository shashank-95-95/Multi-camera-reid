"""
Camera Manager Module
=====================

Provides :class:`CameraManager` which orchestrates one or more
:class:`CameraPipeline` instances.

**Execution model**: The manager is deliberately decoupled from the
pipeline.  :class:`CameraPipeline` exposes a single ``run()`` method;
the manager decides *how* to invoke it.  Swapping from sequential
to threaded or multiprocess execution requires changes only here.

Phase 3 adds optional ReID support: when enabled, the manager creates
a shared :class:`FeatureExtractor` and :class:`IdentityEngine` and
passes them to every pipeline, enabling cross-camera identity matching.
"""

from typing import Dict, List

from camera.camera_pipeline import CameraPipeline


class CameraManager:
    """Orchestrates multiple camera pipelines.

    Each video path produces one :class:`CameraPipeline` with an
    auto-assigned camera ID (1-indexed).  All pipelines share a
    common stop signal so that a user quit in *any* window stops
    all remaining work.

    When ``reid_enabled=True``, a single :class:`FeatureExtractor`
    and :class:`IdentityEngine` are created and shared across all
    pipelines for cross-camera person re-identification.

    Attributes:
        pipelines: List of configured :class:`CameraPipeline` instances.
    """

    def __init__(
        self,
        video_paths: List[str],
        output_dir: str = "outputs",
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.6,
        device: str = "",
        min_bbox_width: int = 40,
        min_bbox_height: int = 60,
        display: bool = True,
        reid_enabled: bool = False,
        similarity_threshold: float = 0.75,
        reid_interval: int = 15,
        debug: bool = False,
    ) -> None:
        """Initialise the camera manager.

        Args:
            video_paths: List of paths to input video files.  One
                :class:`CameraPipeline` is created per path.
            output_dir: Base output directory.  Each camera writes to
                ``<output_dir>/camera_<id>/``.
            model_path: YOLO weights path.  Shared as config — each
                pipeline loads its own model instance at run time.
            confidence_threshold: Minimum detection confidence.
            device: Compute device for inference.
            min_bbox_width: Minimum detection bbox width in pixels.
            min_bbox_height: Minimum detection bbox height in pixels.
            display: If ``True``, each pipeline shows a live window.
            reid_enabled: If ``True``, enable Person Re-Identification
                using OSNet.  Requires ``torchreid`` to be installed.
            similarity_threshold: Minimum cosine similarity for a
                ReID match (only used when *reid_enabled* is True).
            reid_interval: Number of frames between re-extractions
                for cached tracks (default: 15).
            debug: If ``True``, print detailed ReID match decisions.

        Raises:
            ValueError: If *video_paths* is empty.
        """
        if not video_paths:
            raise ValueError("At least one video path is required.")

        self._stop_requested: bool = False

        # ── Phase 3: Shared ReID components ───────────────────────────
        feature_extractor = None
        identity_engine = None

        if reid_enabled:
            from reid.feature_extractor import FeatureExtractor
            from reid.identity_engine import IdentityEngine

            print("[Manager] Initialising ReID components …")
            feature_extractor = FeatureExtractor(device=device)
            identity_engine = IdentityEngine(
                similarity_threshold=similarity_threshold,
            )
            self._identity_engine = identity_engine
        else:
            self._identity_engine = None

        self._debug = debug

        # ── Build pipelines ───────────────────────────────────────────
        self.pipelines: List[CameraPipeline] = []
        for idx, path in enumerate(video_paths, start=1):
            pipeline = CameraPipeline(
                camera_id=idx,
                video_path=path,
                output_dir=output_dir,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
                min_bbox_width=min_bbox_width,
                min_bbox_height=min_bbox_height,
                display=display,
                should_stop=self._is_stop_requested,
                feature_extractor=feature_extractor,
                identity_engine=identity_engine,
                reid_interval=reid_interval,
                debug=debug,
            )
            self.pipelines.append(pipeline)

        reid_label = " (ReID enabled)" if reid_enabled else ""
        print(
            f"[Manager] Configured {len(self.pipelines)} camera(s)"
            f"{reid_label}: "
            + ", ".join(
                f"Camera {p.camera_id}" for p in self.pipelines
            )
        )

    # ------------------------------------------------------------------
    # Stop signal
    # ------------------------------------------------------------------

    def _is_stop_requested(self) -> bool:
        """Shared stop predicate passed to every pipeline."""
        return self._stop_requested

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    def run_sequential(self) -> List[Dict]:
        """Execute pipelines one after another.

        If any pipeline reports a user-quit (Q / Esc), remaining
        pipelines are skipped.  If a pipeline raises an exception,
        the error is captured and processing continues with the next
        camera.

        Returns:
            List of result dicts, one per pipeline.
        """
        results: List[Dict] = []

        for pipeline in self.pipelines:
            if self._stop_requested:
                print(
                    f"[Manager] Skipping Camera {pipeline.camera_id} "
                    f"(stop requested)"
                )
                break

            try:
                print(f"\n{'=' * 55}")
                print(f"  Starting Camera {pipeline.camera_id}")
                print(f"  Video : {pipeline.video_path}")
                print(f"  Output: {pipeline.output_dir}")
                print(f"{'=' * 55}\n")

                result = pipeline.run()
                results.append(result)

                if result.get("quit_requested"):
                    self._stop_requested = True

            except FileNotFoundError as exc:
                msg = f"Video not found — {exc}"
                print(f"[ERROR] Camera {pipeline.camera_id}: {msg}")
                results.append(self._error_result(pipeline, msg))

            except RuntimeError as exc:
                msg = f"Runtime failure — {exc}"
                print(f"[ERROR] Camera {pipeline.camera_id}: {msg}")
                results.append(self._error_result(pipeline, msg))

            except Exception as exc:
                msg = f"Unexpected error — {exc}"
                print(f"[ERROR] Camera {pipeline.camera_id}: {msg}")
                results.append(self._error_result(pipeline, msg))

        return results

    # Future execution strategies (drop-in replacements):
    #
    # def run_threaded(self, max_workers: int = 4) -> List[Dict]:
    #     from concurrent.futures import ThreadPoolExecutor, as_completed
    #     with ThreadPoolExecutor(max_workers=max_workers) as pool:
    #         futures = {pool.submit(p.run): p for p in self.pipelines}
    #         results = []
    #         for future in as_completed(futures):
    #             result = future.result()
    #             results.append(result)
    #             if result.get("quit_requested"):
    #                 self._stop_requested = True
    #     return sorted(results, key=lambda r: r["camera_id"])

    def run_all(self) -> List[Dict]:
        """Execute all pipelines using the active strategy.

        Currently delegates to :meth:`run_sequential`.

        Returns:
            List of result dicts, one per camera.
        """
        return self.run_sequential()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(pipeline: CameraPipeline, error: str) -> Dict:
        """Build a result dict for a failed pipeline."""
        return {
            "camera_id": pipeline.camera_id,
            "video_path": pipeline.video_path,
            "frames_processed": 0,
            "output_dir": pipeline.output_dir,
            "quit_requested": False,
            "error": error,
        }

    def print_summary(self, results: List[Dict]) -> None:
        """Print a human-readable summary of all pipeline results.

        Args:
            results: List of result dicts returned by :meth:`run_all`.
        """
        total_frames = 0
        successful = 0
        failed = 0

        print(f"\n{'=' * 55}")
        print("  MULTI-CAMERA PROCESSING SUMMARY")
        print(f"{'=' * 55}")

        for r in results:
            cam_id = r.get("camera_id", "?")
            frames = r.get("frames_processed", 0)
            error = r.get("error")

            if error:
                failed += 1
                print(f"  ✗ Camera {cam_id}: FAILED — {error}")
            else:
                successful += 1
                total_frames += frames
                print(f"  ✓ Camera {cam_id}: {frames} frames processed")
                print(f"    └─ {r.get('output_dir', 'N/A')}/")

        print(f"{'─' * 55}")
        print(
            f"  Cameras: {successful} succeeded, {failed} failed"
        )
        print(f"  Total frames: {total_frames}")

        # Phase 3: Print ReID summary if engine was active
        if self._identity_engine is not None:
            print(
                f"  Global identities: "
                f"{self._identity_engine.identity_count}"
            )

        print(f"{'=' * 55}\n")

        # Detailed ReID summary
        if self._identity_engine is not None:
            print(self._identity_engine.summary())
            print()
