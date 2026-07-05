"""
ReID Service
============

Wraps the AI engine (``camera/``, ``detection/``, ``tracking/``,
``reid/``) as an importable Python service that FastAPI can call
directly — no ``subprocess`` or shell invocation.

The service:

1. Resolves video paths relative to the project root.
2. Instantiates a :class:`CameraManager` with the requested
   configuration.
3. Runs the pipeline (detection → tracking → ReID) and updates
   the :class:`JobManager` with progress.

All AI imports are **lazy** (inside methods) so the module
can be imported safely before ``sys.path`` is configured.
"""

import sys
from pathlib import Path
from typing import List, Optional

from app.core.config import PROJECT_ROOT
from app.core.logging import get_logger
from app.services.job_manager import JobManager

logger = get_logger(__name__)


def _ensure_engine_importable() -> None:
    """Add the project root to ``sys.path`` if not already present.

    This allows ``from camera.camera_manager import CameraManager``
    and other AI-engine imports to work from the backend process.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
        logger.info("Added AI engine root to sys.path: %s", root)


class ReidService:
    """Service layer wrapping the AI engine.

    All methods are synchronous because the AI pipeline is
    CPU/GPU-bound.  FastAPI calls them inside BackgroundTasks
    (thread pool) to avoid blocking the event loop.
    """

    def process_videos(
        self,
        video_paths: List[str],
        job_id: str,
        job_manager: JobManager,
        *,
        reid_enabled: bool = True,
        confidence_threshold: float = 0.6,
        similarity_threshold: float = 0.75,
        reid_interval: int = 15,
    ) -> None:
        """Run the full AI pipeline for one or more videos.

        This method is designed to be called from a BackgroundTask.
        It updates the :class:`JobManager` with progress and
        final status.

        Args:
            video_paths: Absolute or project-relative video paths.
            job_id: UUID of the job (for progress updates).
            job_manager: Shared job manager instance.
            reid_enabled: Enable cross-camera Person ReID.
            confidence_threshold: YOLO detection confidence.
            similarity_threshold: ReID cosine similarity threshold.
            reid_interval: Frames between ReID re-extractions.

        Raises:
            FileNotFoundError: If any video path does not exist.
        """
        _ensure_engine_importable()

        # ── Lazy import — AI engine ───────────────────────────────────
        from camera.camera_manager import CameraManager  # noqa: E402

        # ── Resolve paths ─────────────────────────────────────────────
        resolved: List[str] = []
        for vp in video_paths:
            p = Path(vp)
            if not p.is_absolute():
                p = PROJECT_ROOT / vp
            p = p.resolve()
            if not p.exists():
                error = f"Video not found: {vp} (resolved: {p})"
                logger.error("[Job %s] %s", job_id, error)
                job_manager.mark_failed(job_id, error)
                return
            resolved.append(str(p))

        # ── Output directory ──────────────────────────────────────────
        output_dir = str(PROJECT_ROOT / "outputs" / job_id)

        logger.info(
            "[Job %s] Starting — %d video(s), reid=%s",
            job_id,
            len(resolved),
            reid_enabled,
        )

        job_manager.mark_running(job_id)

        try:
            manager = CameraManager(
                video_paths=resolved,
                output_dir=output_dir,
                confidence_threshold=confidence_threshold,
                display=False,  # No GUI in backend mode
                reid_enabled=reid_enabled,
                similarity_threshold=similarity_threshold,
                reid_interval=reid_interval,
            )

            # ── Run each camera and report progress ───────────────────
            results = []
            for i, pipeline in enumerate(manager.pipelines, start=1):
                logger.info(
                    "[Job %s] Processing camera %d/%d",
                    job_id,
                    i,
                    len(manager.pipelines),
                )
                result = pipeline.run()
                results.append(result)
                job_manager.update_progress(job_id, i)

            # ── Print summary (same as CLI) ───────────────────────────
            manager.print_summary(results)

            # ── Mark completed ────────────────────────────────────────
            total_frames = sum(
                r.get("frames_processed", 0) for r in results
            )
            logger.info(
                "[Job %s] Completed — %d frames across %d camera(s)",
                job_id,
                total_frames,
                len(results),
            )
            job_manager.mark_completed(job_id, output_dir)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception("[Job %s] Failed — %s", job_id, error_msg)
            job_manager.mark_failed(job_id, error_msg)

    def get_engine_status(self) -> dict:
        """Return basic AI engine availability info.

        Checks whether the AI modules can be imported without
        errors.
        """
        _ensure_engine_importable()

        status = {
            "camera_module": False,
            "detection_module": False,
            "tracking_module": False,
            "reid_module": False,
        }

        try:
            import camera.camera_manager  # noqa: F401
            status["camera_module"] = True
        except ImportError:
            pass

        try:
            import detection.detector  # noqa: F401
            status["detection_module"] = True
        except ImportError:
            pass

        try:
            import tracking.tracker  # noqa: F401
            status["tracking_module"] = True
        except ImportError:
            pass

        try:
            import reid.identity_engine  # noqa: F401
            status["reid_module"] = True
        except ImportError:
            pass

        status["all_modules_available"] = all(status.values())
        return status
