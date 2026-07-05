"""
Job Manager
===========

In-memory job tracking for asynchronous video processing.

Each processing request creates a :class:`Job` that tracks
status, progress, timing, and output location.  The
:class:`JobManager` is a thread-safe singleton shared across
the application.

Thread safety is required because FastAPI BackgroundTasks
execute in a thread pool while API handlers run in the
async event loop.
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class JobStatus(str, Enum):
    """Lifecycle states for a processing job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Mutable state for a single processing job.

    Attributes:
        job_id: UUID string uniquely identifying this job.
        status: Current lifecycle state.
        progress: Completion percentage (0–100).
        created_at: When the job was created.
        started_at: When processing began.
        finished_at: When processing ended (success or failure).
        output_directory: Path to per-job output files.
        error_message: Human-readable error if status is FAILED.
        video_paths: Input video paths.
        reid_enabled: Whether ReID was requested.
        cameras_total: Number of cameras to process.
        cameras_completed: Number of cameras finished so far.
    """

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output_directory: Optional[str] = None
    error_message: Optional[str] = None
    video_paths: List[str] = field(default_factory=list)
    reid_enabled: bool = True
    cameras_total: int = 0
    cameras_completed: int = 0

    @property
    def processing_time(self) -> Optional[float]:
        """Elapsed processing time in seconds, or None."""
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class JobManager:
    """Thread-safe in-memory job registry.

    Stores all :class:`Job` instances for the lifetime of the
    server process.  A future phase may persist jobs to the
    database for crash recovery.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_job(
        self,
        video_paths: List[str],
        reid_enabled: bool = True,
        output_directory: Optional[str] = None,
    ) -> Job:
        """Create and register a new job.

        Args:
            video_paths: List of input video file paths.
            reid_enabled: Whether Person ReID is enabled.
            output_directory: Per-job output directory.

        Returns:
            The newly-created :class:`Job`.
        """
        job = Job(
            job_id=str(uuid.uuid4()),
            video_paths=list(video_paths),
            reid_enabled=reid_enabled,
            cameras_total=len(video_paths),
            output_directory=output_directory,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        """Return all jobs, newest first."""
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def mark_running(self, job_id: str) -> None:
        """Transition a job to RUNNING."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)

    def update_progress(
        self,
        job_id: str,
        cameras_completed: int,
    ) -> None:
        """Update progress after a camera finishes processing.

        Args:
            job_id: The job to update.
            cameras_completed: Number of cameras that have finished.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.cameras_total > 0:
                job.cameras_completed = cameras_completed
                job.progress = round(
                    (cameras_completed / job.cameras_total) * 100, 1
                )

    def mark_completed(
        self,
        job_id: str,
        output_directory: Optional[str] = None,
    ) -> None:
        """Transition a job to COMPLETED."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.finished_at = datetime.now(timezone.utc)
                job.cameras_completed = job.cameras_total
                if output_directory:
                    job.output_directory = output_directory

    def mark_failed(
        self, job_id: str, error_message: str
    ) -> None:
        """Transition a job to FAILED with an error message."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(timezone.utc)
                job.error_message = error_message
