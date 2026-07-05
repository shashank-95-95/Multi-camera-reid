"""
Jobs Route
==========

Provides endpoints for submitting video processing requests
and querying job status.

Endpoints:

- ``POST /process`` — Submit a new processing job.
- ``GET  /jobs``     — List all jobs.
- ``GET  /jobs/{job_id}`` — Get status of a specific job.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.logging import get_logger
from app.dependencies import get_job_manager, get_reid_service
from app.schemas.jobs import (
    JobListResponse,
    JobResponse,
    ProcessRequest,
    ProcessResponse,
)
from app.services.job_manager import Job, JobManager
from app.services.reid_service import ReidService

logger = get_logger(__name__)

router = APIRouter(tags=["Processing"])


# ── Helpers ───────────────────────────────────────────────────────────


def _job_to_response(job: Job) -> JobResponse:
    """Convert a :class:`Job` dataclass to a Pydantic response."""
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        cameras_total=job.cameras_total,
        cameras_completed=job.cameras_completed,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        processing_time=job.processing_time,
        output_directory=job.output_directory,
        error_message=job.error_message,
        video_paths=job.video_paths,
        reid_enabled=job.reid_enabled,
    )


# ── POST /process ────────────────────────────────────────────────────


@router.post(
    "/process",
    response_model=ProcessResponse,
    status_code=202,
    summary="Submit a video processing job",
    description=(
        "Accepts one or more video paths and starts the AI pipeline "
        "(YOLO → DeepSORT → OSNet ReID) in the background.  Returns "
        "immediately with a job ID that can be used to poll status."
    ),
)
async def submit_processing_job(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    reid_service: ReidService = Depends(get_reid_service),
    job_manager: JobManager = Depends(get_job_manager),
) -> ProcessResponse:
    """Create a processing job and launch it in the background."""

    # ── Create job ────────────────────────────────────────────────────
    job = job_manager.create_job(
        video_paths=request.video_paths,
        reid_enabled=request.reid_enabled,
    )

    logger.info(
        "[Job %s] Created — %d video(s), reid=%s",
        job.job_id,
        len(request.video_paths),
        request.reid_enabled,
    )

    # ── Schedule background processing ────────────────────────────────
    background_tasks.add_task(
        reid_service.process_videos,
        video_paths=request.video_paths,
        job_id=job.job_id,
        job_manager=job_manager,
        reid_enabled=request.reid_enabled,
        confidence_threshold=request.confidence_threshold,
        similarity_threshold=request.similarity_threshold,
        reid_interval=request.reid_interval,
    )

    return ProcessResponse(
        job_id=job.job_id,
        status=job.status.value,
    )


# ── GET /jobs/{job_id} ───────────────────────────────────────────────


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get job status",
    description="Returns detailed status of a specific processing job.",
)
async def get_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """Retrieve the current status of a processing job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        )
    return _job_to_response(job)


# ── GET /jobs ────────────────────────────────────────────────────────


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List all jobs",
    description="Returns all processing jobs, newest first.",
)
async def list_jobs(
    job_manager: JobManager = Depends(get_job_manager),
) -> JobListResponse:
    """List all processing jobs."""
    jobs = job_manager.get_all_jobs()
    return JobListResponse(
        total=len(jobs),
        jobs=[_job_to_response(j) for j in jobs],
    )
