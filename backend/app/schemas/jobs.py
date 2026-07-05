"""
Job Schemas
===========

Pydantic v2 request/response models for the processing and
job management API endpoints.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────


class ProcessRequest(BaseModel):
    """Request body for ``POST /process``."""

    video_paths: List[str] = Field(
        ...,
        min_length=1,
        examples=[["sample_videos/cam1.mp4", "sample_videos/cam2.mp4"]],
        description="Paths to input videos (absolute or relative to project root).",
    )
    reid_enabled: bool = Field(
        default=True,
        description="Enable cross-camera Person Re-Identification.",
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.1,
        le=1.0,
        description="YOLO detection confidence threshold.",
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.1,
        le=1.0,
        description="ReID cosine similarity threshold.",
    )
    reid_interval: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Frames between ReID re-extractions per track.",
    )


# ── Responses ─────────────────────────────────────────────────────────


class ProcessResponse(BaseModel):
    """Response from ``POST /process``."""

    job_id: str = Field(
        ...,
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
        description="UUID of the created processing job.",
    )
    status: str = Field(
        ...,
        examples=["queued"],
        description="Initial job status.",
    )


class JobResponse(BaseModel):
    """Detailed response from ``GET /jobs/{job_id}``."""

    job_id: str
    status: str
    progress: float = Field(
        ...,
        description="Completion percentage (0–100).",
    )
    cameras_total: int
    cameras_completed: int
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    processing_time: Optional[float] = Field(
        default=None,
        description="Elapsed processing time in seconds.",
    )
    output_directory: Optional[str] = None
    error_message: Optional[str] = None
    video_paths: List[str] = []
    reid_enabled: bool = True


class JobListResponse(BaseModel):
    """Response from ``GET /jobs``."""

    total: int = Field(..., description="Total number of jobs.")
    jobs: List[JobResponse]
