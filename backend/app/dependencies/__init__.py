"""
Dependencies Module
===================

FastAPI dependency injection callables.

Provides reusable ``Depends()`` factories for settings,
services, and the job manager.  All singletons are created
lazily on first use and reused for subsequent requests.

Usage in a route::

    from fastapi import Depends
    from app.dependencies import get_reid_service

    @router.post("/process")
    async def process(
        service: ReidService = Depends(get_reid_service),
    ):
        ...
"""

from app.core.config import Settings, settings
from app.services.job_manager import JobManager
from app.services.reid_service import ReidService

# ── Singletons ────────────────────────────────────────────────────────
_job_manager: JobManager | None = None
_reid_service: ReidService | None = None


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return settings


def get_job_manager() -> JobManager:
    """Return the shared :class:`JobManager` singleton.

    Thread-safe: the ``JobManager`` uses internal locking.
    """
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def get_reid_service() -> ReidService:
    """Return the shared :class:`ReidService` singleton."""
    global _reid_service
    if _reid_service is None:
        _reid_service = ReidService()
    return _reid_service
