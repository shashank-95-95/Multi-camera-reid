"""
Root Route
==========

Provides the ``GET /`` endpoint that returns project metadata.
Useful as a quick smoke test and for API consumers to discover
documentation URLs.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import RootResponse

router = APIRouter(tags=["Root"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="Project information",
    description="Returns project name, version, and docs URL.",
)
async def root() -> RootResponse:
    """Return project metadata."""
    return RootResponse(
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        docs="/docs",
    )
