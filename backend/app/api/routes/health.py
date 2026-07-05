"""
Health Route
============

Provides the ``GET /health`` endpoint used by load balancers,
orchestrators, and monitoring tools to verify the backend is
responsive.
"""

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current health status of the backend.",
)
async def health_check() -> HealthResponse:
    """Return a simple health status.

    Future phases may extend this to check database connectivity,
    GPU availability, and AI engine readiness.
    """
    return HealthResponse(status="healthy")
