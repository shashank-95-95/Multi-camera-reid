"""
API Router Aggregator
=====================

Collects all route modules into a single :class:`APIRouter`
that is mounted by the application factory in ``main.py``.

To add a new route module:

1. Create ``app/api/routes/your_module.py`` with its own router.
2. Import and include it here.
"""

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.root import router as root_router

router = APIRouter()

router.include_router(root_router)
router.include_router(health_router)
router.include_router(jobs_router)
