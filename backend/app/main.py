"""
FastAPI Backend — Main Application
===================================

Creates and configures the FastAPI application instance with:

- CORS middleware
- Structured logging
- API router registration
- OpenAPI/Swagger documentation
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Runs setup on startup and teardown on shutdown.
    Future phases will initialise database connections and
    AI engine resources here.
    """
    setup_logging()
    logger.info(
        "Starting %s v%s",
        settings.PROJECT_NAME,
        settings.VERSION,
    )
    yield
    logger.info("Shutting down %s", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """Application factory.

    Returns a fully-configured :class:`FastAPI` instance with
    middleware, routers, and OpenAPI metadata.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(api_router)

    return app


app = create_app()
