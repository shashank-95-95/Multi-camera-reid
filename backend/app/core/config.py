"""
Configuration Module
====================

Centralised application settings using Pydantic Settings.

Settings are loaded from environment variables with sensible
defaults.  A ``.env`` file in the project root is automatically
read if present.

Usage::

    from app.core.config import settings
    print(settings.PROJECT_NAME)
"""

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration.

    All values can be overridden via environment variables or a
    ``.env`` file placed in the backend directory.

    Attributes:
        PROJECT_NAME: Display name shown in Swagger docs.
        VERSION: Semantic version string.
        DESCRIPTION: Project description for OpenAPI docs.
        DEBUG: Enable debug mode (verbose logging, auto-reload).
        LOG_LEVEL: Python logging level name.
        CORS_ORIGINS: Allowed CORS origins (comma-separated in env).
        DATABASE_URL: PostgreSQL connection string.  Not used yet
            but defined here for forward-compatibility.
        API_V1_PREFIX: URL prefix for versioned API routes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Project metadata ──────────────────────────────────────────────
    PROJECT_NAME: str = "Multi Camera Person ReID"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "AI-Based Multi-Camera Person Re-Identification System — "
        "Backend API"
    )
    DEBUG: bool = False

    # ── Logging ───────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── CORS ──────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["*"]

    # ── Database (Phase 2+) ───────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/person_reid"
    )

    # ── API versioning ────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"


settings = Settings()

# ── Project root (parent of backend/) ─────────────────────────────
# Used by the service layer to resolve AI engine imports and
# locate video files relative to the project root.
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
