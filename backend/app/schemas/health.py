"""
Response Schemas
================

Pydantic v2 models used as FastAPI response schemas.
Provides compile-time validation and automatic OpenAPI
documentation generation.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``."""

    status: str = Field(
        ...,
        examples=["healthy"],
        description="Current health status of the backend.",
    )


class RootResponse(BaseModel):
    """Response schema for ``GET /``."""

    project: str = Field(
        ...,
        examples=["Multi Camera Person ReID"],
        description="Project display name.",
    )
    version: str = Field(
        ...,
        examples=["1.0.0"],
        description="Semantic version of the deployed backend.",
    )
    docs: str = Field(
        ...,
        examples=["/docs"],
        description="URL path to interactive Swagger documentation.",
    )
