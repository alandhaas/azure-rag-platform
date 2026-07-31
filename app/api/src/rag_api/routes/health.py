"""Health probe routes."""

from typing import Literal, TypedDict

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(TypedDict):
    status: Literal["ok"]


@router.get(
    "/live",
    summary="Liveness probe",
    description="Returns ok when the API process is reachable.",
)
async def live() -> HealthResponse:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Returns ok when the API is ready to receive traffic.",
)
async def ready() -> HealthResponse:
    return {"status": "ok"}
