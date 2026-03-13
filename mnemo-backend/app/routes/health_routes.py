from __future__ import annotations

from fastapi import APIRouter

from controllers.health_controller import health_check


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return health_check()