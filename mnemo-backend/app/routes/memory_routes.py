from __future__ import annotations

from fastapi import APIRouter, status

from controllers.memory_controller import memories_placeholder


router = APIRouter()


@router.post("/memories", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def memories() -> dict[str, str]:
    return memories_placeholder()