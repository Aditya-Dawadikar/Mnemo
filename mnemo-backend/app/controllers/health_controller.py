from __future__ import annotations


def health_check() -> dict[str, str]:
    return {"status": "ok"}