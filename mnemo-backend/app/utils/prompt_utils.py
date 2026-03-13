from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from utils.config import (
    FALLBACK_SYSTEM_PROMPT,
    OLLAMA_SYSTEM_PROMPT_FILE,
    PROMPTS_DIR,
)


def resolve_prompt_file_path(file_name: str) -> Path:
    candidate = (PROMPTS_DIR / file_name).resolve()
    if candidate != PROMPTS_DIR and PROMPTS_DIR not in candidate.parents:
        raise HTTPException(status_code=400, detail="system_prompt_file must be inside prompts directory.")
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"Prompt file not found: {file_name}")
    return candidate


def read_prompt_file(file_name: str) -> str:
    path = resolve_prompt_file_path(file_name)
    try:
        prompt_text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt file {file_name}: {exc}") from exc
    if not prompt_text:
        raise HTTPException(status_code=400, detail=f"Prompt file is empty: {file_name}")
    return prompt_text


def get_default_system_prompt() -> str:
    import os

    env_prompt = (os.getenv("OLLAMA_SYSTEM_PROMPT") or "").strip()
    if env_prompt:
        return env_prompt

    try:
        return read_prompt_file(OLLAMA_SYSTEM_PROMPT_FILE)
    except HTTPException:
        return FALLBACK_SYSTEM_PROMPT


def resolve_system_prompt(system_prompt: str | None, system_prompt_file: str | None) -> str:
    if system_prompt and system_prompt.strip():
        return system_prompt.strip()
    if system_prompt_file:
        return read_prompt_file(system_prompt_file)
    return get_default_system_prompt()