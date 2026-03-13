from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
from fastapi import HTTPException

from utils.config import OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_URL
from utils.prompt_utils import resolve_system_prompt
from utils.stream_utils import log_llm_stream_piece, pop_ready_segments


async def stream_ollama_segments(
    prompt: str,
    system_prompt: str | None = None,
    system_prompt_file: str | None = None,
) -> AsyncIterator[str]:
    resolved_system_prompt = resolve_system_prompt(system_prompt, system_prompt_file)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": resolved_system_prompt,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
        "stream": True,
    }

    buffer = ""

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama returned {resp.status_code}: {error_body.decode('utf-8', errors='ignore')}",
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    piece = str(chunk.get("response") or "")
                    if piece:
                        log_llm_stream_piece(piece)
                        buffer += piece
                        ready_segments, buffer = pop_ready_segments(buffer)
                        for segment in ready_segments:
                            yield segment

                    if chunk.get("done"):
                        break
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to call Ollama: {exc}") from exc

    tail = buffer.strip()
    if tail:
        yield tail


async def generate_llm_response(
    prompt: str,
    system_prompt: str | None = None,
    system_prompt_file: str | None = None,
) -> str:
    resolved_system_prompt = resolve_system_prompt(system_prompt, system_prompt_file)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": resolved_system_prompt,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to call Ollama: {exc}") from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    llm_text = (data.get("response") or "").strip()
    if not llm_text:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
    return llm_text