"""
Speech-to-Text module using Whisper API.
Handles audio transcription with streaming support.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

import httpx
from fastapi import HTTPException

logger = logging.getLogger("mnemo.stt")


WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:8001")


async def transcribe_audio(audio_data: bytes, language: str | None = None) -> str:
    """
    Transcribe audio using Whisper STT service.
    
    Args:
        audio_data: Raw audio bytes (webm, wav, mp3, etc.)
        language: Optional language code for transcription
        
    Returns:
        Transcribed text
        
    Raises:
        HTTPException: If transcription fails
    """
    logger.info(f"Transcribing audio: {len(audio_data)} bytes, language={language}")
    files = {"file": ("audio.webm", audio_data, "audio/webm")}
    data = {}
    if language:
        data["language"] = language
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            logger.debug(f"Sending request to {WHISPER_URL}/transcribe")
            resp = await client.post(
                f"{WHISPER_URL}/transcribe",
                files=files,
                data=data,
            )
        except httpx.HTTPError as exc:
            logger.error(f"Whisper connection failed: {exc}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach Whisper service: {exc}"
            ) from exc
    
    if resp.status_code >= 400:
        logger.error(f"Whisper returned {resp.status_code}: {resp.text}")
        raise HTTPException(
            status_code=502,
            detail=f"Whisper returned {resp.status_code}: {resp.text}",
        )
    
    result = resp.json()
    text = (result.get("text") or "").strip()
    
    if not text:
        logger.warning("No speech detected in audio")
        raise HTTPException(
            status_code=400,
            detail="No speech detected in audio"
        )
    
    logger.info(f"Transcription successful: '{text}'")
    return text


async def stream_transcribe_audio(
    audio_chunks: AsyncIterator[bytes],
    language: str | None = None,
) -> str:
    """
    Stream audio chunks to Whisper for transcription.
    Collects all chunks before sending to Whisper since it's not streaming-native.
    
    Args:
        audio_chunks: Async iterator of audio byte chunks
        language: Optional language code for transcription
        
    Returns:
        Transcribed text
        
    Raises:
        HTTPException: If transcription fails
    """
    # Collect all audio chunks
    audio_buffer = bytearray()
    async for chunk in audio_chunks:
        if chunk:
            audio_buffer.extend(chunk)
    
    if not audio_buffer:
        raise HTTPException(
            status_code=400,
            detail="No audio data received"
        )
    
    # Transcribe the complete audio
    return await transcribe_audio(bytes(audio_buffer), language)
