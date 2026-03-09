"""
Text-to-Speech module using Kokoro API.
Handles TTS synthesis with streaming support.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import AsyncIterator

import httpx
from fastapi import HTTPException

logger = logging.getLogger("mnemo.tts")


TTS_URL = os.getenv("TTS_URL", "http://kokoro:8880")


async def synthesize_audio(
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
) -> AsyncIterator[bytes]:
    """
    Synthesize text to speech using Kokoro TTS service.
    Streams audio chunks as they become available.
    
    Args:
        text: Text to synthesize
        voice: Voice identifier (default: af_heart)
        lang_code: Language code (default: a for American English)
        speed: Speech speed multiplier (0.0 < speed <= 3.0)
        
    Yields:
        Audio chunks as bytes
        
    Raises:
        HTTPException: If synthesis fails
    """
    logger.info(f"Synthesizing audio: voice={voice}, lang={lang_code}, speed={speed}, text_len={len(text)}")
    payload = {
        "text": text,
        "voice": voice,
        "lang_code": lang_code,
        "speed": speed,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            logger.debug(f"Sending request to {TTS_URL}/synthesize")
            async with client.stream(
                "POST",
                f"{TTS_URL}/synthesize",
                params={"stream": "true"},
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    logger.error(f"Kokoro returned {resp.status_code}: {error_body.decode('utf-8', errors='ignore')}")
                    raise HTTPException(
                        status_code=502,
                        detail=f"Kokoro returned {resp.status_code}: {error_body.decode('utf-8', errors='ignore')}",
                    )

                chunk_count = 0
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        chunk_count += 1
                        yield chunk
                
                logger.info(f"TTS synthesis complete: {chunk_count} chunks")
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            logger.error(f"Kokoro connection failed: {exc}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to call Kokoro: {exc}"
            ) from exc


async def synthesize_and_encode_base64(
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
) -> AsyncIterator[str]:
    """
    Synthesize text to speech and encode chunks as base64.
    Useful for JSON/SSE transport.
    
    Args:
        text: Text to synthesize
        voice: Voice identifier
        lang_code: Language code
        speed: Speech speed multiplier
        
    Yields:
        Base64-encoded audio chunks
        
    Raises:
        HTTPException: If synthesis fails
    """
    async for chunk in synthesize_audio(text, voice, lang_code, speed):
        yield base64.b64encode(chunk).decode("ascii")


async def synthesize_full_audio(
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
) -> bytes:
    """
    Synthesize text to speech and return complete audio.
    
    Args:
        text: Text to synthesize
        voice: Voice identifier
        lang_code: Language code
        speed: Speech speed multiplier
        
    Returns:
        Complete audio as bytes
        
    Raises:
        HTTPException: If synthesis fails
    """
    audio_buffer = bytearray()
    async for chunk in synthesize_audio(text, voice, lang_code, speed):
        audio_buffer.extend(chunk)
    
    return bytes(audio_buffer)
