from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Iterator

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from kokoro import KPipeline
from pydantic import BaseModel, Field

logger = logging.getLogger("mnemo.tts_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


app = FastAPI(title="Mnemo Kokoro Service")

# Cache pipelines by language to avoid reloading the model on every request.
_PIPELINES: dict[str, KPipeline] = {}


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(default="af_heart")
    lang_code: str = Field(default="a", min_length=1, max_length=1)
    speed: float = Field(default=1.0, gt=0.0, le=3.0)


def _get_pipeline(lang_code: str) -> KPipeline:
    logger.debug(f"Getting pipeline for lang_code: {lang_code}")
    pipeline = _PIPELINES.get(lang_code)
    if pipeline is None:
        logger.info(f"Loading new pipeline for lang_code: {lang_code}")
        pipeline = KPipeline(lang_code=lang_code)
        _PIPELINES[lang_code] = pipeline
        logger.info(f"Pipeline loaded and cached for lang_code: {lang_code}")
    else:
        logger.debug(f"Using cached pipeline for lang_code: {lang_code}")
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    logger.debug("Health check requested")
    return {"status": "ok"}


def _iter_buffer_chunks(buffer: BytesIO, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    buffer.seek(0)
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        yield chunk


@app.post("/synthesize")
def synthesize(payload: SynthesizeRequest, stream: bool = Query(default=False)) -> Response:
    logger.info(f"Synthesize request: voice={payload.voice}, lang={payload.lang_code}, speed={payload.speed}, stream={stream}, text_len={len(payload.text)}")
    try:
        logger.debug(f"Getting pipeline for lang_code={payload.lang_code}")
        pipeline = _get_pipeline(payload.lang_code)
        logger.debug(f"Pipeline loaded, generating segments")
        segments = pipeline(
            payload.text,
            voice=payload.voice,
            speed=payload.speed,
            split_pattern=r"\n+",
        )

        audio_chunks = [audio for _, _, audio in segments]
        if not audio_chunks:
            logger.warning(f"No audio generated")
            raise HTTPException(status_code=400, detail="No audio generated.")

        logger.info(f"Generated {len(audio_chunks)} audio chunks, merging")
        merged_audio = np.concatenate(audio_chunks)
        logger.info(f"Audio merged: {len(merged_audio)} samples")

        buffer = BytesIO()
        sf.write(buffer, merged_audio, samplerate=24000, format="WAV")
        logger.info(f"WAV written to buffer: {buffer.tell()} bytes")

        if stream:
            logger.info(f"Returning streamed response")
            return StreamingResponse(_iter_buffer_chunks(buffer), media_type="audio/wav")
        
        logger.info(f"Returning complete response")
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except HTTPException:
        logger.error(f"HTTPException in synthesize: {HTTPException}")
        raise
    except Exception as exc:
        logger.exception(f"Synthesis failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Kokoro synthesis failed: {exc}") from exc
