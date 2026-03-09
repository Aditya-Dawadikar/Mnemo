from __future__ import annotations

from io import BytesIO
from typing import Iterator

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from kokoro import KPipeline
from pydantic import BaseModel, Field


app = FastAPI(title="Mnemo Kokoro Service")

# Cache pipelines by language to avoid reloading the model on every request.
_PIPELINES: dict[str, KPipeline] = {}


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(default="af_heart")
    lang_code: str = Field(default="a", min_length=1, max_length=1)
    speed: float = Field(default=1.0, gt=0.0, le=3.0)


def _get_pipeline(lang_code: str) -> KPipeline:
    pipeline = _PIPELINES.get(lang_code)
    if pipeline is None:
        pipeline = KPipeline(lang_code=lang_code)
        _PIPELINES[lang_code] = pipeline
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
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
    try:
        pipeline = _get_pipeline(payload.lang_code)
        segments = pipeline(
            payload.text,
            voice=payload.voice,
            speed=payload.speed,
            split_pattern=r"\n+",
        )

        audio_chunks = [audio for _, _, audio in segments]
        if not audio_chunks:
            raise HTTPException(status_code=400, detail="No audio generated.")

        merged_audio = np.concatenate(audio_chunks)

        buffer = BytesIO()
        sf.write(buffer, merged_audio, samplerate=24000, format="WAV")

        if stream:
            return StreamingResponse(_iter_buffer_chunks(buffer), media_type="audio/wav")

        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kokoro synthesis failed: {exc}") from exc
