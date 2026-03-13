from __future__ import annotations

import json
import re
import time

from utils.config import logger


_SEGMENT_BOUNDARY_PATTERN = re.compile(r"(?:[.!?;:]+(?:['\")\]]*)?(?=\s|$)|\n+)")


def pop_ready_segments(buffer: str) -> tuple[list[str], str]:
    segments: list[str] = []
    cursor = 0

    for match in _SEGMENT_BOUNDARY_PATTERN.finditer(buffer):
        segment = buffer[cursor:match.end()].strip()
        cursor = match.end()
        if segment:
            segments.append(segment)

    return segments, buffer[cursor:]


def to_sse(event: str, payload: dict[str, str | int | bool]) -> bytes:
    data = json.dumps(payload, ensure_ascii=True)
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def log_llm_stream_piece(piece: str) -> None:
    received_at = time.time()
    logger.info(
        f"LLM stream piece received | timestamp={received_at:.6f} | piece={json.dumps(piece, ensure_ascii=True)}"
    )


def log_timing_breakdown(timing: dict[str, float | None], request_start_time: float) -> None:
    request_end_time = time.time()
    total_time = request_end_time - request_start_time

    stt_duration = None
    stt_start_rel = None
    stt_end_rel = None
    if timing["stt_start"] and timing["stt_end"]:
        stt_duration = timing["stt_end"] - timing["stt_start"]
        stt_start_rel = timing["stt_start"] - request_start_time
        stt_end_rel = timing["stt_end"] - request_start_time

    llm_duration = None
    llm_start_rel = None
    llm_end_rel = None
    if timing["llm_start"] and timing["llm_end"]:
        llm_duration = timing["llm_end"] - timing["llm_start"]
        llm_start_rel = timing["llm_start"] - request_start_time
        llm_end_rel = timing["llm_end"] - request_start_time

    tts_duration = None
    tts_start_rel = None
    tts_end_rel = None
    if timing["tts_start"] and timing["tts_end"]:
        tts_duration = timing["tts_end"] - timing["tts_start"]
        tts_start_rel = timing["tts_start"] - request_start_time
        tts_end_rel = timing["tts_end"] - request_start_time

    overlap_duration = 0.0
    if llm_duration and tts_duration and timing["tts_start"] and timing["llm_start"] and timing["llm_end"]:
        if timing["tts_start"] < timing["llm_end"]:
            overlap_duration = min(timing["llm_end"], timing["tts_end"] or request_end_time) - timing["tts_start"]

    logger.info("=" * 70)
    logger.info("VOICE CHAT TIMING BREAKDOWN")
    logger.info("=" * 70)
    if stt_duration is not None:
        logger.info("  STT Phase:")
        logger.info(f"    Start: {stt_start_rel:.3f}s  |  End: {stt_end_rel:.3f}s  |  Duration: {stt_duration:.3f}s")
    if llm_duration is not None:
        logger.info("  LLM Phase:")
        logger.info(f"    Start: {llm_start_rel:.3f}s  |  End: {llm_end_rel:.3f}s  |  Duration: {llm_duration:.3f}s")
    if tts_duration is not None:
        logger.info("  TTS Phase:")
        logger.info(f"    Start: {tts_start_rel:.3f}s  |  End: {tts_end_rel:.3f}s  |  Duration: {tts_duration:.3f}s")
    if overlap_duration > 0:
        logger.info(f"  Concurrent Overlap: {overlap_duration:.3f}s (LLM and TTS running simultaneously)")
    logger.info(f"  TOTAL: {total_time:.3f}s")
    logger.info("=" * 70)