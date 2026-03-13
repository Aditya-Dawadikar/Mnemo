from __future__ import annotations

import asyncio
import base64
from typing import AsyncIterator

from fastapi import HTTPException

from modules.stt_whisper import transcribe_audio
from modules.tts_kokoro import synthesize_and_encode_base64
from schemas.requests import ChatRequest, VoiceChatRequest
from utils.config import logger
from utils.ollama import generate_llm_response, stream_ollama_segments
from utils.stream_utils import log_timing_breakdown, to_sse


EventPayload = dict[str, str | int | bool]
ClauseQueue = asyncio.Queue[tuple[int, str] | None]
EventQueue = asyncio.Queue[tuple[str, EventPayload]]


async def handle_chat(payload: ChatRequest) -> dict[str, str]:
    response_text = await generate_llm_response(
        payload.text,
        payload.system_prompt,
        payload.system_prompt_file,
    )
    return {"response": response_text}


async def _stream_ollama_segments_to_queue(
    prompt: str,
    system_prompt: str | None,
    system_prompt_file: str | None,
    clause_queue: ClauseQueue,
    event_queue: EventQueue,
) -> None:
    clause_index = 0
    try:
        async for clause in stream_ollama_segments(prompt, system_prompt, system_prompt_file):
            clause_index += 1
            await clause_queue.put((clause_index, clause))
    except HTTPException as exc:
        await event_queue.put(("error", {"message": str(exc.detail)}))
    except Exception as exc:
        await event_queue.put(("error", {"message": f"Text stream failed: {exc}"}))
    finally:
        await clause_queue.put(None)
        await event_queue.put(("text_done", {"done": True}))


async def _stream_kokoro_audio_to_events(
    voice: str,
    lang_code: str,
    speed: float,
    clause_queue: ClauseQueue,
    event_queue: EventQueue,
) -> None:
    tts_request_count = 0
    try:
        while True:
            clause_item = await clause_queue.get()
            if clause_item is None:
                break

            clause_index, clause_text = clause_item
            tts_request_count += 1
            clause_for_log = " ".join(clause_text.split())
            if len(clause_for_log) > 240:
                clause_for_log = f"{clause_for_log[:237]}..."

            tts_start = asyncio.get_running_loop().time()
            stream_start: float | None = None
            stream_end: float | None = None
            chunk_count = 0

            logger.info(
                f"TTS request {tts_request_count} started | clause_index={clause_index} | "
                f"tts_start={tts_start:.6f} | clause='{clause_for_log}'"
            )

            async for chunk_b64 in synthesize_and_encode_base64(
                text=clause_text,
                voice=voice,
                lang_code=lang_code,
                speed=speed,
            ):
                if stream_start is None:
                    stream_start = asyncio.get_running_loop().time()
                    logger.info(
                        f"TTS request {tts_request_count} stream started | clause_index={clause_index} | "
                        f"stream_start={stream_start:.6f}"
                    )

                stream_end = asyncio.get_running_loop().time()
                chunk_count += 1
                await event_queue.put(("audio", {"index": clause_index, "chunk_b64": chunk_b64}))

            tts_end = asyncio.get_running_loop().time()
            logger.info(
                f"TTS request {tts_request_count} finished | clause_index={clause_index} | "
                f"tts_start={tts_start:.6f} | tts_end={tts_end:.6f} | "
                f"stream_start={(f'{stream_start:.6f}' if stream_start is not None else 'None')} | "
                f"stream_end={(f'{stream_end:.6f}' if stream_end is not None else 'None')} | "
                f"chunks={chunk_count} | clause='{clause_for_log}'"
            )

            await event_queue.put(("audio_clause_done", {"index": clause_index, "text": clause_text}))
    except HTTPException as exc:
        await event_queue.put(("error", {"message": str(exc.detail)}))
    except Exception as exc:
        await event_queue.put(("error", {"message": f"Audio stream failed: {exc}"}))
    finally:
        logger.info(f"TTS requests made for voice chat request: {tts_request_count}")
        await event_queue.put(("audio_done", {"done": True}))


async def stream_voice_chat_events(payload: VoiceChatRequest) -> AsyncIterator[bytes]:
    request_start_time = asyncio.get_running_loop().time()
    logger.info(f"Voice chat request: audio={bool(payload.audio_b64)}, text={bool(payload.text)}, voice={payload.voice}")

    clause_queue: ClauseQueue = asyncio.Queue()
    event_queue: EventQueue = asyncio.Queue()
    timing: dict[str, float | None] = {
        "stt_start": None,
        "stt_end": None,
        "llm_start": None,
        "llm_end": None,
        "tts_start": None,
        "tts_end": None,
    }

    text_task: asyncio.Task[None] | None = None
    audio_task: asyncio.Task[None] | None = None

    try:
        if payload.audio_b64:
            logger.debug(f"Decoding base64 audio ({len(payload.audio_b64)} chars)")
            try:
                audio_data = base64.b64decode(payload.audio_b64)
                logger.info(f"Audio decoded: {len(audio_data)} bytes")
            except Exception as exc:
                logger.error(f"Failed to decode base64 audio: {exc}")
                yield to_sse("error", {"message": f"Invalid base64 audio data: {exc}"})
                return

            timing["stt_start"] = asyncio.get_running_loop().time()
            logger.info(f"Starting STT transcription for {len(audio_data)} bytes")
            yield to_sse("stt_started", {"status": "transcribing"})
            try:
                transcribed_text = await transcribe_audio(audio_data, payload.audio_language)
                timing["stt_end"] = asyncio.get_running_loop().time()
                logger.info(f"STT complete: '{transcribed_text}'")
                yield to_sse("stt_done", {"text": transcribed_text})
            except HTTPException as exc:
                timing["stt_end"] = asyncio.get_running_loop().time()
                logger.error(f"STT failed: {exc.detail}")
                yield to_sse("error", {"message": f"STT failed: {exc.detail}"})
                return
        else:
            transcribed_text = payload.text or ""
            logger.info(f"Using provided text: '{transcribed_text}'")

        timing["llm_start"] = asyncio.get_running_loop().time()
        logger.info("Starting LLM generation")
        yield to_sse("llm_started", {"status": "generating"})
        text_task = asyncio.create_task(
            _stream_ollama_segments_to_queue(
                transcribed_text,
                payload.system_prompt,
                payload.system_prompt_file,
                clause_queue,
                event_queue,
            )
        )

        timing["tts_start"] = asyncio.get_running_loop().time()
        logger.info("Starting TTS synthesis")
        yield to_sse("tts_started", {"status": "synthesizing"})
        audio_task = asyncio.create_task(
            _stream_kokoro_audio_to_events(
                payload.voice,
                payload.lang_code,
                payload.speed,
                clause_queue,
                event_queue,
            )
        )

        text_done = False
        audio_done = False

        while not (text_done and audio_done):
            event_name, event_payload = await event_queue.get()
            if event_name == "text_done":
                text_done = True
                timing["llm_end"] = asyncio.get_running_loop().time()
                logger.info("LLM generation complete")
            if event_name == "audio_done":
                audio_done = True
                timing["tts_end"] = asyncio.get_running_loop().time()
                logger.info("TTS synthesis complete")

            yield to_sse(event_name, event_payload)

        logger.info("Voice chat request completed successfully")
        yield to_sse("done", {"done": True})
        log_timing_breakdown(timing, request_start_time)
    except HTTPException as exc:
        now = asyncio.get_running_loop().time()
        timing["tts_end"] = timing["tts_end"] or now
        timing["llm_end"] = timing["llm_end"] or now
        timing["stt_end"] = timing["stt_end"] or now
        log_timing_breakdown(timing, request_start_time)
        logger.error(f"HTTPException in voice chat: {exc.detail}")
        yield to_sse("error", {"message": str(exc.detail)})
    except Exception as exc:
        now = asyncio.get_running_loop().time()
        timing["tts_end"] = timing["tts_end"] or now
        timing["llm_end"] = timing["llm_end"] or now
        timing["stt_end"] = timing["stt_end"] or now
        log_timing_breakdown(timing, request_start_time)
        logger.exception(f"Unexpected error in voice chat: {exc}")
        yield to_sse("error", {"message": f"Voice stream failed: {exc}"})
    finally:
        pending_tasks = [task for task in (text_task, audio_task) if task is not None and not task.done()]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)