from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

# Import custom modules
from modules.stt_whisper import transcribe_audio
from modules.tts_kokoro import synthesize_and_encode_base64

logger = logging.getLogger("mnemo.app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


app = FastAPI(title="Mnemo Backend API")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TTS_URL = os.getenv("TTS_URL", "http://kokoro:8880")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "/app/prompts")).resolve()
OLLAMA_SYSTEM_PROMPT_FILE = os.getenv("OLLAMA_SYSTEM_PROMPT_FILE", "system_prompt.txt")
FALLBACK_SYSTEM_PROMPT = "You are Mnemo, a helpful and concise AI assistant."
OLLAMA_INIT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_INIT_TIMEOUT_SECONDS", "900"))

_MODEL_READY = asyncio.Event()
_MODEL_INIT_LOCK = asyncio.Lock()
_MODEL_INIT_TASK: asyncio.Task[None] | None = None
_MODEL_INIT_ERROR: str | None = None

_CLAUSE_BOUNDARY_PATTERN = re.compile(r"(?:[.!?;:](?:\s+|$)|\n+)")
_CLAUSE_MIN_WORDS = 5
_CLAUSE_MIN_CHARS = 24


class ChatRequest(BaseModel):
	text: str = Field(..., min_length=1)
	system_prompt: str | None = Field(default=None)
	system_prompt_file: str | None = Field(default=None)


class VoiceChatRequest(BaseModel):
	text: str | None = Field(default=None, description="Text input (alternative to audio_b64)")
	audio_b64: str | None = Field(default=None, description="Base64-encoded audio data (webm/wav/mp3)")
	voice: str = Field(default="af_heart")
	lang_code: str = Field(default="a", min_length=1, max_length=1)
	speed: float = Field(default=1.0, gt=0.0, le=3.0)
	audio_language: str | None = Field(default=None, description="Language code for STT (e.g., 'en', 'es')")
	system_prompt: str | None = Field(default=None)
	system_prompt_file: str | None = Field(default=None)
	
	@model_validator(mode='after')
	def validate_input(self):
		"""Validate that either text or audio_b64 is provided and not empty."""
		if not self.text and not self.audio_b64:
			raise ValueError("Either 'text' or 'audio_b64' must be provided")
		if self.text and self.audio_b64:
			raise ValueError("Cannot provide both 'text' and 'audio_b64'")
		if self.text is not None and not self.text.strip():
			raise ValueError("'text' cannot be empty")
		if self.audio_b64 is not None and not self.audio_b64.strip():
			raise ValueError("'audio_b64' cannot be empty")
		if self.system_prompt is not None and not self.system_prompt.strip():
			raise ValueError("'system_prompt' cannot be empty")
		if self.system_prompt_file is not None and not self.system_prompt_file.strip():
			raise ValueError("'system_prompt_file' cannot be empty")
		return self


def _resolve_prompt_file_path(file_name: str) -> Path:
	candidate = (PROMPTS_DIR / file_name).resolve()
	if candidate != PROMPTS_DIR and PROMPTS_DIR not in candidate.parents:
		raise HTTPException(status_code=400, detail="system_prompt_file must be inside prompts directory.")
	if not candidate.is_file():
		raise HTTPException(status_code=400, detail=f"Prompt file not found: {file_name}")
	return candidate


def _read_prompt_file(file_name: str) -> str:
	path = _resolve_prompt_file_path(file_name)
	try:
		prompt_text = path.read_text(encoding="utf-8").strip()
	except OSError as exc:
		raise HTTPException(status_code=500, detail=f"Failed to read prompt file {file_name}: {exc}") from exc
	if not prompt_text:
		raise HTTPException(status_code=400, detail=f"Prompt file is empty: {file_name}")
	return prompt_text


def _get_default_system_prompt() -> str:
	env_prompt = (os.getenv("OLLAMA_SYSTEM_PROMPT") or "").strip()
	if env_prompt:
		return env_prompt

	try:
		return _read_prompt_file(OLLAMA_SYSTEM_PROMPT_FILE)
	except HTTPException:
		return FALLBACK_SYSTEM_PROMPT


async def _ollama_has_model(client: httpx.AsyncClient) -> bool:
	resp = await client.get(f"{OLLAMA_URL}/api/tags")
	if resp.status_code >= 400:
		raise HTTPException(
			status_code=502,
			detail=f"Failed to query Ollama models ({resp.status_code}): {resp.text}",
		)

	models = resp.json().get("models", [])
	for model in models:
		name = str(model.get("name") or model.get("model") or "")
		if name == OLLAMA_MODEL:
			return True
	return False


async def _pull_ollama_model(client: httpx.AsyncClient) -> None:
	resp = await client.post(
		f"{OLLAMA_URL}/api/pull",
		json={"name": OLLAMA_MODEL, "stream": False},
	)
	if resp.status_code >= 400:
		raise HTTPException(
			status_code=502,
			detail=f"Failed to pull Ollama model {OLLAMA_MODEL} ({resp.status_code}): {resp.text}",
		)


async def _ensure_ollama_model_ready() -> None:
	global _MODEL_INIT_ERROR

	if _MODEL_READY.is_set():
		return

	async with _MODEL_INIT_LOCK:
		if _MODEL_READY.is_set():
			return

		loop = asyncio.get_running_loop()
		deadline = loop.time() + OLLAMA_INIT_TIMEOUT_SECONDS
		last_error = "Ollama model initialization timed out."

		while loop.time() < deadline:
			try:
				async with httpx.AsyncClient(timeout=120.0) as client:
					if not await _ollama_has_model(client):
						await _pull_ollama_model(client)
					_MODEL_READY.set()
					return
			except HTTPException as exc:
				last_error = str(exc.detail)
			except httpx.HTTPError as exc:
				last_error = f"Failed to reach Ollama: {exc}"

			await asyncio.sleep(3)

		_MODEL_INIT_ERROR = last_error
		raise HTTPException(status_code=502, detail=last_error)


async def _bootstrap_model_in_background() -> None:
	global _MODEL_INIT_ERROR
	try:
		await _ensure_ollama_model_ready()
	except HTTPException as exc:
		_MODEL_INIT_ERROR = str(exc.detail)
	except Exception as exc:  # pragma: no cover - defensive fallback
		_MODEL_INIT_ERROR = f"Unexpected model bootstrap error: {exc}"


def _start_model_bootstrap() -> None:
	global _MODEL_INIT_TASK
	if _MODEL_READY.is_set():
		return
	if _MODEL_INIT_TASK is None or _MODEL_INIT_TASK.done():
		_MODEL_INIT_TASK = asyncio.create_task(_bootstrap_model_in_background())


async def _require_model_ready_for_requests() -> None:
	if _MODEL_READY.is_set():
		return

	_start_model_bootstrap()

	if _MODEL_INIT_ERROR:
		raise HTTPException(
			status_code=503,
			detail=f"Model initialization failed: {_MODEL_INIT_ERROR}",
		)

	raise HTTPException(
		status_code=503,
		detail=f"Model {OLLAMA_MODEL} is being downloaded. Please retry shortly.",
	)


def _resolve_system_prompt(system_prompt: str | None, system_prompt_file: str | None) -> str:
	if system_prompt and system_prompt.strip():
		return system_prompt.strip()
	if system_prompt_file:
		return _read_prompt_file(system_prompt_file)
	return _get_default_system_prompt()


def _is_meaningful_clause(text: str) -> bool:
	stripped = text.strip()
	if not stripped:
		return False
	if len(stripped) >= _CLAUSE_MIN_CHARS:
		return True
	return len(stripped.split()) >= _CLAUSE_MIN_WORDS


def _pop_ready_clauses(buffer: str) -> tuple[list[str], str]:
	clauses: list[str] = []
	cursor = 0

	for match in _CLAUSE_BOUNDARY_PATTERN.finditer(buffer):
		end = match.end()
		candidate = buffer[cursor:end].strip()
		if not candidate:
			cursor = end
			continue

		if not _is_meaningful_clause(candidate):
			break

		clauses.append(candidate)
		cursor = end

	return clauses, buffer[cursor:]


async def _iter_meaningful_clauses(
	prompt: str,
	system_prompt: str | None = None,
	system_prompt_file: str | None = None,
) -> AsyncIterator[str]:
	await _require_model_ready_for_requests()

	resolved_system_prompt = _resolve_system_prompt(system_prompt, system_prompt_file)
	payload = {
		"model": OLLAMA_MODEL,
		"prompt": prompt,
		"system": resolved_system_prompt,
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
						buffer += piece
						ready_clauses, buffer = _pop_ready_clauses(buffer)
						for clause in ready_clauses:
							yield clause

					if chunk.get("done"):
						break
		except HTTPException:
			raise
		except httpx.HTTPError as exc:
			raise HTTPException(status_code=502, detail=f"Failed to call Ollama: {exc}") from exc

	tail = buffer.strip()
	if tail:
		yield tail


def _to_sse(event: str, payload: dict[str, str | int | bool]) -> bytes:
	data = json.dumps(payload, ensure_ascii=True)
	return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def _log_timing_breakdown(timing: dict[str, float | None], request_start_time: float) -> None:
	"""Log a detailed timing breakdown for the voice chat request."""
	request_end_time = time.time()
	total_time = request_end_time - request_start_time
	
	# Calculate phase durations and relative times
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
	
	# Calculate overlap (concurrent LLM and TTS)
	overlap_duration = 0.0
	if llm_duration and tts_duration and timing["tts_start"] and timing["llm_start"] and timing["llm_end"]:
		# TTS starts while LLM is running
		if timing["tts_start"] < timing["llm_end"]:
			overlap_duration = min(timing["llm_end"], timing["tts_end"] or request_end_time) - timing["tts_start"]
	
	# Log the breakdown
	logger.info("=" * 70)
	logger.info("VOICE CHAT TIMING BREAKDOWN")
	logger.info("=" * 70)
	if stt_duration is not None:
		logger.info(f"  STT Phase:")
		logger.info(f"    Start: {stt_start_rel:.3f}s  |  End: {stt_end_rel:.3f}s  |  Duration: {stt_duration:.3f}s")
	if llm_duration is not None:
		logger.info(f"  LLM Phase:")
		logger.info(f"    Start: {llm_start_rel:.3f}s  |  End: {llm_end_rel:.3f}s  |  Duration: {llm_duration:.3f}s")
	if tts_duration is not None:
		logger.info(f"  TTS Phase:")
		logger.info(f"    Start: {tts_start_rel:.3f}s  |  End: {tts_end_rel:.3f}s  |  Duration: {tts_duration:.3f}s")
	if overlap_duration > 0:
		logger.info(f"  Concurrent Overlap: {overlap_duration:.3f}s (LLM and TTS running simultaneously)")
	logger.info(f"  TOTAL: {total_time:.3f}s")
	logger.info("=" * 70)


async def _stream_ollama_tokens_to_queues(
	transcribed_text: str,
	system_prompt: str | None,
	system_prompt_file: str | None,
	clause_queue: asyncio.Queue[tuple[int, str] | None],
	event_queue: asyncio.Queue[tuple[str, dict[str, str | int | bool]]],
) -> None:
	try:
		resolved_system_prompt = _resolve_system_prompt(system_prompt, system_prompt_file)
		request_payload = {
			"model": OLLAMA_MODEL,
			"prompt": transcribed_text,
			"system": resolved_system_prompt,
			"stream": True,
		}

		buffer = ""
		clause_index = 0

		async with httpx.AsyncClient(timeout=None) as client:
			async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=request_payload) as resp:
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
						buffer += piece
						ready_clauses, buffer = _pop_ready_clauses(buffer)
						for clause in ready_clauses:
							clause_index += 1
							await clause_queue.put((clause_index, clause))

					if chunk.get("done"):
						break

		tail = buffer.strip()
		if tail:
			clause_index += 1
			await clause_queue.put((clause_index, tail))
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
	clause_queue: asyncio.Queue[tuple[int, str] | None],
	event_queue: asyncio.Queue[tuple[str, dict[str, str | int | bool]]],
) -> None:
	try:
		while True:
			clause_item = await clause_queue.get()
			if clause_item is None:
				break

			clause_index, clause_text = clause_item

			async for chunk_b64 in synthesize_and_encode_base64(
				text=clause_text,
				voice=voice,
				lang_code=lang_code,
				speed=speed,
			):
				await event_queue.put(("audio", {"index": clause_index, "chunk_b64": chunk_b64}))

			await event_queue.put(("audio_clause_done", {"index": clause_index, "text": clause_text}))
	except HTTPException as exc:
		await event_queue.put(("error", {"message": str(exc.detail)}))
	except Exception as exc:
		await event_queue.put(("error", {"message": f"Audio stream failed: {exc}"}))
	finally:
		await event_queue.put(("audio_done", {"done": True}))


async def _generate_llm_response(
	prompt: str,
	system_prompt: str | None = None,
	system_prompt_file: str | None = None,
) -> str:
	await _require_model_ready_for_requests()

	resolved_system_prompt = _resolve_system_prompt(system_prompt, system_prompt_file)

	payload = {
		"model": OLLAMA_MODEL,
		"prompt": prompt,
		"system": resolved_system_prompt,
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





async def _stream_voice_chat_events(payload: VoiceChatRequest) -> AsyncIterator[bytes]:
	"""Stream voice chat: audio/text input -> (STT) -> LLM -> TTS -> audio output."""
	request_start_time = time.time()
	logger.info(f"Voice chat request: audio={bool(payload.audio_b64)}, text={bool(payload.text)}, voice={payload.voice}")
	await _require_model_ready_for_requests()

	clause_queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
	event_queue: asyncio.Queue[tuple[str, dict[str, str | int | bool]]] = asyncio.Queue()
	
	# Timing tracking
	timing = {
		"stt_start": None,
		"stt_end": None,
		"llm_start": None,
		"llm_end": None,
		"tts_start": None,
		"tts_end": None,
	}

	try:
		# Determine input text (from direct text or from STT)
		if payload.audio_b64:
			# Step 1: Decode audio from base64
			logger.debug(f"Decoding base64 audio ({len(payload.audio_b64)} chars)")
			try:
				audio_data = base64.b64decode(payload.audio_b64)
				logger.info(f"Audio decoded: {len(audio_data)} bytes")
			except Exception as exc:
				logger.error(f"Failed to decode base64 audio: {exc}")
				yield _to_sse("error", {"message": f"Invalid base64 audio data: {exc}"})
				return

			# Step 2: Transcribe audio using Whisper STT
			timing["stt_start"] = time.time()
			logger.info(f"Starting STT transcription for {len(audio_data)} bytes")
			yield _to_sse("stt_started", {"status": "transcribing"})
			try:
				transcribed_text = await transcribe_audio(audio_data, payload.audio_language)
				timing["stt_end"] = time.time()
				logger.info(f"STT complete: '{transcribed_text}'")
				yield _to_sse("stt_done", {"text": transcribed_text})
			except HTTPException as exc:
				timing["stt_end"] = time.time()
				logger.error(f"STT failed: {exc.detail}")
				yield _to_sse("error", {"message": f"STT failed: {exc.detail}"})
				return
		else:
			# Use provided text directly
			logger.info(f"Using provided text: '{payload.text}'")
			transcribed_text = payload.text

		# Step 3: Stream LLM response
		timing["llm_start"] = time.time()
		logger.info(f"Starting LLM generation")
		yield _to_sse("llm_started", {"status": "generating"})
		text_task = asyncio.create_task(
			_stream_ollama_tokens_to_queues(
				transcribed_text,
				payload.system_prompt,
				payload.system_prompt_file,
				clause_queue,
				event_queue,
			)
		)

		# Step 4: Stream TTS audio output
		timing["tts_start"] = time.time()
		logger.info(f"Starting TTS synthesis")
		yield _to_sse("tts_started", {"status": "synthesizing"})
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
				timing["llm_end"] = time.time()
				logger.info("LLM generation complete")
			if event_name == "audio_done":
				audio_done = True
				timing["tts_end"] = time.time()
				logger.info("TTS synthesis complete")

			yield _to_sse(event_name, event_payload)

		logger.info("Voice chat request completed successfully")
		yield _to_sse("done", {"done": True})
		
		# Log timing breakdown
		_log_timing_breakdown(timing, request_start_time)

	except HTTPException as exc:
		timing["tts_end"] = timing["tts_end"] or time.time()
		timing["llm_end"] = timing["llm_end"] or time.time()
		timing["stt_end"] = timing["stt_end"] or time.time()
		_log_timing_breakdown(timing, request_start_time)
		logger.error(f"HTTPException in voice chat: {exc.detail}")
		yield _to_sse("error", {"message": str(exc.detail)})
	except Exception as exc:
		timing["tts_end"] = timing["tts_end"] or time.time()
		timing["llm_end"] = timing["llm_end"] or time.time()
		timing["stt_end"] = timing["stt_end"] or time.time()
		_log_timing_breakdown(timing, request_start_time)
		logger.exception(f"Unexpected error in voice chat: {exc}")
		yield _to_sse("error", {"message": f"Voice stream failed: {exc}"})
	finally:
		if 'text_task' in locals() and not text_task.done():
			text_task.cancel()
		if 'audio_task' in locals() and not audio_task.done():
			audio_task.cancel()
		with contextlib.suppress(Exception):
			if 'text_task' in locals() and 'audio_task' in locals():
				await asyncio.gather(text_task, audio_task)


@app.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}


@app.on_event("startup")
async def bootstrap_models() -> None:
	_start_model_bootstrap()


@app.post("/chat")
async def chat(payload: ChatRequest) -> dict[str, str]:
	response_text = await _generate_llm_response(
		payload.text,
		payload.system_prompt,
		payload.system_prompt_file,
	)
	return {"response": response_text}


@app.post("/voice/chat")
async def voice_chat(payload: VoiceChatRequest) -> StreamingResponse:
	return StreamingResponse(
		_stream_voice_chat_events(payload),
		media_type="text/event-stream",
	)


@app.post("/memories", status_code=501)
def memories_placeholder() -> dict[str, str]:
	return {"message": "Endpoint /memories is not implemented yet."}
