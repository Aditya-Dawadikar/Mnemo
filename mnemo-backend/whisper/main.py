from __future__ import annotations

import io
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel


logger = logging.getLogger("mnemo.whisper")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

TARGET_SAMPLE_RATE = int(os.getenv("WHISPER_TARGET_SAMPLE_RATE", "16000"))
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE")

WINDOW_SECONDS = float(os.getenv("WHISPER_WINDOW_SECONDS", "14"))
MAX_BUFFER_SECONDS = float(os.getenv("WHISPER_MAX_BUFFER_SECONDS", "90"))
INFER_INTERVAL_SECONDS = float(os.getenv("WHISPER_INFER_INTERVAL_SECONDS", "0.45"))
STABILITY_MARGIN_SECONDS = float(os.getenv("WHISPER_STABILITY_MARGIN_SECONDS", "0.8"))


app = FastAPI(title="Mnemo Whisper Service")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@dataclass
class StreamState:
	audio: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
	audio_start_seconds: float = 0.0
	emitted_until_seconds: float = 0.0
	emitted_tokens: list[str] = field(default_factory=list)
	last_infer_monotonic: float = 0.0
	sample_rate: int = TARGET_SAMPLE_RATE
	language: str | None = DEFAULT_LANGUAGE


_MODEL: WhisperModel | None = None


def _get_model() -> WhisperModel:
	global _MODEL
	if _MODEL is None:
		logger.info(
			"Loading faster-whisper model=%s device=%s compute_type=%s",
			MODEL_SIZE,
			MODEL_DEVICE,
			MODEL_COMPUTE_TYPE,
		)
		_MODEL = WhisperModel(MODEL_SIZE, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE_TYPE)
	return _MODEL


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
	if source_rate == target_rate:
		return audio
	if audio.size == 0:
		return audio

	ratio = target_rate / float(source_rate)
	new_length = max(1, int(round(audio.size * ratio)))
	source_index = np.linspace(0, audio.size - 1, num=audio.size, dtype=np.float64)
	target_index = np.linspace(0, audio.size - 1, num=new_length, dtype=np.float64)
	resampled = np.interp(target_index, source_index, audio).astype(np.float32)
	return resampled


def _decode_wav_blob(blob: bytes) -> tuple[np.ndarray, int]:
	with wave.open(io.BytesIO(blob), "rb") as wav_file:
		sample_rate = int(wav_file.getframerate())
		channels = int(wav_file.getnchannels())
		sample_width = int(wav_file.getsampwidth())
		raw = wav_file.readframes(wav_file.getnframes())

	if sample_width != 2:
		raise ValueError("Only 16-bit PCM WAV blobs are supported.")

	pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
	if channels > 1:
		pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)
	return pcm, sample_rate


def _decode_audio_blob(blob: bytes, sample_rate_hint: int) -> np.ndarray:
	if not blob:
		return np.empty((0,), dtype=np.float32)

	if blob.startswith(b"RIFF") and b"WAVE" in blob[:16]:
		audio, source_rate = _decode_wav_blob(blob)
		return _resample_linear(audio, source_rate, TARGET_SAMPLE_RATE)

	usable_len = len(blob) - (len(blob) % 2)
	if usable_len <= 0:
		return np.empty((0,), dtype=np.float32)

	pcm = np.frombuffer(blob[:usable_len], dtype=np.int16).astype(np.float32) / 32768.0
	return _resample_linear(pcm, sample_rate_hint, TARGET_SAMPLE_RATE)


def _append_audio(state: StreamState, chunk: np.ndarray) -> None:
	if chunk.size == 0:
		return

	state.audio = np.concatenate((state.audio, chunk))
	max_samples = int(MAX_BUFFER_SECONDS * TARGET_SAMPLE_RATE)
	if state.audio.size <= max_samples:
		return

	overflow = state.audio.size - max_samples
	state.audio = state.audio[overflow:]
	state.audio_start_seconds += overflow / float(TARGET_SAMPLE_RATE)


def _iter_stable_words(state: StreamState, finalize: bool) -> list[dict[str, float | str]]:
	if state.audio.size == 0:
		return []

	current_audio_end = state.audio_start_seconds + (state.audio.size / float(TARGET_SAMPLE_RATE))
	window_start = max(state.audio_start_seconds, current_audio_end - WINDOW_SECONDS)
	window_offset_samples = int((window_start - state.audio_start_seconds) * TARGET_SAMPLE_RATE)
	window_audio = state.audio[window_offset_samples:]

	if window_audio.size < int(0.25 * TARGET_SAMPLE_RATE):
		return []

	model = _get_model()
	segments, _ = model.transcribe(
		window_audio,
		language=state.language,
		word_timestamps=True,
		beam_size=1,
		temperature=0.0,
		condition_on_previous_text=False,
		vad_filter=True,
	)

	stability_cutoff = current_audio_end if finalize else (current_audio_end - STABILITY_MARGIN_SECONDS)
	if stability_cutoff < state.emitted_until_seconds:
		return []

	emitted: list[dict[str, float | str]] = []
	for segment in segments:
		for word in segment.words:
			text = (word.word or "").strip()
			if not text:
				continue

			start_abs = window_start + float(word.start)
			end_abs = window_start + float(word.end)
			if end_abs <= state.emitted_until_seconds:
				continue
			if end_abs > stability_cutoff:
				continue

			emitted.append({"token": text, "start": start_abs, "end": end_abs})
			state.emitted_until_seconds = end_abs
			state.emitted_tokens.append(text)

	return emitted


async def _send_json(websocket: WebSocket, payload: dict[str, object]) -> None:
	await websocket.send_text(json.dumps(payload, ensure_ascii=True))


@app.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}


@app.on_event("startup")
def warmup_model() -> None:
	_get_model()


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
	await websocket.accept()
	state = StreamState(last_infer_monotonic=time.monotonic())

	await _send_json(
		websocket,
		{
			"event": "ready",
			"sample_rate": TARGET_SAMPLE_RATE,
			"input_format": "binary message as 16-bit PCM mono chunks or WAV blobs",
		},
	)

	try:
		while True:
			message = await websocket.receive()
			blob = message.get("bytes")
			text = message.get("text")

			if blob is not None:
				try:
					decoded = _decode_audio_blob(blob, state.sample_rate)
				except Exception as exc:
					await _send_json(websocket, {"event": "error", "message": f"Invalid audio blob: {exc}"})
					continue

				_append_audio(state, decoded)
				now = time.monotonic()
				if now - state.last_infer_monotonic >= INFER_INTERVAL_SECONDS:
					for token in _iter_stable_words(state, finalize=False):
						await _send_json(websocket, {"event": "token", **token})
					state.last_infer_monotonic = now
				continue

			if text is None:
				continue

			try:
				command = json.loads(text)
			except json.JSONDecodeError:
				await _send_json(websocket, {"event": "error", "message": "Text frames must be valid JSON."})
				continue

			event_name = str(command.get("event") or "").lower()
			if event_name == "start":
				rate = int(command.get("sample_rate") or TARGET_SAMPLE_RATE)
				state.sample_rate = max(8000, min(48000, rate))
				lang = command.get("language")
				state.language = str(lang).strip() if lang else DEFAULT_LANGUAGE
				await _send_json(websocket, {"event": "started", "sample_rate": state.sample_rate, "language": state.language})
				continue

			if event_name == "flush":
				for token in _iter_stable_words(state, finalize=False):
					await _send_json(websocket, {"event": "token", **token})
				continue

			if event_name == "end":
				for token in _iter_stable_words(state, finalize=True):
					await _send_json(websocket, {"event": "token", **token})
				await _send_json(websocket, {"event": "done", "text": " ".join(state.emitted_tokens)})
				break

			await _send_json(websocket, {"event": "error", "message": f"Unsupported event: {event_name}"})
	except WebSocketDisconnect:
		logger.info("WebSocket disconnected")
		return
	finally:
		if websocket.client_state.name != "DISCONNECTED":
			await websocket.close()
