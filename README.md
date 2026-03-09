# Mnemo

Mnemo is a local-first conversational assistant stack with:

- `mnemo-backend/app`: FastAPI orchestration API
- `mnemo-backend/ollama`: LLM runtime (default model: `llama3:8b`)
- `mnemo-backend/kokoro`: FastAPI TTS service (Kokoro)
- `mnemo-backend/whisper`: FastAPI realtime STT service (faster-whisper)
- `mnemo-client`: lightweight HTML/CSS/JS chat UI

It supports:

- **Text Chat** (`/chat`): Text input → LLM → Text response
- **Voice Chat** (`/voice/chat`): Voice/text input → (STT if needed) → LLM → TTS → Streamed audio + text output with low-latency playback
- Customizable system prompts from `app/prompts/*.txt`

The voice chat endpoint streams audio chunks and text clauses as they're generated, enabling the client to start playback before the full response completes.

## Project Structure

```text
mnemo-backend/
	docker-compose.yaml
	app/
		main.py
		prompts/
			system_prompt.txt
	kokoro/
		main.py
mnemo-client/
	index.html
	app.js
	styles.css
```

## Quick Start

### 1. Start services

From `mnemo-backend`:

```powershell
docker compose up -d --build
```

Services exposed:

- App API: `http://localhost:8000` (FastAPI with `/chat` and `/voice/chat` endpoints)
- Ollama: `http://localhost:11434` (internal LLM service)
- Kokoro: `http://localhost:8880` (internal TTS service)
- Whisper: `http://localhost:8001` (internal STT service)
- Postgres: `localhost:5432` (internal database)

All services except the App API are internal and accessed by the backend only. Client requests go to the App API.

### 2. First run model pull behavior

The app automatically checks for and pulls `llama3:8b` from Ollama when missing.

- While downloading, `/chat` and `/voice/chat` return `503` with a retry message.
- Once ready, requests succeed normally.

Optional manual pre-pull:

```powershell
docker exec -it mnemo_ollama ollama pull llama3:8b
```

### 3. Open the client

Serve `mnemo-client` from any static server, then open in browser.

Example (if Node is available):

```powershell
cd mnemo-client
npx serve .
```

Set API Base URL in UI to `http://localhost:8000`.

## Prompt Management

Default prompt file:

- `mnemo-backend/app/prompts/system_prompt.txt`

Resolution priority used by the backend:

1. request `system_prompt`
2. request `system_prompt_file`
3. env `OLLAMA_SYSTEM_PROMPT`
4. file `OLLAMA_SYSTEM_PROMPT_FILE` (default `system_prompt.txt`)
5. internal fallback prompt

You can add additional files under `mnemo-backend/app/prompts/` and pass them via request.

## API Overview

### Health

- `GET /health`

Response:

```json
{"status":"ok"}
```

### Text Chat

- `POST /chat`

Request body:

```json
{
	"text": "Who are you?",
	"system_prompt": "optional inline prompt",
	"system_prompt_file": "optional-file.txt"
}
```

Response:

```json
{
	"response": "..."
}
```

### Voice Chat (Text + Audio Streaming)

- `POST /voice/chat`
- Response `Content-Type`: `text/event-stream`

**Request body** (either `text` or `audio_b64`, not both):

```json
{
	"text": "Explain spaced repetition in simple terms.",
	"voice": "af_heart",
	"lang_code": "a",
	"speed": 1.0,
	"audio_language": "en",
	"system_prompt": "optional inline prompt",
	"system_prompt_file": "optional-file.txt"
}
```

Or with audio input:

```json
{
	"audio_b64": "base64-encoded-audio-data",
	"voice": "af_heart",
	"lang_code": "a",
	"speed": 1.0,
	"audio_language": "en",
	"system_prompt": "optional inline prompt",
	"system_prompt_file": "optional-file.txt"
}
```

**Complete SSE event flow**:

1. **stt_started** (if audio input provided)
   ```json
   {"status": "transcribing"}
   ```

2. **stt_done** (if audio input provided)
   ```json
   {"text": "transcribed user speech"}
   ```

3. **llm_started**
   ```json
   {"status": "generating"}
   ```

4. **tts_started**
   ```json
   {"status": "synthesizing"}
   ```

5. **audio** (multiple events as clauses are synthesized)
   ```json
   {
     "index": 1,
     "chunk_b64": "base64-encoded-audio-chunk"
   }
   ```

6. **audio_clause_done** (after each clause is fully synthesized)
   ```json
   {
     "index": 1,
     "text": "First clause of response."
   }
   ```

7. **text_done** (after LLM finishes generating)
   ```json
   {"done": true}
   ```

8. **audio_done** (after all audio synthesis completes)
   ```json
   {"done": true}
   ```

9. **done** (final completion marker)
   ```json
   {"done": true}
   ```

10. **error** (if any error occurs)
    ```json
    {"message": "Error description"}
    ```

**Event Processing**:
- The client accumulates audio chunks (`audio` events) per clause index
- When `audio_clause_done` arrives, the client queues the complete audio for playback
- Audio chunks play sequentially after each clause completes (not waiting for full response)
- Text is buffered and displayed as each clause is synthesized
- This enables near-real-time playback while maintaining coherent speech

### Realtime Speech-To-Text

To use only STT without LLM/TTS processing, send `audio_b64` without expecting a full voice chat response. The backend handles this via the Voice Chat endpoint with automatic transcription.

## Useful Commands

From `mnemo-backend`:

```powershell
# Start/rebuild
docker compose up -d --build

# Stop
docker compose stop

# Tail logs
docker compose logs -f app
docker compose logs -f ollama
docker compose logs -f kokoro
```

## Notes

- CORS is enabled in the app for local browser development.
- `kokoro` service exposes `/health` and `/synthesize` (`?stream=true` supported).
- Postgres service is running but memory endpoints are placeholders for now.
