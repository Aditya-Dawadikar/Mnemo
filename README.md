# Mnemo

Mnemo is a local-first conversational assistant stack with:

- `mnemo-backend/app`: FastAPI orchestration API
- `mnemo-backend/ollama`: LLM runtime (default model: `llama3:8b`)
- `mnemo-backend/kokoro`: FastAPI TTS service (Kokoro)
- `mnemo-client`: lightweight HTML/CSS/JS chat UI

It supports:

- text chat (`/chat`)
- voice chat (`/voice/chat`) with simultaneous streamed text tokens and streamed audio chunks
- customizable system prompts from `app/prompts/*.txt`

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

Services:

- App API: `http://localhost:8000`
- Ollama: `http://localhost:11434`
- Kokoro: `http://localhost:8880`
- Postgres: `localhost:5432`

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

Request body:

```json
{
	"text": "Explain spaced repetition in simple terms.",
	"voice": "af_heart",
	"lang_code": "a",
	"speed": 1.0,
	"system_prompt": "optional inline prompt",
	"system_prompt_file": "optional-file.txt"
}
```

SSE events emitted by backend:

- `token`: incremental text token(s) from Ollama
- `clause`: clause boundary chosen for TTS
- `audio`: base64 WAV audio chunk for current clause
- `audio_clause_done`: no more audio chunks for that clause
- `text_done`: text generation finished
- `audio_done`: audio generation finished
- `done`: stream complete
- `error`: recoverable/final stream error information

The client renders `token` events to the chat bubble as they arrive and buffers `audio` chunks per clause. It queues playback only after `audio_clause_done`, so chunks play sequentially and smoothly.

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
