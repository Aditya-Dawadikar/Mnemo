# Mnemo Backend - Voice Chat API Design

## Architecture Overview

The Mnemo backend implements a fully streaming voice-to-voice conversational AI system with the following flow:

```
User Voice Input → Whisper STT → LLM (Ollama) → Kokoro TTS → Audio Output
```

All components are designed for streaming to minimize latency and provide real-time responsiveness.

## Module Structure

### 1. `modules/stt_whisper.py`
**Purpose**: Speech-to-Text using Whisper API

**Key Functions**:
- `transcribe_audio(audio_data: bytes, language: str | None) -> str`
  - Transcribes audio bytes to text
  - Supports multiple audio formats (webm, wav, mp3)
  - Optional language specification for better accuracy

- `stream_transcribe_audio(audio_chunks: AsyncIterator[bytes], language: str | None) -> str`
  - Handles streaming audio input
  - Accumulates chunks before sending to Whisper (since Whisper is not natively streaming)

**Environment Variables**:
- `WHISPER_URL`: URL of the Whisper service (default: `http://whisper:8881`)

### 2. `modules/tts_kokoro.py`
**Purpose**: Text-to-Speech using Kokoro API

**Key Functions**:
- `synthesize_audio(text: str, voice: str, lang_code: str, speed: float) -> AsyncIterator[bytes]`
  - Streams audio chunks as they're synthesized
  - Real-time audio generation

- `synthesize_and_encode_base64(text: str, voice: str, lang_code: str, speed: float) -> AsyncIterator[str]`
  - Streams base64-encoded audio chunks
  - Ideal for JSON/SSE transport

- `synthesize_full_audio(text: str, voice: str, lang_code: str, speed: float) -> bytes`
  - Returns complete audio (non-streaming)

**Environment Variables**:
- `TTS_URL`: URL of the Kokoro TTS service (default: `http://kokoro:8880`)

## API Endpoints

### POST `/voice/chat`

**Purpose**: Complete voice-to-voice conversation flow

**Request Body** (JSON):
```json
{
  "audio_b64": "base64-encoded-audio-data",
  "voice": "af_heart",
  "lang_code": "a",
  "speed": 1.0,
  "audio_language": "en",
  "system_prompt": "Optional custom system prompt",
  "system_prompt_file": "Optional prompt file name"
}
```

**Request Fields**:
- `audio_b64` (required): Base64-encoded audio input (webm/wav/mp3)
- `voice` (optional): TTS voice identifier (default: "af_heart")
- `lang_code` (optional): TTS language code (default: "a" for American English)
- `speed` (optional): TTS speed multiplier, 0.0 < speed <= 3.0 (default: 1.0)
- `audio_language` (optional): STT language hint (e.g., "en", "es")
- `system_prompt` (optional): Custom system prompt for LLM
- `system_prompt_file` (optional): Reference to prompt file in `/app/prompts/`

**Response**: Server-Sent Events (SSE) stream

**Event Flow**:

1. **stt_started**
   ```json
   {"status": "transcribing"}
   ```

2. **stt_done**
   ```json
   {"text": "transcribed user speech"}
   ```

3. **llm_started**
   ```json
   {"status": "generating"}
   ```

4. **audio** (multiple events as LLM generates and TTS synthesizes)
   ```json
   {
     "index": 1,
     "chunk_b64": "base64-encoded-audio-chunk"
   }
   ```

5. **audio_clause_done** (after each clause)
   ```json
   {
     "index": 1,
     "text": "First clause of response."
   }
   ```

6. **text_done**
   ```json
   {"done": true}
   ```

7. **audio_done**
   ```json
   {"done": true}
   ```

8. **done** (final event)
   ```json
   {"done": true}
   ```

9. **error** (if any error occurs)
   ```json
   {"message": "Error description"}
   ```

## Streaming Flow Details

### Phase 1: Speech-to-Text (STT)
1. Client sends base64-encoded audio
2. Backend decodes audio
3. Audio sent to Whisper service
4. Transcribed text returned

**Status Events**: `stt_started`, `stt_done`

### Phase 2: LLM Processing
1. Transcribed text sent to Ollama LLM
2. LLM streams response tokens
3. Tokens accumulated into "meaningful clauses" (sentences/phrases)
4. Clauses queued for TTS as they complete

**Status Events**: `llm_started`, `text_done`

**Clause Detection**:
- Buffers text until a punctuation boundary is reached (`.!?;:`)
- Enables low-latency TTS streaming

### Phase 3: Text-to-Speech (TTS)
1. Each clause sent to Kokoro TTS immediately
2. Audio chunks streamed as they're synthesized
3. Chunks base64-encoded and sent via SSE
4. Client can start playback before full response completes

**Status Events**: `tts_started`, `audio`, `audio_clause_done`, `audio_done`

## Concurrent Processing

The LLM and TTS stages run **concurrently**:
- While LLM generates clause N+1, TTS synthesizes clause N
- Minimizes overall latency
- Provides responsive user experience

## Error Handling

Errors at any stage result in an `error` event:
```json
event: error
data: {"message": "Detailed error description"}
```

**Common Error Scenarios**:
- Invalid base64 audio data
- STT service unavailable
- No speech detected in audio
- LLM service unavailable
- TTS service unavailable
- Audio synthesis failure

## Environment Configuration

**Core Services**:
- `OLLAMA_URL`: LLM service URL (default: `http://ollama:11434`)
- `WHISPER_URL`: STT service URL (default: `http://whisper:8881`)
- `TTS_URL`: TTS service URL (default: `http://kokoro:8880`)

**LLM Settings**:
- `OLLAMA_MODEL`: Model name (default: `llama3.2:3b`)
- `OLLAMA_SYSTEM_PROMPT_FILE`: Default prompt file (default: `system_prompt.txt`)
- `OLLAMA_SYSTEM_PROMPT`: Optional inline default prompt

**Prompts**:
- `PROMPTS_DIR`: Prompt files directory (default: `/app/prompts`)

## Client Integration Example

```javascript
async function sendVoiceMessage(audioBlob) {
  // Convert audio to base64
  const audioBuffer = await audioBlob.arrayBuffer();
  const audioB64 = btoa(
    String.fromCharCode(...new Uint8Array(audioBuffer))
  );

  // Send request
  const response = await fetch('/voice/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio_b64: audioB64,
      voice: 'af_heart',
      speed: 1.0
    })
  });

  // Process SSE stream
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const audioChunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const events = parseSSE(text);

    for (const event of events) {
      if (event.type === 'stt_done') {
        console.log('User said:', event.data.text);
      }
      if (event.type === 'audio') {
        const audioData = atob(event.data.chunk_b64);
        audioChunks.push(audioData);
        // Play audio incrementally
        playAudioChunk(audioData);
      }
    }
  }
}
```

## Performance Characteristics

**Latency Breakdown**:
1. STT: ~1-3 seconds (depends on audio length)
2. LLM first token: ~100-500ms
3. LLM clause complete: ~1-2 seconds
4. TTS first chunk: ~200-500ms

**Total Time to First Audio**: ~2-4 seconds from voice input

**Streaming Benefits**:
- User hears response while LLM still generating
- Perceived latency much lower than batch processing
- Better conversational flow

## Security Considerations

- Audio data transmitted as base64 (consider encryption for production)
- System prompts validated against directory traversal
- File paths sanitized
- CORS configured (currently allows all origins - restrict in production)

## Future Enhancements

1. **True Streaming STT**: Use streaming Whisper implementation
2. **Voice Activity Detection**: Auto-detect speech boundaries
3. **Multi-turn Context**: Conversation history management
4. **Voice Cloning**: Custom voice profiles
5. **Interrupt Handling**: Allow user to interrupt AI response
6. **WebSocket Support**: Alternative to SSE for bidirectional streaming
