from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_input(self) -> "VoiceChatRequest":
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