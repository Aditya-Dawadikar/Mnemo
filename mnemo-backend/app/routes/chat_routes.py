from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from controllers.chat_controller import handle_chat, stream_voice_chat_events
from schemas.requests import ChatRequest, VoiceChatRequest


router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest) -> dict[str, str]:
    return await handle_chat(payload)


@router.post("/voice/chat")
async def voice_chat(payload: VoiceChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_voice_chat_events(payload),
        media_type="text/event-stream",
    )