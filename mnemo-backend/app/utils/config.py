from __future__ import annotations

import logging
import os
from pathlib import Path


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("mnemo.app")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "/app/prompts")).resolve()
OLLAMA_SYSTEM_PROMPT_FILE = os.getenv("OLLAMA_SYSTEM_PROMPT_FILE", "system_prompt.txt")
FALLBACK_SYSTEM_PROMPT = "You are Mnemo, a helpful and concise AI assistant."
OLLAMA_INIT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_INIT_TIMEOUT_SECONDS", "900"))