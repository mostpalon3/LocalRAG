import os
from pathlib import Path


LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"))
EMBED_MODEL = os.getenv("EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))  # nomic-embed-text limit is 512
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "sample-lancedb"))
OLLAMA_URL = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
EMBED_DIMENSIONS = int(os.getenv("OLLAMA_EMBED_DIMENSIONS", "768"))