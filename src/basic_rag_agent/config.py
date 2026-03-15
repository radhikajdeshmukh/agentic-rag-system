from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str
    embed_timeout_seconds: int
    chat_timeout_seconds: int
    data_path: str
    chunk_size: int
    chunk_overlap: int
    initial_top_k: int
    retry_top_k_step: int
    max_top_k: int
    max_retries: int
    preview_sources: int


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_required_int(name: str) -> int:
    return int(_get_required(name))


def get_settings() -> Settings:
    return Settings(
        ollama_base_url=_get_required("OLLAMA_BASE_URL"),
        ollama_embed_model=_get_required("OLLAMA_EMBED_MODEL"),
        ollama_chat_model=_get_required("OLLAMA_CHAT_MODEL"),
        embed_timeout_seconds=_get_required_int("OLLAMA_EMBED_TIMEOUT_SECONDS"),
        chat_timeout_seconds=_get_required_int("OLLAMA_CHAT_TIMEOUT_SECONDS"),
        data_path=_get_required("RAG_DATA_PATH"),
        chunk_size=_get_required_int("RAG_CHUNK_SIZE"),
        chunk_overlap=_get_required_int("RAG_CHUNK_OVERLAP"),
        initial_top_k=_get_required_int("RAG_INITIAL_TOP_K"),
        retry_top_k_step=_get_required_int("RAG_RETRY_TOP_K_STEP"),
        max_top_k=_get_required_int("RAG_MAX_TOP_K"),
        max_retries=_get_required_int("RAG_MAX_RETRIES"),
        preview_sources=_get_required_int("RAG_PREVIEW_SOURCES"),
    )
