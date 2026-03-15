from __future__ import annotations

from typing import List

import requests

from ..config import get_settings


class OllamaClientError(RuntimeError):
    pass


class OllamaTimeoutError(OllamaClientError):
    pass


def ollama_embed(texts: List[str], model: str | None = None) -> List[List[float]]:
    settings = get_settings()
    embed_model = model or settings.ollama_embed_model
    vectors = []
    for text in texts:
        try:
            response = requests.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": embed_model, "prompt": text},
                timeout=settings.embed_timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise OllamaTimeoutError(
                "Embedding request to Ollama timed out. Increase OLLAMA_EMBED_TIMEOUT_SECONDS or verify Ollama is healthy."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaClientError(f"Embedding request failed: {exc}") from exc
        vectors.append(response.json()["embedding"])
    return vectors


def ollama_chat(prompt: str, model: str | None = None) -> str:
    settings = get_settings()
    chat_model = model or settings.ollama_chat_model
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": chat_model, "prompt": prompt, "stream": False},
            timeout=settings.chat_timeout_seconds,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise OllamaTimeoutError(
            "Chat request to Ollama timed out. Increase OLLAMA_CHAT_TIMEOUT_SECONDS or verify Ollama is healthy."
        ) from exc
    except requests.RequestException as exc:
        raise OllamaClientError(f"Chat request failed: {exc}") from exc
    return response.json()["response"].strip()
