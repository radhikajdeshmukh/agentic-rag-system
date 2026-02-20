# ollama_client.py
from __future__ import annotations
import requests
from typing import List

OLLAMA_BASE_URL = "http://localhost:11434"

def ollama_embed(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=60,
        )
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return vectors

def ollama_chat(prompt: str, model: str = "mistral") -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"].strip()
