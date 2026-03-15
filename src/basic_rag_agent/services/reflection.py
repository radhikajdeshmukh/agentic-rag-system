from __future__ import annotations

from ..clients.ollama import ollama_chat


def analyze_failure(query: str, answer: str, context: str) -> str:
    prompt = (
        "You are a strict fact-checker for a RAG system.\n"
        "The answer below was rejected because it may contain claims not supported by the context.\n\n"
        "Your task:\n"
        "1) Say what part of the answer is unsupported.\n"
        "2) Say what information is missing from the context.\n"
        "Keep it short (3-6 lines).\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n"
    )
    return ollama_chat(prompt)
