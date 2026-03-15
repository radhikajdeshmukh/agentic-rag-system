from __future__ import annotations

from ..clients.ollama import ollama_chat


def generate_answer(query: str, context: str) -> str:
    prompt = (
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say exactly: Not found in documents.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n"
    )
    return ollama_chat(prompt)
