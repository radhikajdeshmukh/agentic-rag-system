from __future__ import annotations

from ..clients.ollama import ollama_chat


def verify_answer(answer: str, context: str) -> bool:
    prompt = (
        "Check if the answer is fully supported by the context.\n"
        "Respond with only YES or NO.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n"
    )
    verdict = ollama_chat(prompt).strip().upper()
    return verdict == "YES"
