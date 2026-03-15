from __future__ import annotations

from pathlib import Path

from .clients.ollama import OllamaClientError, OllamaTimeoutError
from .config import get_settings
from .pipeline import RAGPipeline


def load_knowledge_text(data_path: str) -> str:
    return Path(data_path).read_text(encoding="utf-8").strip()


def print_context_preview(top_chunks: list[str], preview_count: int) -> None:
    print("\n--- Retrieved Context (preview) ---")
    for i, chunk in enumerate(top_chunks[:preview_count]):
        preview = chunk.replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        print(f"Source {i + 1}: {preview}")


def run_cli() -> None:
    settings = get_settings()
    text = load_knowledge_text(settings.data_path)

    if not text:
        print(f"{settings.data_path} is empty. Add some text and try again.")
        return

    try:
        pipeline = RAGPipeline.from_text(text, settings)
    except (OllamaTimeoutError, OllamaClientError) as exc:
        print(f"Startup failed: {exc}")
        return

    print(f"Indexed {len(pipeline.chunks)} chunks. Ready.\n")

    while True:
        query = input("Ask a question (or type 'exit'): ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        try:
            result = pipeline.answer(query)
        except OllamaTimeoutError as exc:
            print(f"\nRequest timed out: {exc}\n")
            continue
        except OllamaClientError as exc:
            print(f"\nRequest failed: {exc}\n")
            continue

        print_context_preview(result.top_chunks, settings.preview_sources)

        print("\n--- Answer ---")
        if result.grounded:
            print(result.answer)
        else:
            print("Not found in documents (or answer could not be grounded).")
            print("\nModel draft was:\n", result.answer)

        if result.reflections:
            print("\n--- Reflection ---")
            for reflection in result.reflections:
                print(reflection)

        print("\n")
