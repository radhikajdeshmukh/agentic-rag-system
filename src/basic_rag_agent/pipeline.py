from __future__ import annotations

from dataclasses import dataclass

import faiss

from .config import Settings
from .services.generation import generate_answer
from .services.ingest import build_faiss_index, chunk_text
from .services.reflection import analyze_failure
from .services.retrieval import embed_texts, retrieve_top_k
from .services.verification import verify_answer


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    grounded: bool
    top_chunks: list[str]
    chunk_indices: list[int]
    reflections: list[str]


class RAGPipeline:
    def __init__(
        self, index: faiss.Index, chunks: list[str], settings: Settings
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.settings = settings

    @classmethod
    def from_text(cls, text: str, settings: Settings) -> "RAGPipeline":
        chunks = chunk_text(
            text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        chunk_vectors = embed_texts(chunks)
        index = build_faiss_index(chunk_vectors)
        return cls(index=index, chunks=chunks, settings=settings)

    def _build_context(self, top_chunks: list[str], idxs: list[int]) -> str:
        parts = []
        for i, chunk in enumerate(top_chunks):
            parts.append(f"[Source {i + 1} | chunk {idxs[i]}]\n{chunk}")
        return "\n\n".join(parts)

    def answer(self, query: str) -> AnswerResult:
        k = self.settings.initial_top_k
        top_chunks, idxs = retrieve_top_k(self.index, self.chunks, query, k=k)
        reflections: list[str] = []

        for attempt in range(self.settings.max_retries + 1):
            context = self._build_context(top_chunks, idxs)
            answer = generate_answer(query, context)
            grounded = verify_answer(answer, context)

            if grounded:
                return AnswerResult(
                    answer=answer,
                    grounded=True,
                    top_chunks=top_chunks,
                    chunk_indices=idxs,
                    reflections=reflections,
                )

            if attempt < self.settings.max_retries:
                reflection = analyze_failure(query, answer, context)
                reflections.append(reflection)
                k = min(self.settings.max_top_k, k + self.settings.retry_top_k_step)
                top_chunks, idxs = retrieve_top_k(self.index, self.chunks, query, k=k)
                continue

            return AnswerResult(
                answer=answer,
                grounded=False,
                top_chunks=top_chunks,
                chunk_indices=idxs,
                reflections=reflections,
            )

        raise RuntimeError("Unreachable pipeline state")
