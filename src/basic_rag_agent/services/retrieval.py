from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np

from ..clients.ollama import ollama_embed


def embed_texts(texts: List[str]) -> np.ndarray:
    vectors = ollama_embed(texts)
    return np.array(vectors, dtype="float32")


def retrieve_top_k(
    index: faiss.Index,
    chunks: List[str],
    query: str,
    k: int = 3,
) -> Tuple[List[str], List[int]]:
    query_vector = embed_texts([query])
    _, indices = index.search(query_vector, k)
    idxs = [int(i) for i in indices[0] if i != -1]
    return [chunks[i] for i in idxs], idxs
