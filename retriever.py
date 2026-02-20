# retriever.py
from __future__ import annotations
import numpy as np
import faiss
from typing import List, Tuple

from ollama_client import ollama_embed

def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = ollama_embed(texts, model="nomic-embed-text")
    return np.array(vecs, dtype="float32")

def retrieve_top_k(
    index: faiss.Index,
    chunks: List[str],
    query: str,
    k: int = 3
) -> Tuple[List[str], List[int]]:
    q_vec = embed_texts([query])
    distances, indices = index.search(q_vec, k)
    idxs = [int(i) for i in indices[0] if i != -1]
    return [chunks[i] for i in idxs], idxs
