from typing import List

import faiss
import numpy as np


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    text = text.strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore[call-arg]
    return index
