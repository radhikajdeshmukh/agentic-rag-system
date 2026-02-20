# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# def build_index(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
#     docs = splitter.create_documents([text])
#     vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
#     return vectorstore

import numpy as np
import faiss
from typing import List, Tuple

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
    # embeddings: (n, d) float32
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

