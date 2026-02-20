# from dotenv import load_dotenv
# import os
# from openai import OpenAI

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# import numpy as np
# from ingest import chunk_text, build_faiss_index
# from retriever import embed_texts, retrieve_top_k
# from generator import generate_answer
# from verifier import verify_answer

# def main():
#     with open("data/knowledge.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     chunks = chunk_text(text, chunk_size=400, overlap=60)
#     chunk_vectors = embed_texts(chunks)
#     index = build_faiss_index(chunk_vectors)

#     query = input("Ask a question: ").strip()

#     top_chunks, _ = retrieve_top_k(index, chunks, query, k=3)
#     context = "\n\n---\n\n".join(top_chunks)

#     answer = generate_answer(query, context)
#     is_valid = verify_answer(answer, context)

#     if is_valid:
#         print("\nAnswer:", answer)
#     else:
#         print("\nAnswer rejected (not grounded in documents)")
#     # print("Draft:", answer)


# if __name__ == "__main__":
    # main()

# ---------------------------- Code with retry logic --------------------------
# main.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from ingest import chunk_text, build_faiss_index
from retriever import embed_texts, retrieve_top_k
from generator import generate_answer
from verifier import verify_answer
from reflector import analyze_failure


MAX_RETRIES = 2


def main() -> None:
    # 1) Load text
    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("data/knowledge.txt is empty. Add some text and try again.")
        return

    # 2) Build index
    chunks = chunk_text(text, chunk_size=400, overlap=60)
    chunk_vectors = embed_texts(chunks)
    index = build_faiss_index(chunk_vectors)

    print(f"Indexed {len(chunks)} chunks. Ready.\n")

    # 3) REPL loop
    while True:
        query = input("Ask a question (or type 'exit'): ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        # First retrieval (k=3)
        k = 3
        top_chunks, idxs = retrieve_top_k(index, chunks, query, k=k)

        final_answer = None
        final_context = None
        grounded = False

        for attempt in range(MAX_RETRIES + 1):
            # Build context with simple source tags (helps later for citations)
            context = ""
            for i, ch in enumerate(top_chunks):
                context += f"[Source {i+1} | chunk {idxs[i]}]\n{ch}\n\n"

            answer = generate_answer(query, context)
            grounded = verify_answer(answer, context)

            if grounded:
                final_answer = answer
                final_context = context
                break

            # If not grounded and we still have retries, reflect + expand retrieval
            if attempt < MAX_RETRIES:
                print(f"\nAttempt {attempt+1} failed verification. Retrying...\n")
                feedback = analyze_failure(query, answer, context)
                print("Reflection:\n", feedback, "\n")

                # Improve retrieval by increasing k (simple but effective)
                k = min(8, k + 2)
                top_chunks, idxs = retrieve_top_k(index, chunks, query, k=k)
            else:
                final_answer = answer
                final_context = context

        # 4) Print result
        print("\n--- Retrieved Context (preview) ---")
        for i, ch in enumerate(top_chunks[:3]):
            preview = ch.replace("\n", " ")
            preview = preview[:200] + ("..." if len(preview) > 200 else "")
            print(f"Source {i+1}: {preview}")

        print("\n--- Answer ---")
        if grounded:
            print(final_answer)
        else:
            print("Not found in documents (or answer could not be grounded).")
            print("\nModel draft was:\n", final_answer)

        print("\n")


if __name__ == "__main__":
    main()