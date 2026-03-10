# Building a Basic RAG System

In the previous section we discussed why Large Language Models hallucinate and why grounding responses in external knowledge is important.

In this section we will build a **basic Retrieval Augmented Generation (RAG) pipeline**.

The goal is simple:

> Instead of letting the model guess answers, we retrieve relevant information from documents and use that information to generate grounded responses.

---

# The RAG Pipeline

A typical RAG system consists of the following stages:

Documents → Chunking → Embeddings → Vector Database → Retrieval → Generation


Each stage plays a specific role in making the system reliable and scalable.

Let's go through them step by step.

---

# Step 1 — Preparing the Documents

RAG systems start with a set of documents that contain knowledge the model should use.

Examples include:

- internal documentation  
- research papers  
- product manuals  
- knowledge base articles  

In this tutorial we will use a simple text file:
data/knowledge.txt


We load the document:

```python
with open("data/knowledge.txt") as f:
    text = f.read()
```
However, we cannot embed the entire document at once.

Large documents must first be split into smaller chunks.

# Step 1 — Chunking the Text

Chunking divides large documents into manageable pieces that can be embedded and retrieved effectively.

Example Code:

```python
chunks = chunk_text(text, chunk_size=400, overlap=60)
```

Chunking improves retrieval quality because:

- LLMs have context size limits

- smaller chunks improve semantic search precision

- overlapping chunks preserve context continuity

Typical chunking strategy:

| Parameter  | Purpose                                  |
| ---------- | ---------------------------------------- |
| chunk_size | number of characters or tokens per chunk |
| overlap    | shared text between adjacent chunks      |

The overlap ensures that important sentences spanning chunk boundaries are not lost.

# Step 3 — Creating Embeddings

Once the text is chunked, we convert each chunk into a vector embedding.

Embeddings represent the semantic meaning of text as numerical vectors.

Example:

```python
chunk_vectors = embed_texts(chunks)
```
This transformation allows us to compare text based on semantic similarity rather than exact keywords.

For example, the following queries may produce similar embeddings:
```
"What year was Tesla founded?"
"When did Tesla start?"
```

Even though the wording is different, the meaning is similar.

# Step 4 — Building a Vector Index

After generating embeddings, we store them in a vector database.

In this tutorial we use FAISS (Facebook AI Similarity Search).

Example:

```python
index = build_faiss_index(chunk_vectors)
```

The vector index enables efficient similarity search across thousands or millions of document chunks.

Instead of scanning all documents, FAISS finds the nearest vectors to the query embedding.

# Step 5 — Retrieving Relevant Context

When a user submits a query, we follow this process:

1. Convert the query into an embedding
2. Search the vector database
3. Retrieve the most relevant chunks

Example:

```python
docs = retrieve(index, query)
```

The retrieved chunks represent the most relevant pieces of information for the user question.

# Step 6 — Generating a Grounded Answer

The retrieved chunks are combined and injected into the prompt before calling the language model.

Example:

```python
context = "\n".join([doc.page_content for doc in docs])

answer = generate_answer(query, context)
```

The model now generates an answer based on retrieved context rather than guessing from memory.

# Architecture Overview:

At this point our system follows the architecture below:


User Query
↓
Embed Query
↓
Vector Search (FAISS)
↓
Retrieve Relevant Chunks
↓
Inject Context into Prompt
↓
Generate Answer

This is the core idea behind Retrieval Augmented Generation.

# Why This Works

This approach reduces hallucinations because:

1. The model is given relevant context before generating

2. The system can use updated knowledge without retraining

3. The model operates within controlled knowledge boundaries

However, even with RAG, hallucinations can still occur.

For example:

- the model may misinterpret context

- it may combine unrelated chunks

- it may infer unsupported conclusions

To address this, we introduce grounded generation in the next section.