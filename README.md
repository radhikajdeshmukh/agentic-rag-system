# Basic RAG Agent

Reliability-aware retrieval-augmented generation with verification and retry logic.

## Overview

This project is a Python CLI RAG application that:

- chunks a local knowledge source
- embeds chunks with Ollama-backed embeddings
- retrieves relevant evidence with FAISS
- generates answers constrained to retrieved context
- verifies groundedness and retries with a broader retrieval window

The repo uses `uv` for Python package management, a `src/` package layout, and `.env` for runtime configuration.

## Project layout

- `src/basic_rag_agent/cli.py` - packaged CLI entrypoint via `run_cli`
- `src/basic_rag_agent/config.py` - environment-driven settings
- `src/basic_rag_agent/pipeline.py` - RAG orchestration flow
- `src/basic_rag_agent/clients/ollama.py` - Ollama HTTP client
- `src/basic_rag_agent/services/` - ingestion, retrieval, generation, verification, and reflection services
- `src/basic_rag_agent/__main__.py` - module execution hook
- `data/knowledge.txt` - local knowledge base used by the CLI
- `docs/` - supporting documentation and notes

## Requirements

- Python 3.13+
- `uv`
- Ollama running locally with the configured models available

## Setup

```bash
make sync
```

Review `.env`, then run:

```bash
make run
```

Or run the installed entrypoint directly:

```bash
uv run basic-rag-agent
```

## Environment variables

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_EMBED_TIMEOUT_SECONDS=60
OLLAMA_CHAT_TIMEOUT_SECONDS=300
RAG_DATA_PATH=data/knowledge.txt
RAG_CHUNK_SIZE=400
RAG_CHUNK_OVERLAP=60
RAG_INITIAL_TOP_K=3
RAG_RETRY_TOP_K_STEP=2
RAG_MAX_TOP_K=8
RAG_MAX_RETRIES=2
RAG_PREVIEW_SOURCES=3
```

## Common commands

```bash
make sync
make run
make lock
make clean
```

## License

MIT. See `LICENSE`.
