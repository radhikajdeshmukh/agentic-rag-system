# Basic RAG Agent  
### Reliability-Aware Agentic Retrieval System

---

## Overview

This project implements a **reliability-aware retrieval-augmented agent** designed to reduce hallucinations through evaluation-driven control logic.

Rather than treating retrieval and generation as a linear pipeline, the system is structured as a **closed-loop architecture** where verification feedback influences subsequent retrieval behavior. This enables adaptive response correction instead of static one-shot generation.

The agent separates responsibilities into distinct phases:

- **Perception** — Semantic retrieval via embeddings + FAISS  
- **Reasoning** — Context-constrained generation using a local LLM (Ollama)  
- **Evaluation** — Groundedness verification against retrieved evidence  
- **Control** — Conditional retry with adaptive retrieval scope  

By introducing a feedback mechanism between evaluation and retrieval, the system shifts from prompt-driven generation to **decision-aware orchestration**.

The result is a minimal but structured example of an agentic AI system focused on:

- Hallucination mitigation  
- Retrieval reliability  
- Guardrail-enforced execution  
- System-level evaluation  

This repository demonstrates how small architectural decisions — particularly around feedback and control — materially improve the reliability of LLM-based systems.
