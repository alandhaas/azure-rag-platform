# rag-core

Shared Python package for reusable RAG logic consumed by the API and worker.

This package owns domain-level RAG concerns such as embeddings, retrieval,
ingestion, prompt handling, and shared models. It should stay independent from
FastAPI route code, Azure Functions entry points, and infrastructure definitions.
