# rag-api

FastAPI application for the Azure RAG platform.

The API owns HTTP routing, request logging, health probes, and application-level
dependency construction. Reusable RAG behavior should live in `rag_core`.
