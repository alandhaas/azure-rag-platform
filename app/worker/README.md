# RAG Worker

Azure Functions worker for asynchronous ingestion workflows.

The worker owns queue and blob-triggered orchestration while reusable RAG logic
stays in `packages/rag_core`.
