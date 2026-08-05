# Local development

This project can run locally with Google AI Studio for model calls, Qdrant for
vectors, and Azurite for Blob/Queue/Table storage.

## Commands

```bash
make install
make check
make services-up
make api
make worker
make compose-down
```

For now, run only Qdrant and Azurite in Docker with `make services-up`, then run
the API and local queue worker in separate terminals with `make api` and
`make worker`. The local worker uses the same ingestion pipeline as the Azure
Functions trigger, but avoids Azure Functions Core Tools' local Python/gRPC
host. Use `make worker-functions` only when you specifically want to test the
Functions host.

## Upload and query

Upload a PDF for indexing:

```bash
curl -sS -X POST http://localhost:8000/documents \
  -H "x-request-id: local-test-1" \
  -F "file=@/path/to/document.pdf;type=application/pdf"
```

Check ingestion status with the returned `document_id`:

```bash
curl -sS http://localhost:8000/documents/{document_id}
```

After the worker logs `document_chunks_indexed`, query the indexed chunks:

```bash
curl -sS -X POST http://localhost:8000/queries/retrieval \
  -H "content-type: application/json" \
  -H "x-request-id: local-test-2" \
  -d '{"text":"What is this document about?","top_k":5}'
```

## Local services

| Service | URL |
| --- | --- |
| API | `http://localhost:${API_PORT:-8000}` |
| Worker health | `http://localhost:${WORKER_PORT:-7071}/api/health/live` |
| Qdrant HTTP | `http://localhost:${QDRANT_HTTP_PORT:-6333}` |
| Qdrant gRPC | `localhost:${QDRANT_GRPC_PORT:-6334}` |
| Azurite Blob | `http://localhost:${AZURITE_BLOB_PORT:-10000}` |
| Azurite Queue | `http://localhost:${AZURITE_QUEUE_PORT:-10001}` |
| Azurite Table | `http://localhost:${AZURITE_TABLE_PORT:-10002}` |

Ollama is still available as a fallback provider:

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Environment variables

Copy `.env.example` to `.env` for local development, then set `GEMINI_API_KEY`
to your Google AI Studio API key. The host commands use `localhost` endpoints.
Docker Compose uses the `COMPOSE_*` variables where a container needs a
different address.
