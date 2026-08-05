# Local development

This project can run locally with Google AI Studio for model calls and Qdrant,
Azurite, the API, and the worker in Docker Compose.

## Commands

```bash
make install
make check
make compose-up
make compose-logs
make compose-down
```

Run the API directly from the workspace with `make api`. Run the Azure Functions
worker directly with `make worker`.

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