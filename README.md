# azure-rag-platform
Cloud-native RAG platform for document processing, embeddings, vector search, and AI-powered Q&amp;A on Azure.

## Local development

Use Google AI Studio for models and Docker Compose for the local platform
services. Run Qdrant and Azurite in Docker, then run the API and worker in
terminals:

```bash
make install
make check
make services-up
make api
make worker
```

Upload a PDF with `POST /documents`, then check ingestion progress with
`GET /documents/{document_id}`.

See [local development](docs/local-development.md) for service URLs, environment
variables, and compose commands.
