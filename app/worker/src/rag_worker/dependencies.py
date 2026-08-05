"""Worker dependency construction."""

from __future__ import annotations

from rag_core.embeddings import OllamaEmbeddingProvider
from rag_core.ingestion import ChunkingConfig, TextChunker

from rag_worker.blob_loader import BlobDocumentLoader
from rag_worker.config import WorkerSettings, get_settings
from rag_worker.ingestion import DocumentEmbeddingPipeline
from rag_worker.text_extraction import DocumentTextExtractor


def create_document_embedding_pipeline(
    settings: WorkerSettings | None = None,
) -> DocumentEmbeddingPipeline:
    settings = settings or get_settings()
    return DocumentEmbeddingPipeline(
        blob_loader=BlobDocumentLoader(settings.storage_connection_string()),
        text_extractor=DocumentTextExtractor(),
        text_chunker=TextChunker(
            ChunkingConfig(
                max_chars=settings.document_chunk_max_chars,
                overlap_chars=settings.document_chunk_overlap_chars,
            )
        ),
        embedding_provider=OllamaEmbeddingProvider(settings.ollama_embedding_config()),
    )
