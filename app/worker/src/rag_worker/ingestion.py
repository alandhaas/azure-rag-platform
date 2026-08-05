"""Worker ingestion orchestration before vector indexing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from rag_core.embeddings import Embedding, EmbeddingProvider
from rag_core.ingestion import TextChunk
from rag_core.vectorstore import VectorPoint

from rag_worker.blob_loader import LoadedBlobDocument
from rag_worker.commands import IngestionCommand


class BlobLoader(Protocol):
    def load(self, *, document_id: str, blob_uri: str) -> LoadedBlobDocument: ...


class TextExtractor(Protocol):
    def extract_text(self, document: LoadedBlobDocument) -> str: ...


class Chunker(Protocol):
    def chunk_text(
        self,
        text: str,
        *,
        source_metadata: dict[str, object],
    ) -> list[TextChunk]: ...


class VectorIndexer(Protocol):
    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...


@dataclass(frozen=True, slots=True)
class EmbeddedDocumentChunk:
    """A source text chunk with its generated embedding."""

    chunk: TextChunk
    embedding: Embedding


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    """Result of loading, parsing, chunking, and embedding one document."""

    document_id: str
    blob_uri: str
    chunks: tuple[EmbeddedDocumentChunk, ...]


class DocumentEmbeddingPipeline:
    """Load a document, chunk extracted text, and generate chunk embeddings."""

    def __init__(
        self,
        *,
        blob_loader: BlobLoader,
        text_extractor: TextExtractor,
        text_chunker: Chunker,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._blob_loader = blob_loader
        self._text_extractor = text_extractor
        self._text_chunker = text_chunker
        self._embedding_provider = embedding_provider

    async def process(self, command: IngestionCommand) -> EmbeddedDocument:
        document = self._blob_loader.load(
            document_id=command.document_id,
            blob_uri=command.blob_uri,
        )
        text = self._text_extractor.extract_text(document)
        chunks = self._text_chunker.chunk_text(
            text,
            source_metadata={
                "document_id": command.document_id,
                "blob_uri": command.blob_uri,
                "content_type": document.content_type or "",
            },
        )
        embeddings = await self._embedding_provider.embed(
            [chunk.content for chunk in chunks]
        )
        return EmbeddedDocument(
            document_id=command.document_id,
            blob_uri=command.blob_uri,
            chunks=_combine_chunks_and_embeddings(chunks, embeddings),
        )


class DocumentIndexingPipeline:
    """Generate document chunk embeddings and upsert them into the vector store."""

    def __init__(
        self,
        *,
        embedding_pipeline: DocumentEmbeddingPipeline,
        vector_store: VectorIndexer,
    ) -> None:
        self._embedding_pipeline = embedding_pipeline
        self._vector_store = vector_store

    async def process(self, command: IngestionCommand) -> EmbeddedDocument:
        document = await self._embedding_pipeline.process(command)
        await self._vector_store.upsert(embedded_document_to_vector_points(document))
        return document


def embedded_document_to_vector_points(document: EmbeddedDocument) -> list[VectorPoint]:
    """Convert embedded chunks into deterministic vector-store points."""
    return [
        VectorPoint(
            id=_point_id(document, embedded_chunk.chunk),
            embedding=embedded_chunk.embedding,
            payload={
                **embedded_chunk.chunk.metadata,
                "document_id": document.document_id,
                "blob_uri": document.blob_uri,
                "content": embedded_chunk.chunk.content,
            },
        )
        for embedded_chunk in document.chunks
    ]


def _combine_chunks_and_embeddings(
    chunks: Sequence[TextChunk],
    embeddings: Sequence[Embedding],
) -> tuple[EmbeddedDocumentChunk, ...]:
    if len(chunks) != len(embeddings):
        raise RuntimeError("Embedding count must match document chunk count.")
    return tuple(
        EmbeddedDocumentChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    )


def _point_id(document: EmbeddedDocument, chunk: TextChunk) -> str:
    stable_key = (
        f"{document.document_id}:{chunk.chunk_index}:"
        f"{chunk.char_start}:{chunk.char_end}"
    )
    return str(uuid5(NAMESPACE_URL, stable_key))
