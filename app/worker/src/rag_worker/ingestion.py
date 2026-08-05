"""Worker ingestion orchestration before vector indexing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_core.embeddings import Embedding, EmbeddingProvider
from rag_core.ingestion import TextChunk

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
