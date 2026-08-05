from collections.abc import Sequence

from rag_core.embeddings import Embedding
from rag_core.ingestion import ChunkingConfig, TextChunker
from rag_worker.blob_loader import LoadedBlobDocument
from rag_worker.commands import IngestionCommand
from rag_worker.ingestion import DocumentEmbeddingPipeline
from rag_worker.text_extraction import DocumentTextExtractor


async def test_document_embedding_pipeline_loads_chunks_and_embeds_document() -> None:
    embedding_provider = FakeEmbeddingProvider()
    pipeline = DocumentEmbeddingPipeline(
        blob_loader=FakeBlobLoader(),
        text_extractor=DocumentTextExtractor(),
        text_chunker=TextChunker(ChunkingConfig(max_chars=12, overlap_chars=2)),
        embedding_provider=embedding_provider,
    )

    result = await pipeline.process(
        IngestionCommand(
            document_id="doc-123",
            blob_uri="azurite://documents/doc-123.txt",
            correlation_id="request-123",
        )
    )

    assert result.document_id == "doc-123"
    assert result.blob_uri == "azurite://documents/doc-123.txt"
    assert [item.chunk.content for item in result.chunks] == [
        "abcdefghijkl",
        "klmnopqrstuv",
        "uvwxyz",
    ]
    assert [item.embedding.values for item in result.chunks] == [
        (12.0,),
        (12.0,),
        (6.0,),
    ]
    assert embedding_provider.texts == [
        "abcdefghijkl",
        "klmnopqrstuv",
        "uvwxyz",
    ]
    assert result.chunks[0].chunk.metadata["document_id"] == "doc-123"
    assert result.chunks[0].chunk.metadata["blob_uri"] == "azurite://documents/doc-123.txt"
    assert result.chunks[0].chunk.metadata["content_type"] == "text/plain"


class FakeBlobLoader:
    def load(self, *, document_id: str, blob_uri: str) -> LoadedBlobDocument:
        return LoadedBlobDocument(
            document_id=document_id,
            blob_uri=blob_uri,
            content=b"abcdefghijklmnopqrstuvwxyz",
            content_type="text/plain",
            metadata={},
        )


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.texts: Sequence[str] | None = None

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.texts = texts
        return [Embedding.from_iterable([float(len(text))]) for text in texts]
