from collections.abc import Sequence

from rag_core.embeddings import Embedding
from rag_core.ingestion import ChunkingConfig, TextChunker
from rag_core.vectorstore import VectorPoint
from rag_worker.blob_loader import LoadedBlobDocument
from rag_worker.commands import IngestionCommand
from rag_worker.ingestion import (
    DocumentEmbeddingPipeline,
    DocumentIndexingPipeline,
    embedded_document_to_vector_points,
)
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


async def test_document_indexing_pipeline_upserts_embedded_chunks() -> None:
    vector_store = FakeVectorStore()
    status_store = FakeStatusStore()
    pipeline = DocumentIndexingPipeline(
        embedding_pipeline=DocumentEmbeddingPipeline(
            blob_loader=FakeBlobLoader(),
            text_extractor=DocumentTextExtractor(),
            text_chunker=TextChunker(ChunkingConfig(max_chars=12, overlap_chars=2)),
            embedding_provider=FakeEmbeddingProvider(),
        ),
        vector_store=vector_store,
        status_store=status_store,
    )
    command = IngestionCommand(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.txt",
        correlation_id="request-123",
    )

    result = await pipeline.process(command)

    assert result.document_id == "doc-123"
    assert vector_store.points is not None
    assert len(vector_store.points) == 3
    assert vector_store.points[0].payload["content"] == "abcdefghijkl"
    assert vector_store.points[0].payload["document_id"] == "doc-123"
    assert vector_store.points[0].payload["blob_uri"] == "azurite://documents/doc-123.txt"
    assert vector_store.points[0].payload["chunk_index"] == 0
    assert status_store.calls == [
        "processing:doc-123",
        "indexed:doc-123:3",
    ]


async def test_document_indexing_pipeline_marks_failed_when_indexing_fails() -> None:
    status_store = FakeStatusStore()
    pipeline = DocumentIndexingPipeline(
        embedding_pipeline=DocumentEmbeddingPipeline(
            blob_loader=FakeBlobLoader(),
            text_extractor=DocumentTextExtractor(),
            text_chunker=TextChunker(ChunkingConfig(max_chars=12, overlap_chars=2)),
            embedding_provider=FakeEmbeddingProvider(),
        ),
        vector_store=FailingVectorStore(),
        status_store=status_store,
    )
    command = IngestionCommand(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.txt",
        correlation_id="request-123",
    )

    try:
        await pipeline.process(command)
    except RuntimeError:
        pass

    assert status_store.calls == [
        "processing:doc-123",
        "failed:doc-123:RuntimeError: no index",
    ]


async def test_document_indexing_pipeline_is_idempotent_for_duplicate_messages() -> None:
    vector_store = FakeVectorStore()
    pipeline = DocumentIndexingPipeline(
        embedding_pipeline=DocumentEmbeddingPipeline(
            blob_loader=FakeBlobLoader(),
            text_extractor=DocumentTextExtractor(),
            text_chunker=TextChunker(ChunkingConfig(max_chars=12, overlap_chars=2)),
            embedding_provider=FakeEmbeddingProvider(),
        ),
        vector_store=vector_store,
    )
    command = IngestionCommand(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.txt",
        correlation_id="request-123",
    )

    await pipeline.process(command)
    await pipeline.process(command)

    first_upsert_ids = [point.id for point in vector_store.upsert_calls[0]]
    second_upsert_ids = [point.id for point in vector_store.upsert_calls[1]]

    assert first_upsert_ids == second_upsert_ids
    assert len({*first_upsert_ids, *second_upsert_ids}) == 3


async def test_embedded_document_to_vector_points_uses_stable_ids() -> None:
    pipeline = DocumentEmbeddingPipeline(
        blob_loader=FakeBlobLoader(),
        text_extractor=DocumentTextExtractor(),
        text_chunker=TextChunker(ChunkingConfig(max_chars=12, overlap_chars=2)),
        embedding_provider=FakeEmbeddingProvider(),
    )
    command = IngestionCommand(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.txt",
        correlation_id="request-123",
    )

    document = await pipeline.process(command)
    first_points = embedded_document_to_vector_points(document)
    second_points = embedded_document_to_vector_points(document)

    assert [point.id for point in first_points] == [point.id for point in second_points]
    assert len({point.id for point in first_points}) == 3


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


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: Sequence[VectorPoint] | None = None
        self.upsert_calls: list[Sequence[VectorPoint]] = []

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        self.points = points
        self.upsert_calls.append(points)


class FailingVectorStore:
    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        raise RuntimeError("no index")


class FakeStatusStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def mark_processing(self, command: IngestionCommand) -> None:
        self.calls.append(f"processing:{command.document_id}")

    def mark_indexed(self, command: IngestionCommand, *, chunk_count: int) -> None:
        self.calls.append(f"indexed:{command.document_id}:{chunk_count}")

    def mark_failed(self, command: IngestionCommand, *, error: str) -> None:
        self.calls.append(f"failed:{command.document_id}:{error}")
