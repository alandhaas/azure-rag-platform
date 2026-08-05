import pytest
from rag_core.ingestion import ChunkingConfig, IngestionValidationError, TextChunker


def test_text_chunker_splits_text_with_overlap() -> None:
    chunker = TextChunker(ChunkingConfig(max_chars=5, overlap_chars=2))

    chunks = chunker.chunk_text(
        "abcdefghijkl",
        source_metadata={"document_id": "doc-123"},
    )

    assert [chunk.content for chunk in chunks] == ["abcde", "defgh", "ghijk", "jkl"]
    assert [(chunk.char_start, chunk.char_end) for chunk in chunks] == [
        (0, 5),
        (3, 8),
        (6, 11),
        (9, 12),
    ]
    assert chunks[0].metadata["document_id"] == "doc-123"
    assert chunks[1].metadata["chunk_index"] == 1


def test_text_chunker_rejects_blank_text() -> None:
    chunker = TextChunker(ChunkingConfig(max_chars=100, overlap_chars=10))

    with pytest.raises(IngestionValidationError):
        chunker.chunk_text("   ")
