import pytest
from rag_core.ingestion import ChunkingConfig, IngestionValidationError, TextChunk


def test_text_chunk_exposes_metadata_with_offsets() -> None:
    chunk = TextChunk(
        content="Chunk content",
        chunk_index=2,
        char_start=10,
        char_end=23,
        source_metadata={"document_id": "document-1"},
    )

    assert chunk.metadata == {
        "document_id": "document-1",
        "chunk_index": 2,
        "char_start": 10,
        "char_end": 23,
    }


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunking_config_rejects_invalid_values(max_chars: int, overlap_chars: int) -> None:
    with pytest.raises(IngestionValidationError):
        ChunkingConfig(max_chars=max_chars, overlap_chars=overlap_chars)


@pytest.mark.parametrize(
    ("content", "chunk_index", "char_start", "char_end"),
    [
        (" ", 0, 0, 1),
        ("content", -1, 0, 7),
        ("content", 0, -1, 7),
        ("content", 0, 7, 7),
    ],
)
def test_text_chunk_rejects_invalid_values(
    content: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
) -> None:
    with pytest.raises(IngestionValidationError):
        TextChunk(
            content=content,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
        )
