"""Domain models for document chunking."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from rag_core.ingestion.exceptions import IngestionValidationError

ChunkMetadata = Mapping[str, object]


def _empty_metadata() -> ChunkMetadata:
    return {}


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based chunking settings."""

    max_chars: int
    overlap_chars: int

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise IngestionValidationError("Chunk max_chars must be at least 1.")
        if self.overlap_chars < 0:
            raise IngestionValidationError("Chunk overlap_chars cannot be negative.")
        if self.overlap_chars >= self.max_chars:
            raise IngestionValidationError("Chunk overlap_chars must be smaller than max_chars.")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk of source text with character offsets."""

    content: str
    chunk_index: int
    char_start: int
    char_end: int
    source_metadata: ChunkMetadata = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise IngestionValidationError("Text chunk content is required.")
        if self.chunk_index < 0:
            raise IngestionValidationError("Text chunk index cannot be negative.")
        if self.char_start < 0:
            raise IngestionValidationError("Text chunk char_start cannot be negative.")
        if self.char_end <= self.char_start:
            raise IngestionValidationError("Text chunk char_end must be greater than char_start.")

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **self.source_metadata,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }
