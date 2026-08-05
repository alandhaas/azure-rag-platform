"""Text chunking transformations."""

from __future__ import annotations

from collections.abc import Mapping

from rag_core.ingestion.exceptions import IngestionValidationError
from rag_core.ingestion.models import ChunkingConfig, TextChunk


class TextChunker:
    """Split plain text into overlapping character chunks."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_text(
        self,
        text: str,
        *,
        source_metadata: Mapping[str, object] | None = None,
    ) -> list[TextChunk]:
        if not text.strip():
            raise IngestionValidationError("Text content is required for chunking.")

        chunks: list[TextChunk] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.config.max_chars, text_length)
            content = text[start:end]
            if content.strip():
                chunks.append(
                    TextChunk(
                        content=content,
                        chunk_index=len(chunks),
                        char_start=start,
                        char_end=end,
                        source_metadata=source_metadata or {},
                    )
                )

            if end == text_length:
                break
            start = end - self.config.overlap_chars

        if not chunks:
            raise IngestionValidationError("Text content produced no chunks.")

        return chunks
