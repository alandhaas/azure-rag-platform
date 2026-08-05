"""Document ingestion status models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

DocumentIngestionStatusValue = Literal["queued", "processing", "indexed", "failed"]


@dataclass(frozen=True, slots=True)
class DocumentIngestionStatus:
    """Current ingestion state for an uploaded document."""

    document_id: str
    status: DocumentIngestionStatusValue
    blob_uri: str
    file_name: str
    content_type: str
    queue_name: str
    correlation_id: str
    created_at: str
    updated_at: str
    chunk_count: int | None = None
    error: str | None = None


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for status rows."""
    return datetime.now(UTC).isoformat()
