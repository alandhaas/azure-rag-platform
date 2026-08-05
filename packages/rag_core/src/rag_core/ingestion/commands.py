"""Queue command contracts for document ingestion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast


class IngestionCommandError(ValueError):
    """Raised when an ingestion queue message does not match the command contract."""


@dataclass(frozen=True)
class IngestionCommand:
    """Command consumed by the ingestion worker queue trigger."""

    document_id: str
    blob_uri: str
    correlation_id: str

    def __post_init__(self) -> None:
        _require_text(self.document_id, "document_id")
        _require_text(self.blob_uri, "blob_uri")
        _require_text(self.correlation_id, "correlation_id")

    @classmethod
    def from_json(cls, payload: str | bytes) -> IngestionCommand:
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise IngestionCommandError("Ingestion command must be valid JSON.") from exc

        if not isinstance(decoded, dict):
            raise IngestionCommandError("Ingestion command must be a JSON object.")

        return cls.from_mapping(cast(Mapping[str, Any], decoded))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IngestionCommand:
        return cls(
            document_id=_required_payload_text(payload, "document_id"),
            blob_uri=_required_payload_text(payload, "blob_uri"),
            correlation_id=_required_payload_text(payload, "correlation_id"),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _required_payload_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise IngestionCommandError(f"Ingestion command requires a non-empty {name}.")
    return value


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise IngestionCommandError(f"Ingestion command requires a non-empty {name}.")
