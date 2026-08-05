"""Blob loading utilities for worker ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import unquote, urlparse

from azure.storage.blob import BlobServiceClient


class BlobLoaderError(ValueError):
    """Raised when a source document blob cannot be addressed or loaded."""


class _Downloader(Protocol):
    def readall(self) -> bytes: ...


class _BlobClient(Protocol):
    def download_blob(self) -> _Downloader: ...
    def get_blob_properties(self) -> object: ...


class _BlobServiceClient(Protocol):
    def get_blob_client(self, *, container: str, blob: str) -> _BlobClient: ...


@dataclass(frozen=True, slots=True)
class BlobLocation:
    """Container/blob pair resolved from a command blob URI."""

    container_name: str
    blob_name: str


@dataclass(frozen=True, slots=True)
class LoadedBlobDocument:
    """Document bytes loaded from Blob Storage."""

    document_id: str
    blob_uri: str
    content: bytes
    content_type: str | None
    metadata: Mapping[str, str]


class BlobDocumentLoader:
    """Load source document bytes from Azure Blob Storage."""

    def __init__(
        self,
        connection_string: str,
        *,
        service_client: _BlobServiceClient | None = None,
    ) -> None:
        if not connection_string.strip():
            raise BlobLoaderError("Blob storage connection string is required.")

        self._account_name = _connection_string_value(connection_string, "AccountName")
        self._service_client = service_client or cast(
            _BlobServiceClient,
            BlobServiceClient.from_connection_string(connection_string),
        )

    def load(self, *, document_id: str, blob_uri: str) -> LoadedBlobDocument:
        if not document_id.strip():
            raise BlobLoaderError("Document id is required.")

        location = parse_blob_uri(blob_uri, account_name=self._account_name)
        blob_client = self._service_client.get_blob_client(
            container=location.container_name,
            blob=location.blob_name,
        )

        content = blob_client.download_blob().readall()
        properties = blob_client.get_blob_properties()

        return LoadedBlobDocument(
            document_id=document_id,
            blob_uri=blob_uri,
            content=content,
            content_type=_content_type(properties),
            metadata=_metadata(properties),
        )


def parse_blob_uri(blob_uri: str, *, account_name: str | None = None) -> BlobLocation:
    if not blob_uri.strip():
        raise BlobLoaderError("Blob URI is required.")

    parsed = urlparse(blob_uri)
    if parsed.scheme in {"azurite", "blob"}:
        return _location_from_path(parsed.netloc, parsed.path)

    if parsed.scheme in {"http", "https"}:
        return _location_from_http_path(parsed.path, account_name=account_name)

    raise BlobLoaderError(f"Unsupported blob URI scheme: {parsed.scheme or '<missing>'}.")


def _location_from_http_path(path: str, *, account_name: str | None) -> BlobLocation:
    parts = _path_parts(path)
    if account_name and parts and parts[0] == account_name:
        parts = parts[1:]

    if len(parts) < 2:
        raise BlobLoaderError("Blob URI path must include container and blob name.")

    return BlobLocation(container_name=parts[0], blob_name="/".join(parts[1:]))


def _location_from_path(container: str, path: str) -> BlobLocation:
    blob_name = "/".join(_path_parts(path))
    if not container.strip() or not blob_name:
        raise BlobLoaderError("Blob URI must include container and blob name.")
    return BlobLocation(container_name=container, blob_name=blob_name)


def _path_parts(path: str) -> list[str]:
    return [unquote(part) for part in path.split("/") if part]


def _connection_string_value(connection_string: str, key: str) -> str | None:
    prefix = f"{key}="
    for part in connection_string.split(";"):
        if part.startswith(prefix):
            value = part.removeprefix(prefix).strip()
            return value or None
    return None


def _content_type(properties: object) -> str | None:
    content_settings = getattr(properties, "content_settings", None)
    value = getattr(content_settings, "content_type", None)
    return value if isinstance(value, str) and value else None


def _metadata(properties: object) -> Mapping[str, str]:
    value = getattr(properties, "metadata", None)
    if isinstance(value, Mapping):
        metadata = cast(Mapping[Any, Any], value)
        return {
            str(metadata_key): str(metadata_value)
            for metadata_key, metadata_value in metadata.items()
        }
    return {}
