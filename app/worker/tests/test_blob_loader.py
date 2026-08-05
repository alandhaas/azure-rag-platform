from dataclasses import dataclass

import pytest
from rag_worker.blob_loader import BlobDocumentLoader, BlobLoaderError, BlobLocation, parse_blob_uri


def test_parse_blob_uri_supports_azurite_shorthand() -> None:
    location = parse_blob_uri("azurite://documents/folder/file.pdf")

    assert location == BlobLocation(
        container_name="documents",
        blob_name="folder/file.pdf",
    )


def test_parse_blob_uri_supports_azurite_http_url() -> None:
    location = parse_blob_uri(
        "http://127.0.0.1:10000/devstoreaccount1/documents/folder/file%20name.pdf",
        account_name="devstoreaccount1",
    )

    assert location == BlobLocation(
        container_name="documents",
        blob_name="folder/file name.pdf",
    )


def test_parse_blob_uri_supports_azure_blob_url() -> None:
    location = parse_blob_uri(
        "https://account.blob.core.windows.net/documents/file.pdf",
        account_name="account",
    )

    assert location == BlobLocation(container_name="documents", blob_name="file.pdf")


@pytest.mark.parametrize(
    "blob_uri",
    [
        "",
        "file:///tmp/file.pdf",
        "azurite://documents",
        "https://account.blob.core.windows.net/documents",
    ],
)
def test_parse_blob_uri_rejects_invalid_uris(blob_uri: str) -> None:
    with pytest.raises(BlobLoaderError):
        parse_blob_uri(blob_uri, account_name="account")


def test_blob_document_loader_downloads_document() -> None:
    service_client = FakeBlobServiceClient()
    loader = BlobDocumentLoader(
        "AccountName=devstoreaccount1;AccountKey=fake;",
        service_client=service_client,
    )

    document = loader.load(
        document_id="doc-123",
        blob_uri="http://127.0.0.1:10000/devstoreaccount1/documents/doc-123.pdf",
    )

    assert service_client.requested_container == "documents"
    assert service_client.requested_blob == "doc-123.pdf"
    assert document.document_id == "doc-123"
    assert document.content == b"pdf-bytes"
    assert document.content_type == "application/pdf"
    assert document.metadata == {"source": "upload"}


def test_blob_document_loader_rejects_missing_connection_string() -> None:
    with pytest.raises(BlobLoaderError):
        BlobDocumentLoader(" ")


def test_blob_document_loader_rejects_missing_document_id() -> None:
    loader = BlobDocumentLoader(
        "AccountName=devstoreaccount1;AccountKey=fake;",
        service_client=FakeBlobServiceClient(),
    )

    with pytest.raises(BlobLoaderError):
        loader.load(document_id=" ", blob_uri="azurite://documents/doc.pdf")


class FakeBlobServiceClient:
    def __init__(self) -> None:
        self.requested_container: str | None = None
        self.requested_blob: str | None = None

    def get_blob_client(self, *, container: str, blob: str) -> FakeBlobClient:
        self.requested_container = container
        self.requested_blob = blob
        return FakeBlobClient()


class FakeBlobClient:
    def download_blob(self) -> FakeDownloader:
        return FakeDownloader()

    def get_blob_properties(self) -> FakeBlobProperties:
        return FakeBlobProperties()


class FakeDownloader:
    def readall(self) -> bytes:
        return b"pdf-bytes"


@dataclass(frozen=True, slots=True)
class FakeContentSettings:
    content_type: str = "application/pdf"


@dataclass(frozen=True, slots=True)
class FakeBlobProperties:
    content_settings: FakeContentSettings = FakeContentSettings()
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {"source": "upload"})
