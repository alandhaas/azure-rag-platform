import pytest
from rag_worker.blob_loader import LoadedBlobDocument
from rag_worker.text_extraction import DocumentTextExtractionError, DocumentTextExtractor


def test_document_text_extractor_reads_plain_text() -> None:
    extractor = DocumentTextExtractor()
    document = LoadedBlobDocument(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.txt",
        content=b"hello from a document",
        content_type="text/plain",
        metadata={},
    )

    assert extractor.extract_text(document) == "hello from a document"


def test_document_text_extractor_uses_text_file_extension() -> None:
    extractor = DocumentTextExtractor()
    document = LoadedBlobDocument(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.md",
        content=b"# Notes",
        content_type=None,
        metadata={},
    )

    assert extractor.extract_text(document) == "# Notes"


def test_document_text_extractor_rejects_invalid_utf8_text() -> None:
    extractor = DocumentTextExtractor()
    document = LoadedBlobDocument(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.txt",
        content=b"\xff",
        content_type="text/plain",
        metadata={},
    )

    with pytest.raises(DocumentTextExtractionError):
        extractor.extract_text(document)


def test_document_text_extractor_rejects_unsupported_document_type() -> None:
    extractor = DocumentTextExtractor()
    document = LoadedBlobDocument(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.docx",
        content=b"not-yet-supported",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        metadata={},
    )

    with pytest.raises(DocumentTextExtractionError):
        extractor.extract_text(document)
