"""Document text extraction for worker ingestion."""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader

from rag_worker.blob_loader import LoadedBlobDocument


class DocumentTextExtractionError(ValueError):
    """Raised when document bytes cannot be converted to text."""


class DocumentTextExtractor:
    """Extract plain text from supported source document formats."""

    def extract_text(self, document: LoadedBlobDocument) -> str:
        if _is_pdf(document):
            text = _extract_pdf_text(document.content)
        elif _is_plain_text(document):
            text = _extract_utf8_text(document.content)
        else:
            raise DocumentTextExtractionError(
                f"Unsupported document type for blob URI: {document.blob_uri}"
            )

        if not text.strip():
            raise DocumentTextExtractionError("Document text extraction produced no content.")
        return text


def _is_pdf(document: LoadedBlobDocument) -> bool:
    return document.content_type == "application/pdf" or _suffix(document.blob_uri) == ".pdf"


def _is_plain_text(document: LoadedBlobDocument) -> bool:
    return document.content_type in {"text/plain", "text/markdown"} or _suffix(
        document.blob_uri
    ) in {".txt", ".md", ".markdown"}


def _extract_utf8_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentTextExtractionError("Document text must be valid UTF-8.") from exc


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(text for text in page_text if text.strip())


def _suffix(blob_uri: str) -> str:
    return PurePosixPath(blob_uri).suffix.lower()
