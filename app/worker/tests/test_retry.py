from __future__ import annotations

from rag_worker.retry import classify_ingestion_failure


def test_classify_ingestion_failure_marks_known_validation_errors_permanent() -> None:
    class DocumentTextExtractionError(ValueError):
        pass

    failure = classify_ingestion_failure(DocumentTextExtractionError("unsupported"))

    assert failure.kind == "permanent"
    assert failure.retryable is False
    assert failure.reason == "DocumentTextExtractionError"


def test_classify_ingestion_failure_marks_provider_errors_transient() -> None:
    class EmbeddingProviderError(Exception):
        pass

    failure = classify_ingestion_failure(EmbeddingProviderError("ollama unavailable"))

    assert failure.kind == "transient"
    assert failure.retryable is True
    assert failure.reason == "EmbeddingProviderError"


def test_classify_ingestion_failure_checks_wrapped_causes() -> None:
    class VectorStoreProviderError(Exception):
        pass

    try:
        raise VectorStoreProviderError("qdrant unavailable")
    except VectorStoreProviderError as exc:
        wrapped = RuntimeError("indexing failed")
        wrapped.__cause__ = exc

    failure = classify_ingestion_failure(wrapped)

    assert failure.kind == "transient"
    assert failure.retryable is True
    assert failure.reason == "VectorStoreProviderError"


def test_classify_ingestion_failure_retries_unknown_errors() -> None:
    failure = classify_ingestion_failure(RuntimeError("unexpected"))

    assert failure.kind == "transient"
    assert failure.retryable is True
    assert failure.reason == "RuntimeError"
