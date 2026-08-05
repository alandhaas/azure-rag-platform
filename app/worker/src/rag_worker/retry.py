"""Retry classification for worker ingestion failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FailureKind = Literal["permanent", "transient"]

_PERMANENT_ERROR_NAMES = frozenset(
    {
        "BlobLoaderError",
        "DocumentTextExtractionError",
        "EmbeddingBatchSizeError",
        "EmbeddingConfigurationError",
        "EmbeddingDimensionError",
        "EmbeddingValidationError",
        "EmptyEmbeddingInputError",
        "IngestionCommandError",
        "IngestionValidationError",
        "ResourceNotFoundError",
        "VectorStoreConfigurationError",
        "VectorStoreValidationError",
    }
)
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "AzureError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "EmbeddingProviderError",
        "HttpResponseError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServiceRequestError",
        "ServiceResponseError",
        "TimeoutError",
        "VectorStoreProviderError",
        "WriteError",
        "WriteTimeout",
    }
)


@dataclass(frozen=True)
class IngestionFailureClassification:
    """Operational classification used for queue retry decisions and logs."""

    kind: FailureKind
    retryable: bool
    reason: str


def classify_ingestion_failure(exc: BaseException) -> IngestionFailureClassification:
    """Classify ingestion failures without importing provider packages at startup."""
    for candidate in _exception_chain(exc):
        error_name = type(candidate).__name__
        if error_name in _PERMANENT_ERROR_NAMES:
            return IngestionFailureClassification(
                kind="permanent",
                retryable=False,
                reason=error_name,
            )
        if error_name in _TRANSIENT_ERROR_NAMES:
            return IngestionFailureClassification(
                kind="transient",
                retryable=True,
                reason=error_name,
            )

    return IngestionFailureClassification(
        kind="transient",
        retryable=True,
        reason=type(exc).__name__,
    )


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain = [exc]
    current = exc
    while current.__cause__ is not None:
        chain.append(current.__cause__)
        current = current.__cause__
    return chain
