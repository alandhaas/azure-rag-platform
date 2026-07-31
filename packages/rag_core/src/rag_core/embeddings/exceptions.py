"""Domain-specific exceptions for embedding workflows."""


class EmbeddingError(Exception):
    """Base class for embedding domain errors."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding provider is missing required configuration."""


class EmbeddingValidationError(EmbeddingError, ValueError):
    """Raised when embedding inputs or provider outputs are invalid."""


class EmptyEmbeddingInputError(EmbeddingValidationError):
    """Raised when an embedding request contains no usable text."""


class EmbeddingDimensionError(EmbeddingValidationError):
    """Raised when an embedding vector has an unexpected dimension."""


class EmbeddingBatchSizeError(EmbeddingValidationError):
    """Raised when a batch size is outside the supported range."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider returns an invalid response or request error."""
