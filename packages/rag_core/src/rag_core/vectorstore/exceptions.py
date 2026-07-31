"""Domain-specific exceptions for vector-store workflows."""


class VectorStoreError(Exception):
    """Base class for vector-store domain errors."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when a vector-store adapter is missing required configuration."""


class VectorStoreProviderError(VectorStoreError):
    """Raised when a vector-store provider returns an invalid response or request error."""


class VectorStoreValidationError(VectorStoreError, ValueError):
    """Raised when vector-store inputs or outputs are invalid."""
