"""Domain-specific exceptions for vector-store workflows."""


class VectorStoreError(Exception):
    """Base class for vector-store domain errors."""


class VectorStoreValidationError(VectorStoreError, ValueError):
    """Raised when vector-store inputs or outputs are invalid."""
