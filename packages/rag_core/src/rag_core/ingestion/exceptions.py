"""Domain-specific exceptions for ingestion workflows."""


class IngestionError(Exception):
    """Base class for ingestion domain errors."""


class IngestionValidationError(IngestionError, ValueError):
    """Raised when ingestion inputs or models are invalid."""
