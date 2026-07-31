"""Domain models for embeddings."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Embedding:
    """Vector representation returned by an embedding provider."""

    values: tuple[float, ...]

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> Embedding:
        return cls(values=tuple(values))
