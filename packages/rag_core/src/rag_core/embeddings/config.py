"""Configuration models for embedding providers."""

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ as process_environ
from typing import Final, Self

from rag_core.embeddings.exceptions import EmbeddingConfigurationError

OLLAMA_BASE_URL_ENV: Final = "OLLAMA_BASE_URL"
OLLAMA_EMBEDDING_MODEL_ENV: Final = "OLLAMA_EMBEDDING_MODEL"


@dataclass(frozen=True, slots=True)
class OllamaEmbeddingConfig:
    """Runtime configuration for Ollama embedding requests."""

    base_url: str
    embedding_model: str
    request_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = environ or process_environ
        return cls(
            base_url=_required_env(env, OLLAMA_BASE_URL_ENV),
            embedding_model=_required_env(env, OLLAMA_EMBEDDING_MODEL_ENV),
        )


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise EmbeddingConfigurationError(f"Missing required environment variable: {name}")
    return value
