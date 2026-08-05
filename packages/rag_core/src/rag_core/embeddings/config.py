"""Configuration models for embedding providers."""

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ as process_environ
from typing import Final, Self

from rag_core.embeddings.exceptions import EmbeddingConfigurationError

EMBEDDING_PROVIDER_ENV: Final = "EMBEDDING_PROVIDER"
OLLAMA_BASE_URL_ENV: Final = "OLLAMA_BASE_URL"
OLLAMA_EMBEDDING_MODEL_ENV: Final = "OLLAMA_EMBEDDING_MODEL"
GEMINI_API_KEY_ENV: Final = "GEMINI_API_KEY"
GOOGLE_API_KEY_ENV: Final = "GOOGLE_API_KEY"
GEMINI_BASE_URL_ENV: Final = "GEMINI_BASE_URL"
GEMINI_EMBEDDING_MODEL_ENV: Final = "GEMINI_EMBEDDING_MODEL"
GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV: Final = (
    "GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY"
)
DEFAULT_GEMINI_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta"


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


@dataclass(frozen=True, slots=True)
class GeminiEmbeddingConfig:
    """Runtime configuration for Google AI Studio Gemini embedding requests."""

    api_key: str
    embedding_model: str
    base_url: str = DEFAULT_GEMINI_BASE_URL
    output_dimensionality: int | None = None
    request_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = environ or process_environ
        return cls(
            api_key=_required_first_env(env, (GOOGLE_API_KEY_ENV, GEMINI_API_KEY_ENV)),
            embedding_model=_required_env(env, GEMINI_EMBEDDING_MODEL_ENV),
            base_url=_optional_env(env, GEMINI_BASE_URL_ENV) or DEFAULT_GEMINI_BASE_URL,
            output_dimensionality=_optional_positive_int(
                env,
                GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV,
            ),
        )


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise EmbeddingConfigurationError(f"Missing required environment variable: {name}")
    return value


def _required_first_env(environ: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = environ.get(name)
        if value is not None and value.strip():
            return value
    joined_names = " or ".join(names)
    raise EmbeddingConfigurationError(f"Missing required environment variable: {joined_names}")


def _optional_env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _optional_positive_int(environ: Mapping[str, str], name: str) -> int | None:
    value = _optional_env(environ, name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise EmbeddingConfigurationError(
            f"Environment variable must be an integer: {name}"
        ) from exc
    if parsed < 1:
        raise EmbeddingConfigurationError(
            f"Environment variable must be a positive integer: {name}"
        )
    return parsed
