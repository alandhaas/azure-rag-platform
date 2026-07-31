"""Typed settings for the API application."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rag_core.embeddings import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_EMBEDDING_MODEL_ENV,
    OllamaEmbeddingConfig,
)
from rag_core.vectorstore import (
    QDRANT_API_KEY_ENV,
    QDRANT_COLLECTION_NAME_ENV,
    QDRANT_URL_ENV,
    QdrantVectorStoreConfig,
)


class ApiSettings(BaseSettings):
    """Environment-backed API settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_title: str = Field(default="Azure RAG Platform API", alias="API_TITLE")
    api_version: str = Field(default="0.1.0", alias="API_VERSION")
    otel_service_name: str = Field(default="rag-api", alias="OTEL_SERVICE_NAME")
    applicationinsights_connection_string: str | None = Field(
        default=None,
        alias="APPLICATIONINSIGHTS_CONNECTION_STRING",
    )

    ollama_base_url: str | None = Field(default=None, alias=OLLAMA_BASE_URL_ENV)
    ollama_embedding_model: str | None = Field(default=None, alias=OLLAMA_EMBEDDING_MODEL_ENV)

    qdrant_url: str | None = Field(default=None, alias=QDRANT_URL_ENV)
    qdrant_collection_name: str | None = Field(default=None, alias=QDRANT_COLLECTION_NAME_ENV)
    qdrant_api_key: str | None = Field(default=None, alias=QDRANT_API_KEY_ENV)

    azure_tenant_id: str | None = Field(default=None, alias="AZURE_TENANT_ID")
    azure_client_id: str | None = Field(default=None, alias="AZURE_CLIENT_ID")
    azure_audience: str | None = Field(default=None, alias="AZURE_AUDIENCE")

    def ollama_embedding_config(self) -> OllamaEmbeddingConfig:
        return OllamaEmbeddingConfig.from_env(
            {
                OLLAMA_BASE_URL_ENV: _required_setting(
                    self.ollama_base_url,
                    OLLAMA_BASE_URL_ENV,
                ),
                OLLAMA_EMBEDDING_MODEL_ENV: _required_setting(
                    self.ollama_embedding_model,
                    OLLAMA_EMBEDDING_MODEL_ENV,
                ),
            }
        )

    def qdrant_vector_store_config(self) -> QdrantVectorStoreConfig:
        env = {
            QDRANT_URL_ENV: _required_setting(self.qdrant_url, QDRANT_URL_ENV),
            QDRANT_COLLECTION_NAME_ENV: _required_setting(
                self.qdrant_collection_name,
                QDRANT_COLLECTION_NAME_ENV,
            ),
        }
        if self.qdrant_api_key:
            env[QDRANT_API_KEY_ENV] = self.qdrant_api_key
        return QdrantVectorStoreConfig.from_env(env)


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()


def _required_setting(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required API setting: {name}")
    return value
