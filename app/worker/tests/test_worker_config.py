from typing import Any, cast

import pytest
from pydantic import ValidationError
from rag_core.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_API_KEY_ENV,
    GEMINI_BASE_URL_ENV,
    GEMINI_EMBEDDING_MODEL_ENV,
    GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV,
    GOOGLE_API_KEY_ENV,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_EMBEDDING_MODEL_ENV,
)
from rag_core.vectorstore import QDRANT_COLLECTION_NAME_ENV, QDRANT_URL_ENV
from rag_worker.config import (
    AZURE_WEBJOBS_STORAGE_ENV,
    DOCUMENT_CHUNK_MAX_CHARS_ENV,
    DOCUMENT_CHUNK_OVERLAP_CHARS_ENV,
    DOCUMENT_METADATA_TABLE_NAME_ENV,
    DOCUMENTS_CONTAINER_NAME_ENV,
    INGESTION_QUEUE_NAME_ENV,
    WORKER_RETRY_LIMIT_ENV,
    WorkerSettings,
)


def test_worker_settings_use_local_defaults() -> None:
    settings_factory = cast(Any, WorkerSettings)
    settings = settings_factory(_env_file=None)

    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    assert settings.otel_service_name == "rag-worker"
    assert settings.ingestion_queue_name == "documents-to-ingest"
    assert settings.documents_container_name == "documents"
    assert settings.document_metadata_table_name == "DocumentMetadata"
    assert settings.worker_retry_limit == 5
    assert settings.document_chunk_max_chars == 1200
    assert settings.document_chunk_overlap_chars == 200


def test_worker_settings_builds_ollama_embedding_config_from_environment_values() -> None:
    settings = WorkerSettings(
        OLLAMA_BASE_URL="http://ollama.test",
        OLLAMA_EMBEDDING_MODEL="embedding-model",
    )

    config = settings.ollama_embedding_config()

    assert config.base_url == "http://ollama.test"
    assert config.embedding_model == "embedding-model"


def test_worker_settings_builds_gemini_embedding_config_from_environment_values() -> None:
    settings = WorkerSettings(
        GEMINI_API_KEY="test-key",
        GEMINI_BASE_URL="https://gemini.test",
        GEMINI_EMBEDDING_MODEL="gemini-embedding-001",
        GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY=768,
    )

    config = settings.gemini_embedding_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://gemini.test"
    assert config.embedding_model == "gemini-embedding-001"
    assert config.output_dimensionality == 768


def test_worker_settings_allows_google_api_key_alias_for_gemini() -> None:
    settings = WorkerSettings(
        GEMINI_API_KEY=None,
        GOOGLE_API_KEY="google-key",
    )

    config = settings.gemini_embedding_config()

    assert config.api_key == "google-key"


def test_worker_settings_builds_qdrant_config_from_environment_values() -> None:
    settings = WorkerSettings(
        QDRANT_URL="http://qdrant.test:6333",
        QDRANT_COLLECTION_NAME="documents",
    )

    config = settings.qdrant_vector_store_config()

    assert config.url == "http://qdrant.test:6333"
    assert config.collection_name == "documents"
    assert config.api_key is None


def test_worker_settings_reads_storage_connection_string() -> None:
    settings = WorkerSettings(AzureWebJobsStorage="UseDevelopmentStorage=true")

    assert settings.storage_connection_string() == "UseDevelopmentStorage=true"


@pytest.mark.parametrize(
    "setting_method",
    [
        "storage_connection_string",
        "ollama_embedding_config",
        "gemini_embedding_config",
        "qdrant_vector_store_config",
    ],
)
def test_worker_settings_raise_when_required_provider_values_are_missing(
    setting_method: str,
) -> None:
    settings = WorkerSettings(
        AzureWebJobsStorage=None,
        OLLAMA_BASE_URL=None,
        OLLAMA_EMBEDDING_MODEL=None,
        GEMINI_API_KEY=None,
        QDRANT_URL=None,
        QDRANT_COLLECTION_NAME=None,
    )

    with pytest.raises(RuntimeError):
        getattr(settings, setting_method)()


def test_worker_settings_validate_retry_limit() -> None:
    with pytest.raises(ValidationError):
        WorkerSettings(WORKER_RETRY_LIMIT=0)


def test_worker_settings_validate_chunking_values() -> None:
    with pytest.raises(ValidationError):
        WorkerSettings(DOCUMENT_CHUNK_MAX_CHARS=0)


def test_worker_settings_use_shared_environment_names() -> None:
    assert WorkerSettings.model_fields["azure_webjobs_storage"].alias == AZURE_WEBJOBS_STORAGE_ENV
    assert WorkerSettings.model_fields["ingestion_queue_name"].alias == INGESTION_QUEUE_NAME_ENV
    assert (
        WorkerSettings.model_fields["documents_container_name"].alias
        == DOCUMENTS_CONTAINER_NAME_ENV
    )
    assert (
        WorkerSettings.model_fields["document_metadata_table_name"].alias
        == DOCUMENT_METADATA_TABLE_NAME_ENV
    )
    assert WorkerSettings.model_fields["worker_retry_limit"].alias == WORKER_RETRY_LIMIT_ENV
    assert (
        WorkerSettings.model_fields["document_chunk_max_chars"].alias
        == DOCUMENT_CHUNK_MAX_CHARS_ENV
    )
    assert (
        WorkerSettings.model_fields["document_chunk_overlap_chars"].alias
        == DOCUMENT_CHUNK_OVERLAP_CHARS_ENV
    )
    assert WorkerSettings.model_fields["embedding_provider"].alias == EMBEDDING_PROVIDER_ENV
    assert WorkerSettings.model_fields["ollama_base_url"].alias == OLLAMA_BASE_URL_ENV
    assert WorkerSettings.model_fields["ollama_embedding_model"].alias == OLLAMA_EMBEDDING_MODEL_ENV
    assert WorkerSettings.model_fields["gemini_api_key"].alias == GEMINI_API_KEY_ENV
    assert WorkerSettings.model_fields["google_api_key"].alias == GOOGLE_API_KEY_ENV
    assert WorkerSettings.model_fields["gemini_base_url"].alias == GEMINI_BASE_URL_ENV
    assert WorkerSettings.model_fields["gemini_embedding_model"].alias == GEMINI_EMBEDDING_MODEL_ENV
    assert (
        WorkerSettings.model_fields["gemini_embedding_output_dimensionality"].alias
        == GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV
    )
    assert WorkerSettings.model_fields["qdrant_url"].alias == QDRANT_URL_ENV
    assert WorkerSettings.model_fields["qdrant_collection_name"].alias == QDRANT_COLLECTION_NAME_ENV
