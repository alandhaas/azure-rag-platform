import pytest
from rag_api.config import ApiSettings
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


def test_api_settings_builds_ollama_embedding_config_from_environment_values() -> None:
    settings = ApiSettings(
        OLLAMA_BASE_URL="http://ollama.test",
        OLLAMA_EMBEDDING_MODEL="embedding-model",
    )

    config = settings.ollama_embedding_config()

    assert config.base_url == "http://ollama.test"
    assert config.embedding_model == "embedding-model"


def test_api_settings_builds_gemini_embedding_config_from_environment_values() -> None:
    settings = ApiSettings(
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


def test_api_settings_allows_google_api_key_alias_for_gemini() -> None:
    settings = ApiSettings(
        GEMINI_API_KEY=None,
        GOOGLE_API_KEY="google-key",
    )

    config = settings.gemini_embedding_config()

    assert config.api_key == "google-key"


def test_api_settings_builds_qdrant_config_from_environment_values() -> None:
    settings = ApiSettings(
        QDRANT_URL="http://qdrant.test:6333",
        QDRANT_COLLECTION_NAME="documents",
    )

    config = settings.qdrant_vector_store_config()

    assert config.url == "http://qdrant.test:6333"
    assert config.collection_name == "documents"
    assert config.api_key is None


def test_api_settings_builds_storage_config_from_environment_values() -> None:
    settings = ApiSettings(
        AzureWebJobsStorage="UseDevelopmentStorage=true",
        INGESTION_QUEUE_NAME="queue",
        DOCUMENTS_CONTAINER_NAME="documents",
        DOCUMENT_METADATA_TABLE_NAME="DocumentMetadata",
    )

    assert settings.storage_connection_string() == "UseDevelopmentStorage=true"
    assert settings.ingestion_queue_name == "queue"
    assert settings.documents_container_name == "documents"
    assert settings.document_metadata_table_name == "DocumentMetadata"


@pytest.mark.parametrize(
    "setting_method",
    [
        "ollama_embedding_config",
        "gemini_embedding_config",
        "qdrant_vector_store_config",
        "storage_connection_string",
    ],
)
def test_api_settings_raise_when_required_provider_values_are_missing(
    setting_method: str,
) -> None:
    settings = ApiSettings(
        OLLAMA_BASE_URL=None,
        OLLAMA_EMBEDDING_MODEL=None,
        GEMINI_API_KEY=None,
        QDRANT_URL=None,
        QDRANT_COLLECTION_NAME=None,
        AzureWebJobsStorage=None,
    )

    with pytest.raises(RuntimeError):
        getattr(settings, setting_method)()


def test_api_settings_use_shared_environment_names() -> None:
    assert ApiSettings.model_fields["embedding_provider"].alias == EMBEDDING_PROVIDER_ENV
    assert ApiSettings.model_fields["ollama_base_url"].alias == OLLAMA_BASE_URL_ENV
    assert ApiSettings.model_fields["ollama_embedding_model"].alias == OLLAMA_EMBEDDING_MODEL_ENV
    assert ApiSettings.model_fields["gemini_api_key"].alias == GEMINI_API_KEY_ENV
    assert ApiSettings.model_fields["google_api_key"].alias == GOOGLE_API_KEY_ENV
    assert ApiSettings.model_fields["gemini_base_url"].alias == GEMINI_BASE_URL_ENV
    assert ApiSettings.model_fields["gemini_embedding_model"].alias == GEMINI_EMBEDDING_MODEL_ENV
    assert (
        ApiSettings.model_fields["gemini_embedding_output_dimensionality"].alias
        == GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV
    )
    assert ApiSettings.model_fields["qdrant_url"].alias == QDRANT_URL_ENV
    assert ApiSettings.model_fields["qdrant_collection_name"].alias == QDRANT_COLLECTION_NAME_ENV
    assert ApiSettings.model_fields["azure_webjobs_storage"].alias == "AzureWebJobsStorage"
    assert ApiSettings.model_fields["ingestion_queue_name"].alias == "INGESTION_QUEUE_NAME"
    assert ApiSettings.model_fields["documents_container_name"].alias == "DOCUMENTS_CONTAINER_NAME"
    assert (
        ApiSettings.model_fields["document_metadata_table_name"].alias
        == "DOCUMENT_METADATA_TABLE_NAME"
    )
