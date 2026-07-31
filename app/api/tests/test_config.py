import pytest
from rag_api.config import ApiSettings
from rag_core.embeddings import (
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


def test_api_settings_builds_qdrant_config_from_environment_values() -> None:
    settings = ApiSettings(
        QDRANT_URL="http://qdrant.test:6333",
        QDRANT_COLLECTION_NAME="documents",
    )

    config = settings.qdrant_vector_store_config()

    assert config.url == "http://qdrant.test:6333"
    assert config.collection_name == "documents"
    assert config.api_key is None


@pytest.mark.parametrize(
    "setting_method",
    ["ollama_embedding_config", "qdrant_vector_store_config"],
)
def test_api_settings_raise_when_required_provider_values_are_missing(
    setting_method: str,
) -> None:
    settings = ApiSettings(
        OLLAMA_BASE_URL=None,
        OLLAMA_EMBEDDING_MODEL=None,
        QDRANT_URL=None,
        QDRANT_COLLECTION_NAME=None,
    )

    with pytest.raises(RuntimeError):
        getattr(settings, setting_method)()


def test_api_settings_use_shared_environment_names() -> None:
    assert ApiSettings.model_fields["ollama_base_url"].alias == OLLAMA_BASE_URL_ENV
    assert ApiSettings.model_fields["ollama_embedding_model"].alias == OLLAMA_EMBEDDING_MODEL_ENV
    assert ApiSettings.model_fields["qdrant_url"].alias == QDRANT_URL_ENV
    assert ApiSettings.model_fields["qdrant_collection_name"].alias == QDRANT_COLLECTION_NAME_ENV
