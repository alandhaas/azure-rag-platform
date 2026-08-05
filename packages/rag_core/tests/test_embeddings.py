import json
from collections.abc import Sequence

import httpx
import pytest
from rag_core.embeddings import (
    GEMINI_API_KEY_ENV,
    GEMINI_BASE_URL_ENV,
    GEMINI_EMBEDDING_MODEL_ENV,
    GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV,
    GOOGLE_API_KEY_ENV,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_EMBEDDING_MODEL_ENV,
    Embedding,
    EmbeddingBatchSizeError,
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingValidationError,
    EmptyEmbeddingInputError,
    GeminiEmbeddingConfig,
    GeminiEmbeddingProvider,
    OllamaEmbeddingConfig,
    OllamaEmbeddingProvider,
    batch_texts,
    validate_embedding_response,
    validate_texts,
)


class StaticEmbeddingProvider:
    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [Embedding.from_iterable([float(len(text))]) for text in texts]


def test_embedding_from_iterable_stores_values_as_tuple() -> None:
    embedding = Embedding.from_iterable([0.1, 0.2, 0.3])

    assert embedding.values == (0.1, 0.2, 0.3)
    assert embedding.dimension == 3


def test_embedding_rejects_empty_vectors() -> None:
    with pytest.raises(EmbeddingDimensionError):
        Embedding.from_iterable([])


@pytest.mark.asyncio
async def test_embedding_provider_contract_supports_async_batches() -> None:
    provider: EmbeddingProvider = StaticEmbeddingProvider()

    embeddings = await provider.embed(["one", "three"])

    assert [embedding.values for embedding in embeddings] == [(3.0,), (5.0,)]


def test_validate_texts_returns_tuple_for_non_empty_texts() -> None:
    assert validate_texts(["hello", " world "]) == ("hello", " world ")


@pytest.mark.parametrize("texts", [[], [""], ["   "], ["valid", "\t"]])
def test_validate_texts_rejects_empty_inputs(texts: list[str]) -> None:
    with pytest.raises(EmptyEmbeddingInputError):
        validate_texts(texts)


def test_batch_texts_splits_validated_input() -> None:
    assert batch_texts(["one", "two", "three"], batch_size=2) == [
        ("one", "two"),
        ("three",),
    ]


def test_batch_texts_rejects_invalid_batch_size() -> None:
    with pytest.raises(EmbeddingBatchSizeError):
        batch_texts(["one"], batch_size=0)


def test_validate_embedding_response_returns_valid_embeddings() -> None:
    embeddings = [
        Embedding.from_iterable([0.1, 0.2]),
        Embedding.from_iterable([0.3, 0.4]),
    ]

    validated = validate_embedding_response(
        ["one", "two"],
        embeddings,
        expected_dimension=2,
    )

    assert validated == tuple(embeddings)


def test_validate_embedding_response_rejects_count_mismatch() -> None:
    with pytest.raises(EmbeddingValidationError):
        validate_embedding_response(
            ["one", "two"],
            [Embedding.from_iterable([0.1])],
        )


def test_validate_embedding_response_rejects_dimension_mismatch() -> None:
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding_response(
            ["one", "two"],
            [
                Embedding.from_iterable([0.1, 0.2]),
                Embedding.from_iterable([0.3]),
            ],
        )


def test_validate_embedding_response_rejects_invalid_expected_dimension() -> None:
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding_response(
            ["one"],
            [Embedding.from_iterable([0.1])],
            expected_dimension=0,
        )


def test_ollama_embedding_config_reads_expected_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _ollama_env()
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    config = OllamaEmbeddingConfig.from_env()

    assert config.base_url == env[OLLAMA_BASE_URL_ENV]
    assert config.embedding_model == env[OLLAMA_EMBEDDING_MODEL_ENV]


def test_ollama_embedding_config_can_be_built_from_env_mapping() -> None:
    env = _ollama_env()

    config = OllamaEmbeddingConfig.from_env(env)

    assert config == OllamaEmbeddingConfig(
        base_url=env[OLLAMA_BASE_URL_ENV],
        embedding_model=env[OLLAMA_EMBEDDING_MODEL_ENV],
    )


@pytest.mark.parametrize(
    "missing_name",
    [OLLAMA_BASE_URL_ENV, OLLAMA_EMBEDDING_MODEL_ENV],
)
def test_ollama_embedding_config_requires_environment(missing_name: str) -> None:
    env = _ollama_env()
    env.pop(missing_name)

    with pytest.raises(EmbeddingConfigurationError):
        OllamaEmbeddingConfig.from_env(env)


def test_gemini_embedding_config_reads_expected_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _gemini_env()
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    config = GeminiEmbeddingConfig.from_env()

    assert config.api_key == env[GEMINI_API_KEY_ENV]
    assert config.base_url == env[GEMINI_BASE_URL_ENV]
    assert config.embedding_model == env[GEMINI_EMBEDDING_MODEL_ENV]
    assert config.output_dimensionality == 768


def test_gemini_embedding_config_requires_api_key() -> None:
    env = _gemini_env()
    env.pop(GEMINI_API_KEY_ENV)

    with pytest.raises(EmbeddingConfigurationError):
        GeminiEmbeddingConfig.from_env(env)


def test_gemini_embedding_config_rejects_invalid_output_dimensionality() -> None:
    env = _gemini_env()
    env[GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV] = "0"

    with pytest.raises(EmbeddingConfigurationError):
        GeminiEmbeddingConfig.from_env(env)


def test_gemini_embedding_config_prefers_google_api_key_when_both_are_set() -> None:
    env = _gemini_env()
    env[GOOGLE_API_KEY_ENV] = "google-key"

    config = GeminiEmbeddingConfig.from_env(env)

    assert config.api_key == "google-key"


@pytest.mark.asyncio
async def test_ollama_embedding_provider_posts_batch_to_embed_endpoint() -> None:
    env = _ollama_env()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://ollama.test") as client:
        provider = OllamaEmbeddingProvider(
            OllamaEmbeddingConfig(
                base_url=env[OLLAMA_BASE_URL_ENV],
                embedding_model=env[OLLAMA_EMBEDDING_MODEL_ENV],
            ),
            client=client,
        )

        embeddings = await provider.embed(["first", "second"])

    assert [embedding.values for embedding in embeddings] == [(0.1, 0.2), (0.3, 0.4)]
    assert requests[0].url.path == "/api/embed"
    assert json.loads(requests[0].content) == {
        "model": env[OLLAMA_EMBEDDING_MODEL_ENV],
        "input": ["first", "second"],
    }


@pytest.mark.asyncio
async def test_ollama_embedding_provider_rejects_invalid_payload() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"items": []}))

    async with httpx.AsyncClient(transport=transport, base_url="http://ollama.test") as client:
        provider = OllamaEmbeddingProvider(
            OllamaEmbeddingConfig.from_env(_ollama_env()),
            client=client,
        )

        with pytest.raises(EmbeddingProviderError):
            await provider.embed(["first"])


@pytest.mark.asyncio
async def test_ollama_embedding_provider_wraps_http_errors() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, json={"error": "missing"}))

    async with httpx.AsyncClient(transport=transport, base_url="http://ollama.test") as client:
        provider = OllamaEmbeddingProvider(
            OllamaEmbeddingConfig.from_env(_ollama_env()),
            client=client,
        )

        with pytest.raises(EmbeddingProviderError):
            await provider.embed(["first"])


@pytest.mark.asyncio
async def test_gemini_embedding_provider_posts_batch_to_embed_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://gemini.test") as client:
        provider = GeminiEmbeddingProvider(
            GeminiEmbeddingConfig(
                api_key="test-key",
                base_url="https://gemini.test",
                embedding_model="gemini-embedding-001",
                output_dimensionality=2,
            ),
            client=client,
        )

        embeddings = await provider.embed(["first", "second"])

    assert [embedding.values for embedding in embeddings] == [(0.1, 0.2), (0.3, 0.4)]
    assert requests[0].url.path == "/models/gemini-embedding-001:batchEmbedContents"
    assert requests[0].headers["x-goog-api-key"] == "test-key"
    assert json.loads(requests[0].content) == {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "first"}]},
                "outputDimensionality": 2,
                "embedContentConfig": {"outputDimensionality": 2},
            },
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "second"}]},
                "outputDimensionality": 2,
                "embedContentConfig": {"outputDimensionality": 2},
            },
        ]
    }


@pytest.mark.asyncio
async def test_gemini_embedding_provider_rejects_invalid_payload() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"items": []}))

    async with httpx.AsyncClient(transport=transport, base_url="https://gemini.test") as client:
        provider = GeminiEmbeddingProvider(
            GeminiEmbeddingConfig.from_env(_gemini_env()),
            client=client,
        )

        with pytest.raises(EmbeddingProviderError):
            await provider.embed(["first"])


@pytest.mark.asyncio
async def test_gemini_embedding_provider_wraps_http_errors() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "key"}))

    async with httpx.AsyncClient(transport=transport, base_url="https://gemini.test") as client:
        provider = GeminiEmbeddingProvider(
            GeminiEmbeddingConfig.from_env(_gemini_env()),
            client=client,
        )

        with pytest.raises(EmbeddingProviderError):
            await provider.embed(["first"])


def _ollama_env() -> dict[str, str]:
    return {
        OLLAMA_BASE_URL_ENV: "http://ollama.test",
        OLLAMA_EMBEDDING_MODEL_ENV: "embedding-model",
    }


def _gemini_env() -> dict[str, str]:
    return {
        GEMINI_API_KEY_ENV: "test-key",
        GEMINI_BASE_URL_ENV: "https://gemini.test",
        GEMINI_EMBEDDING_MODEL_ENV: "gemini-embedding-001",
        GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV: "768",
    }
