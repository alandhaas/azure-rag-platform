"""Google AI Studio Gemini embedding provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from rag_core.embeddings.config import GeminiEmbeddingConfig
from rag_core.embeddings.exceptions import EmbeddingProviderError
from rag_core.embeddings.models import Embedding
from rag_core.embeddings.validation import validate_embedding_response, validate_texts


class GeminiEmbeddingProvider:
    """Embedding provider backed by the Gemini API `batchEmbedContents` endpoint."""

    def __init__(
        self,
        config: GeminiEmbeddingConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or GeminiEmbeddingConfig.from_env()
        self._client = client

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        batch = validate_texts(texts)
        payload = _embedding_payload(batch, self.config)

        if self._client is not None:
            raw_embeddings = await self._post_embeddings(self._client, payload)
        else:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.request_timeout_seconds,
            ) as client:
                raw_embeddings = await self._post_embeddings(client, payload)

        embeddings = [Embedding.from_iterable(values) for values in raw_embeddings]
        return list(
            validate_embedding_response(
                batch,
                embeddings,
                expected_dimension=self.config.output_dimensionality,
            )
        )

    async def _post_embeddings(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, object],
    ) -> list[list[float]]:
        model = _model_resource_name(self.config.embedding_model)
        try:
            response = await client.post(
                f"/{model}:batchEmbedContents",
                headers={"x-goog-api-key": self.config.api_key},
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError("Gemini embedding request failed.") from exc

        response_payload: object = response.json()
        return _parse_embedding_response(response_payload)


def _embedding_payload(
    texts: Sequence[str],
    config: GeminiEmbeddingConfig,
) -> dict[str, object]:
    model = _model_resource_name(config.embedding_model)
    requests: list[dict[str, object]] = []
    for text in texts:
        request: dict[str, object] = {
            "model": model,
            "content": {"parts": [{"text": text}]},
        }
        if config.output_dimensionality is not None:
            request["outputDimensionality"] = config.output_dimensionality
            request["embedContentConfig"] = {
                "outputDimensionality": config.output_dimensionality
            }
        requests.append(request)
    return {"requests": requests}


def _model_resource_name(model: str) -> str:
    stripped = model.strip()
    if stripped.startswith("models/"):
        return stripped
    return f"models/{stripped}"


def _parse_embedding_response(payload: object) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise EmbeddingProviderError("Gemini embedding response must be a JSON object.")

    response_payload = cast(Mapping[str, object], payload)
    embeddings = response_payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise EmbeddingProviderError("Gemini embedding response is missing embeddings.")

    raw_embeddings = cast(list[object], embeddings)
    parsed: list[list[float]] = []
    for embedding in raw_embeddings:
        if not isinstance(embedding, Mapping):
            raise EmbeddingProviderError("Gemini embedding item must be a JSON object.")
        values = cast(Mapping[str, object], embedding).get("values")
        if not isinstance(values, list):
            raise EmbeddingProviderError("Gemini embedding item is missing values.")
        raw_values = cast(list[object], values)
        parsed.append([_parse_float(value) for value in raw_values])

    return parsed


def _parse_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EmbeddingProviderError("Gemini embedding values must be numeric.")
    return float(value)
