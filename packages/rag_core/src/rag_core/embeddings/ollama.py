"""Ollama embedding provider."""

from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from rag_core.embeddings.config import OllamaEmbeddingConfig
from rag_core.embeddings.exceptions import EmbeddingProviderError
from rag_core.embeddings.models import Embedding
from rag_core.embeddings.validation import validate_embedding_response, validate_texts


class OllamaEmbeddingProvider:
    """Embedding provider backed by Ollama's `/api/embed` endpoint."""

    def __init__(
        self,
        config: OllamaEmbeddingConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or OllamaEmbeddingConfig.from_env()
        self._client = client

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        batch = validate_texts(texts)
        payload: dict[str, object] = {
            "model": self.config.embedding_model,
            "input": list(batch),
        }

        if self._client is not None:
            raw_embeddings = await self._post_embed(self._client, payload)
        else:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.request_timeout_seconds,
            ) as client:
                raw_embeddings = await self._post_embed(client, payload)

        embeddings = [Embedding.from_iterable(values) for values in raw_embeddings]
        return list(validate_embedding_response(batch, embeddings))

    async def _post_embed(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, object],
    ) -> list[list[float]]:
        try:
            response = await client.post("/api/embed", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError("Ollama embedding request failed.") from exc

        response_payload: object = response.json()
        return _parse_embedding_response(response_payload)


def _parse_embedding_response(payload: object) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise EmbeddingProviderError("Ollama embedding response must be a JSON object.")

    response_payload = cast(Mapping[str, object], payload)
    embeddings = response_payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise EmbeddingProviderError("Ollama embedding response is missing embeddings.")

    raw_embeddings = cast(list[object], embeddings)
    parsed: list[list[float]] = []
    for embedding in raw_embeddings:
        if not isinstance(embedding, list):
            raise EmbeddingProviderError("Ollama embedding item must be a list.")
        values = cast(list[object], embedding)
        parsed.append([_parse_float(value) for value in values])

    return parsed


def _parse_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EmbeddingProviderError("Ollama embedding values must be numeric.")
    return float(value)
