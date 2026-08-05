"""Qdrant vector-store adapter."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import environ as process_environ
from typing import Final, Protocol, Self, cast

from qdrant_client import AsyncQdrantClient, models

from rag_core.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreProviderError,
    VectorStoreValidationError,
)
from rag_core.vectorstore.models import VectorPoint, VectorSearchQuery, VectorSearchResult

QDRANT_URL_ENV: Final = "QDRANT_URL"
QDRANT_COLLECTION_NAME_ENV: Final = "QDRANT_COLLECTION_NAME"
QDRANT_API_KEY_ENV: Final = "QDRANT_API_KEY"
QDRANT_VECTOR_SIZE_ENV: Final = "QDRANT_VECTOR_SIZE"


class _QdrantClient(Protocol):
    async def upsert(
        self,
        *,
        collection_name: str,
        points: Sequence[models.PointStruct],
    ) -> object: ...

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        query_filter: models.Filter | None,
        limit: int,
        with_payload: bool,
    ) -> object: ...

    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: models.PointIdsList,
    ) -> object: ...

    async def collection_exists(self, collection_name: str) -> bool: ...

    async def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: models.VectorParams,
    ) -> object: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class QdrantVectorStoreConfig:
    """Runtime configuration for Qdrant vector-store operations."""

    url: str
    collection_name: str
    api_key: str | None = None
    vector_size: int = 768
    request_timeout_seconds: int = 30

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = environ or process_environ
        return cls(
            url=_required_env(env, QDRANT_URL_ENV),
            collection_name=_required_env(env, QDRANT_COLLECTION_NAME_ENV),
            api_key=_optional_env(env, QDRANT_API_KEY_ENV),
            vector_size=_optional_positive_int_env(env, QDRANT_VECTOR_SIZE_ENV, default=768),
        )


class QdrantVectorStore:
    """Vector-store adapter backed by a Qdrant collection."""

    def __init__(
        self,
        config: QdrantVectorStoreConfig,
        *,
        client: _QdrantClient | None = None,
    ) -> None:
        self.config = config
        self._collection_ready = False
        self._client = client or cast(
            _QdrantClient,
            AsyncQdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.request_timeout_seconds,
                check_compatibility=False,
            ),
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return

        await self._ensure_collection()
        await self._client.upsert(
            collection_name=self.config.collection_name,
            points=[_to_point_struct(point) for point in points],
        )

    async def search(self, query: VectorSearchQuery) -> list[VectorSearchResult]:
        await self._ensure_collection()
        response = await self._client.query_points(
            collection_name=self.config.collection_name,
            query=list(query.embedding.values),
            query_filter=_to_qdrant_filter(query.filter),
            limit=query.top_k,
            with_payload=True,
        )
        return [_to_search_result(point) for point in _response_points(response)]

    async def delete(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return

        await self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
        )

    async def close(self) -> None:
        await self._client.close()

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return

        if not await self._client.collection_exists(self.config.collection_name):
            await self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        self._collection_ready = True


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise VectorStoreConfigurationError(f"Missing required environment variable: {name}")
    return value


def _optional_env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _optional_positive_int_env(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    value = environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise VectorStoreConfigurationError(
            f"Environment variable {name} must be an integer."
        ) from exc
    if parsed < 1:
        raise VectorStoreConfigurationError(
            f"Environment variable {name} must be greater than zero."
        )
    return parsed


def _to_point_struct(point: VectorPoint) -> models.PointStruct:
    return models.PointStruct(
        id=point.id,
        vector=list(point.embedding.values),
        payload=dict(point.payload),
    )


def _to_qdrant_filter(payload_filter: Mapping[str, object] | None) -> models.Filter | None:
    if payload_filter is None:
        return None

    conditions = [
        cast(
            models.Condition,
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=_match_value(value)),
            ),
        )
        for key, value in payload_filter.items()
    ]
    return models.Filter(must=conditions)


def _match_value(value: object) -> bool | int | str:
    if isinstance(value, bool | int | str):
        return value
    raise VectorStoreValidationError("Qdrant filters support bool, int, and str values.")


def _response_points(response: object) -> list[object]:
    points = getattr(response, "points", None)
    if not isinstance(points, Sequence) or isinstance(points, str):
        raise VectorStoreProviderError("Qdrant query response is missing points.")
    return list(cast(Sequence[object], points))


def _to_search_result(point: object) -> VectorSearchResult:
    point_id = getattr(point, "id", None)
    score = getattr(point, "score", None)
    payload = getattr(point, "payload", None)

    if point_id is None:
        raise VectorStoreProviderError("Qdrant point is missing id.")
    if not isinstance(score, int | float):
        raise VectorStoreProviderError("Qdrant point is missing a numeric score.")
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise VectorStoreProviderError("Qdrant point payload must be a mapping.")

    return VectorSearchResult(
        point_id=str(point_id),
        score=float(score),
        payload=dict(cast(Mapping[str, object], payload)),
    )
