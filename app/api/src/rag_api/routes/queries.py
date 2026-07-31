"""Query-related API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from rag_core.embeddings import EmbeddingProvider

from rag_api.dependencies import get_embedding_provider

router = APIRouter(prefix="/queries", tags=["queries"])


class QueryEmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)


class QueryEmbeddingResponse(BaseModel):
    embedding: list[float]
    dimension: int


@router.post(
    "/embedding",
    response_model=QueryEmbeddingResponse,
    summary="Embed query text",
    description="Embeds a single query string using the configured embedding provider.",
)
async def embed_query(
    request: QueryEmbeddingRequest,
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> QueryEmbeddingResponse:
    embedding = (await embedding_provider.embed([request.text]))[0]
    return QueryEmbeddingResponse(
        embedding=list(embedding.values),
        dimension=embedding.dimension,
    )
