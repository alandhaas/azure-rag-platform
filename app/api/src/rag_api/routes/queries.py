"""Query-related API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from rag_core.embeddings import EmbeddingProvider
from rag_core.vectorstore import VectorStore

from rag_api.dependencies import get_embedding_provider, get_vector_store
from rag_api.services.retrieval import retrieve_query_matches

router = APIRouter(prefix="/queries", tags=["queries"])

PayloadFilterValue = bool | int | str


class QueryEmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)


class QueryRetrievalRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filter: dict[str, PayloadFilterValue] | None = None


class QueryRetrievalResult(BaseModel):
    point_id: str
    score: float
    payload: dict[str, object]


class QueryRetrievalResponse(BaseModel):
    results: list[QueryRetrievalResult]


@router.post(
    "/retrieval",
    response_model=QueryRetrievalResponse,
    summary="Retrieve query matches",
    description="Embeds a query string and searches the configured vector store.",
)
async def retrieve_query(
    request: QueryRetrievalRequest,
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> QueryRetrievalResponse:
    results = await retrieve_query_matches(
        text=request.text,
        top_k=request.top_k,
        payload_filter=request.filter,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    return QueryRetrievalResponse(
        results=[
            QueryRetrievalResult(
                point_id=result.point_id,
                score=result.score,
                payload=dict(result.payload),
            )
            for result in results
        ]
    )
