"""Retrieval service helpers."""

from collections.abc import Mapping

from rag_core.embeddings import EmbeddingProvider
from rag_core.vectorstore import VectorSearchQuery, VectorSearchResult, VectorStore


async def retrieve_query_matches(
    *,
    text: str,
    top_k: int,
    payload_filter: Mapping[str, object] | None,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> list[VectorSearchResult]:
    embedding = (await embedding_provider.embed([text]))[0]
    return await vector_store.search(
        VectorSearchQuery(
            embedding=embedding,
            top_k=top_k,
            filter=payload_filter,
        )
    )
