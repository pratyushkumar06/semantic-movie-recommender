"""Semantic search interface for querying movie recommendations.

This module provides search functionality for the movie recommendation system.
It supports vector similarity search with optional filtering and result grouping
by movie to reduce redundancy.

Functions:
    search: Perform semantic search with multiple chunking strategies.
    format_results: Format search results for display.
"""
from __future__ import annotations

from typing import List, Optional

from qdrant_client.models import FieldCondition, Filter, Range

from db.qdrant_client import COLLECTION_NAME, VECTOR_NAMES, get_client
from embeddings.encoder import MODEL_NAME, embed_text, load_encoder


def _build_filter(year_gte: Optional[int]) -> Optional[Filter]:
    """Create a Qdrant filter for minimum release year.
    
    Args:
        year_gte: Minimum year (inclusive). If None, no filter applied.
        
    Returns:
        Qdrant Filter object or None.
    """
    if year_gte is None:
        return None
    return Filter(must=[FieldCondition(key="year", range=Range(gte=year_gte))])


def search(
    query: str,
    chunk_method: str,
    k: int = 3,
    year_gte: Optional[int] = None,
    grouped: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> List[dict]:
    """Perform semantic search for movie recommendations.
    
    Args:
        query: Natural language search query.
        chunk_method: Chunking strategy ('fixed_chunk', 'sentence_chunk', or 'semantic_chunk').
        k: Number of results to return (or groups if grouped=True).
        year_gte: Filter to movies released in this year or later (optional).
        grouped: If True, return one chunk per movie; else return top k chunks.
        collection_name: Qdrant collection to search.
        
    Returns:
        List of result dictionaries with movie metadata and relevance scores.
        
    Raises:
        ValueError: If chunk_method is not a recognized strategy.
    """
    if chunk_method not in VECTOR_NAMES:
        raise ValueError(f"chunk_method must be one of {VECTOR_NAMES}")

    encoder = load_encoder(MODEL_NAME)
    query_vector = embed_text(encoder.model, query)
    client = get_client()

    query_filter = _build_filter(year_gte)

    if grouped:
        groups = client.query_points_groups(
            collection_name=collection_name,
            query=query_vector,
            using=chunk_method,
            query_filter=query_filter,
            limit=k,
            group_by="movie_name",
            group_size=1,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for group in groups.groups:
            if not group.hits:
                continue
            hit = group.hits[0]
            results.append(
                {
                    "movie": hit.payload.get("movie_name"),
                    "score": hit.score,
                    "chunk": hit.payload.get("chunk_text"),
                    "year": hit.payload.get("year"),
                    "director": hit.payload.get("director"),
                }
            )
        return results

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=chunk_method,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "movie": hit.payload.get("movie_name"),
            "score": hit.score,
            "chunk": hit.payload.get("chunk_text"),
            "year": hit.payload.get("year"),
            "director": hit.payload.get("director"),
        }
        for hit in response.points
    ]


def format_results(results: List[dict]) -> str:
    """Format search results for human-readable display.
    
    Args:
        results: List of result dictionaries from search().
        
    Returns:
        Formatted string representation of results.
    """
    lines: List[str] = []
    for result in results:
        lines.append(f"Movie: {result['movie']}")
        lines.append(f"Score: {result['score']:.4f}")
        lines.append(f"Chunk: {result['chunk']}")
        lines.append("-")
    return "\n".join(lines)


if __name__ == "__main__":
    res = search("Alien invasion", "fixed_chunk", k=3)
    print(format_results(res))
