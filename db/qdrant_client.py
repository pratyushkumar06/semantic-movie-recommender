"""Qdrant vector database client and configuration management.

This module handles initialization and configuration of the Qdrant vector
database client. It manages multiple vector indices for different chunking
strategies and provides utilities for collection setup.

Constants:
    COLLECTION_NAME: Default collection name in Qdrant database.
    VECTOR_NAMES: List of vector field names corresponding to chunking strategies.
"""
from __future__ import annotations

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    Language,
    VectorParams,
)

COLLECTION_NAME = "my_movies"
DENSE_VECTOR_NAME = "dense"
VECTOR_NAMES = [DENSE_VECTOR_NAME]

_CLIENT: QdrantClient | None = None


def _client_config() -> dict:
    """Build Qdrant client configuration from environment variables.
    
    Reads QDRANT_URL and QDRANT_API_KEY from environment. Falls back to
    localhost:6333 if URL not specified.
    
    Returns:
        Dictionary of configuration parameters for QdrantClient.
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return {"url": url, "api_key": api_key}
    return {"url": url}


def get_client() -> QdrantClient:
    """Get or create a singleton Qdrant client instance.
    
    Uses lazy initialization and caching to ensure efficient reuse
    of the database connection.
    
    Returns:
        QdrantClient instance.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QdrantClient(**_client_config())
    return _CLIENT


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="year",
        field_schema=PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="director_norm",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="cast_norm",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="themes_norm",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="name_norm",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="sparse_text",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.WORD,
            lowercase=True,
            stopwords=Language.ENGLISH,
        ),
    )


def recreate_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME, vector_size: int = 384) -> None:
    """Create or recreate a Qdrant collection for movie-level retrieval.
    
    Initializes a collection with a single dense vector field and
    payload indexes for filtering and BM25 text search.
    
    Args:
        client: Qdrant client instance.
        collection_name: Name of collection to create.
        vector_size: Dimensionality of embeddings (default 384 for MiniLM).
    """
    vectors = {DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)}
    client.recreate_collection(collection_name=collection_name, vectors_config=vectors)
    _create_payload_indexes(client, collection_name)
