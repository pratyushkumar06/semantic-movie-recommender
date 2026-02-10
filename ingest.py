"""Data ingestion pipeline for loading movies into the Qdrant vector database.

This module orchestrates the end-to-end data ingestion process:
1. Load movie data
2. Apply multiple chunking strategies
3. Generate embeddings for each chunk
4. Store vectors and metadata in Qdrant

The ingestion process creates multiple vector fields for comparative analysis
of different chunking strategies on search quality and performance.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from qdrant_client.models import PointStruct

from chunking.chunkers import Chunkers
from data.movies import movies
from db.qdrant_client import COLLECTION_NAME, VECTOR_NAMES, get_client, recreate_collection
from embeddings.encoder import (
    EMBEDDING_SIZE,
    MODEL_NAME,
    TOKEN_LIMIT,
    embed_texts,
    inspect_text_tokens,
    load_encoder,
)


def _chunk_movie(text: str, chunkers: Chunkers) -> Dict[str, List[str]]:
    """Apply all chunking strategies to movie description text.
    
    Args:
        text: Movie description to chunk.
        chunkers: Chunkers instance with all strategies initialized.
        
    Returns:
        Dictionary mapping strategy names to lists of chunks.
    """
    return {
        "fixed_chunk": chunkers.fixed_token_chunks(text),
        "sentence_chunk": chunkers.sentence_chunks(text),
        "semantic_chunk": chunkers.semantic_chunks(text),
    }


def ingest_movies(
    token_limit: int = TOKEN_LIMIT,
    overlap: int = 40,
    collection_name: str = COLLECTION_NAME,
    show_token_report: bool = False,
) -> Tuple[int, int]:
    """Ingest movie descriptions into Qdrant with multiple chunking strategies.
    
    Performs full pipeline: load models, chunk texts, generate embeddings,
    store in database. Each movie generates multiple vectors (one per strategy).
    
    Args:
        token_limit: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.
        collection_name: Qdrant collection name.
        show_token_report: If True, print token analysis before ingesting.
        
    Returns:
        Tuple of (total_vectors_created, total_movies_processed).
    """
    encoder = load_encoder(MODEL_NAME)
    chunkers = Chunkers(
        token_limit=token_limit,
        overlap=overlap,
        tokenizer=encoder.tokenizer,
        semantic_model_name=MODEL_NAME,
    )

    if show_token_report:
        for movie in movies:
            print(inspect_text_tokens(movie["name"], movie["description"], encoder.tokenizer, token_limit))
            print("-")

    client = get_client()
    recreate_collection(client, collection_name=collection_name, vector_size=EMBEDDING_SIZE)

    points: List[PointStruct] = []
    point_id = 1

    for movie in movies:
        chunk_sets = _chunk_movie(movie["description"], chunkers)

        for method_name in VECTOR_NAMES:
            chunks = chunk_sets[method_name]
            if not chunks:
                continue
            embeddings = embed_texts(encoder.model, chunks)
            for chunk_text, embedding in zip(chunks, embeddings):
                payload = {
                    "movie_name": movie["name"],
                    "year": movie["year"],
                    "author": movie["author"],
                    "chunk_method": method_name,
                    "chunk_text": chunk_text,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={method_name: embedding},
                        payload=payload,
                    )
                )
                point_id += 1

    client.upload_points(collection_name=collection_name, points=points, wait=True)
    return len(points), len(movies)


if __name__ == "__main__":
    total_points, total_movies = ingest_movies(token_limit=TOKEN_LIMIT, overlap=40, show_token_report=True)
    print(f"Ingested {total_movies} movies -> {total_points} vectors")
