"""Data ingestion pipeline for loading movies into the Qdrant vector database.

This module performs movie-level ingestion:
1. Load movie data
2. Normalize payload fields
3. Build sparse_text (BM25)
4. Generate dense embeddings for descriptions
5. Store one point per movie in Qdrant
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client.models import PointStruct

from data.movies import movies
from db.qdrant_client import COLLECTION_NAME, DENSE_VECTOR_NAME, get_client, recreate_collection
from embeddings.encoder import EMBEDDING_SIZE, MODEL_NAME, embed_text, load_encoder


def ingest_movies(
    collection_name: str = COLLECTION_NAME,
    recreate: bool = True,
) -> Tuple[int, int]:
    """Ingest movie descriptions into Qdrant as one point per movie.

    Args:
        collection_name: Qdrant collection name.
        recreate: If True, recreate the collection before ingesting.

    Returns:
        Tuple of (total_points_upserted, total_movies_processed).
    """
    encoder = load_encoder(MODEL_NAME)

    client = get_client()

    if recreate:
        recreate_collection(client, collection_name=collection_name, vector_size=EMBEDDING_SIZE)

    points: List[PointStruct] = []
    movies_skipped: List[str] = []

    for movie in movies:
        valid, reason = _validate_movie(movie)
        if not valid:
            movies_skipped.append(f"{movie.get('name', '<unknown>')}: {reason}")
            continue

        payload = _build_payload(movie)
        dense_vector = embed_text(encoder.model, payload["description"])
        point_id = _movie_uuid(payload["name"], payload["year"])

        points.append(
            PointStruct(
                id=point_id,
                vector={DENSE_VECTOR_NAME: dense_vector},
                payload=payload,
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)

    model_version = _get_model_version(encoder.model)
    _log_ingestion_summary(len(movies), len(points), movies_skipped, model_version)
    return len(points), len(movies)


def _validate_movie(movie: dict) -> Tuple[bool, str]:
    required = ["name", "description", "year", "director", "cast", "themes"]
    for key in required:
        if key not in movie:
            return False, f"missing field '{key}'"
    if not isinstance(movie["name"], str) or not movie["name"].strip():
        return False, "empty name"
    if not isinstance(movie["description"], str) or not movie["description"].strip():
        return False, "empty description"
    if not isinstance(movie["year"], int):
        return False, "year is not int"
    if not isinstance(movie["director"], str) or not movie["director"].strip():
        return False, "empty director"
    if not isinstance(movie["cast"], list) or not movie["cast"]:
        return False, "cast is empty or not a list"
    if not isinstance(movie["themes"], list) or not movie["themes"]:
        return False, "themes is empty or not a list"
    return True, ""


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _build_sparse_text(name: str, director: str, cast: List[str], themes: List[str]) -> str:
    parts = [name, director] + cast + themes
    return " ".join([part for part in parts if part])


def _movie_uuid(name: str, year: int) -> str:
    return str(uuid5(NAMESPACE_DNS, f"{name}|{year}"))


def _build_payload(movie: dict) -> dict:
    name = movie["name"].strip()
    description = movie["description"].strip()
    director = movie["director"].strip()
    cast = _dedupe_preserve_order(movie["cast"])
    themes = _dedupe_preserve_order(movie["themes"])

    payload = {
        "name": name,
        "description": description,
        "year": movie["year"],
        "director": director,
        "cast": cast,
        "themes": themes,
        "name_norm": name.lower(),
        "director_norm": director.lower(),
        "cast_norm": [c.lower() for c in cast],
        "themes_norm": [t.lower() for t in themes],
        "sparse_text": _build_sparse_text(name, director, cast, themes),
    }
    return payload


def _get_model_version(model) -> str | None:
    candidates: List[str | None] = [
        getattr(model, "model_name_or_path", None),
        getattr(model, "model_card", None),
    ]
    try:
        auto_model = model._first_module().auto_model  # type: ignore[attr-defined]
        candidates.append(getattr(auto_model.config, "_name_or_path", None))
        candidates.append(getattr(auto_model.config, "name_or_path", None))
    except Exception:
        pass
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _log_ingestion_summary(
    total_movies: int,
    total_points: int,
    skipped: List[str],
    model_version: str | None,
) -> None:
    print(f"Ingestion summary:")
    print(f"- total_movies_seen: {total_movies}")
    print(f"- total_points_upserted: {total_points}")
    print(f"- movies_skipped: {len(skipped)}")
    if skipped:
        for item in skipped:
            print(f"  - {item}")
    print(f"- embedding_model: {MODEL_NAME}")
    if model_version:
        print(f"- embedding_model_version: {model_version}")
    print(f"- embedding_dim: {EMBEDDING_SIZE}")


if __name__ == "__main__":
    total_points, total_movies = ingest_movies()
    print(f"Ingested {total_movies} movies -> {total_points} points")
