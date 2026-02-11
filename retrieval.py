"""Retrieval pipelines for movie recommendations (baseline: dense-only).

This module provides a unified retrieve() entry point and implements the
dense-only baseline strategy. Other strategies will be added incrementally.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import math
import re

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Bm25Config,
    Document,
    FieldCondition,
    Filter,
    HasIdCondition,
    Language,
    MatchAny,
    MatchValue,
    Range,
    TokenizerType,
)

from db.qdrant_client import COLLECTION_NAME, DENSE_VECTOR_NAME, get_client
from embeddings.encoder import MODEL_NAME, embed_text, load_encoder


def retrieve(
    query: str,
    strategy: str,
    params: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """Unified retrieval entry point.

    Args:
        query: Natural language query string.
        strategy: Retrieval strategy name (currently supports "dense_only").
        params: Strategy-specific parameters (e.g., dense_top_k).
        filters: Hard filters (year range, director, cast, themes).
        collection_name: Qdrant collection to search.

    Returns:
        Ranked list of results with dense_score, sparse_score, final_score,
        and match_explanation.
    """
    if params is None:
        params = {}
    if strategy == "dense_only":
        return _retrieve_dense_only(query, params, filters, collection_name)
    if strategy == "sparse_only":
        return _retrieve_sparse_only(query, params, filters, collection_name)
    if strategy == "dense_recall_sparse_rerank":
        return _retrieve_dense_recall_sparse_rerank(query, params, filters, collection_name)
    if strategy == "sparse_prefilter_dense_rank":
        return _retrieve_sparse_prefilter_dense_rank(query, params, filters, collection_name)
    if strategy == "sparse_recall_dense_rerank":
        return _retrieve_sparse_recall_dense_rerank(query, params, filters, collection_name)
    if strategy == "hybrid_combined":
        return _retrieve_hybrid_combined(query, params, filters, collection_name)
    raise ValueError(f"Unsupported strategy: {strategy}")


def _retrieve_dense_only(
    query: str,
    params: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    dense_top_k = int(params.get("dense_top_k", 10))
    query_filter = _build_filter(filters)

    encoder = load_encoder(MODEL_NAME)
    query_vector = embed_text(encoder.model, query)

    client = get_client()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=dense_top_k,
        with_payload=True,
        with_vectors=False,
    )

    results: List[Dict[str, Any]] = []
    for hit in response.points:
        payload = hit.payload or {}
        dense_score = hit.score
        if dense_score is None:
            continue
        results.append(
            {
                "id": str(hit.id),
                "name": payload.get("name"),
                "dense_score": dense_score,
                "sparse_score": None,
                "final_score": dense_score,
                "score": dense_score,
                "match_explanation": _explain_dense_only(filters),
                "year": payload.get("year"),
                "director": payload.get("director"),
                "cast": payload.get("cast"),
                "themes": payload.get("themes"),
            }
        )
    return results


def _retrieve_sparse_only(
    query: str,
    params: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    sparse_top_k = int(params.get("sparse_top_k", 10))
    bm25_k1 = float(params.get("bm25_k1", 1.2))
    bm25_b = float(params.get("bm25_b", 0.75))
    backend = params.get("bm25_backend", "qdrant")

    client = get_client()
    query_filter = _build_filter(filters)

    if backend == "qdrant":
        try:
            bm25_config = Bm25Config(
                k=bm25_k1,
                b=bm25_b,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                stopwords=Language.ENGLISH,
            )
            bm25_query = Document(text=query, model="Qdrant/Bm25", options=bm25_config)
            response = client.query_points(
                collection_name=collection_name,
                query=bm25_query,
                query_filter=query_filter,
                limit=sparse_top_k,
                with_payload=True,
                with_vectors=False,
            )
            return _format_sparse_results(query, response.points, filters)
        except (UnexpectedResponse, ValueError):
            # Fall back to local BM25 scoring if Qdrant BM25 is unavailable
            pass

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=query_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return _local_bm25_rank(query, points, sparse_top_k, bm25_k1, bm25_b, filters)


def _retrieve_dense_recall_sparse_rerank(
    query: str,
    params: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    dense_top_k = int(params.get("dense_top_k", 50))
    rerank_depth = int(params.get("rerank_depth", dense_top_k))
    bm25_k1 = float(params.get("bm25_k1", 1.2))
    bm25_b = float(params.get("bm25_b", 0.75))
    rerank_mode = params.get("rerank_mode", "fusion")  # "fusion" or "sparse"
    fusion_alpha = float(params.get("fusion_alpha", 0.5))
    fusion_beta = float(params.get("fusion_beta", 0.5))
    score_norm = params.get("score_norm", "minmax")

    query_filter = _build_filter(filters)

    encoder = load_encoder(MODEL_NAME)
    query_vector = embed_text(encoder.model, query)

    client = get_client()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=dense_top_k,
        with_payload=True,
        with_vectors=False,
    )

    candidates: List[Dict[str, Any]] = []
    for hit in response.points:
        payload = hit.payload or {}
        if hit.score is None:
            continue
        candidates.append(
            {
                "id": str(hit.id),
                "payload": payload,
                "dense_score": hit.score,
            }
        )

    if not candidates:
        return []

    rerank_depth = min(rerank_depth, len(candidates))
    candidates = candidates[:rerank_depth]

    query_tokens = _tokenize(query)
    docs = [(c["id"], _tokenize(c["payload"].get("sparse_text") or "")) for c in candidates]
    sparse_scores = _bm25_scores(query_tokens, docs, bm25_k1, bm25_b)

    dense_vals = [c["dense_score"] for c in candidates]
    sparse_vals = [sparse_scores.get(c["id"], 0.0) for c in candidates]

    if rerank_mode == "sparse":
        final_scores = sparse_vals
    else:
        dense_norm = _normalize_scores(dense_vals, score_norm)
        sparse_norm = _normalize_scores(sparse_vals, score_norm)
        final_scores = [
            fusion_alpha * d + fusion_beta * s for d, s in zip(dense_norm, sparse_norm)
        ]

    results: List[Dict[str, Any]] = []
    for candidate, sparse_score, final_score in zip(candidates, sparse_vals, final_scores):
        payload = candidate["payload"]
        results.append(
            {
                "id": candidate["id"],
                "name": payload.get("name"),
                "dense_score": candidate["dense_score"],
                "sparse_score": sparse_score,
                "final_score": final_score,
                "score": final_score,
                "match_explanation": _explain_dense_then_sparse(query, payload, filters, sparse_score),
                "year": payload.get("year"),
                "director": payload.get("director"),
                "cast": payload.get("cast"),
                "themes": payload.get("themes"),
            }
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


def _retrieve_sparse_prefilter_dense_rank(
    query: str,
    params: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    sparse_top_k = int(params.get("sparse_top_k", 50))
    dense_top_k = int(params.get("dense_top_k", 10))
    bm25_k1 = float(params.get("bm25_k1", 1.2))
    bm25_b = float(params.get("bm25_b", 0.75))
    backend = params.get("bm25_backend", "qdrant")

    sparse_params = {
        "sparse_top_k": sparse_top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_backend": backend,
    }
    sparse_results = _retrieve_sparse_only(query, sparse_params, filters, collection_name)
    if not sparse_results:
        return []

    candidate_ids = [r["id"] for r in sparse_results]
    sparse_by_id = {r["id"]: r for r in sparse_results}

    encoder = load_encoder(MODEL_NAME)
    query_vector = embed_text(encoder.model, query)

    id_filter = Filter(must=[HasIdCondition(has_id=candidate_ids)])

    client = get_client()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        query_filter=id_filter,
        limit=len(candidate_ids),
        with_payload=True,
        with_vectors=False,
    )

    results: List[Dict[str, Any]] = []
    for hit in response.points:
        payload = hit.payload or {}
        dense_score = hit.score
        if dense_score is None:
            continue
        sparse_score = sparse_by_id.get(str(hit.id), {}).get("sparse_score")
        results.append(
            {
                "id": str(hit.id),
                "name": payload.get("name"),
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "final_score": dense_score,
                "score": dense_score,
                "match_explanation": _explain_sparse_prefilter_dense(
                    query, payload, filters, sparse_score
                ),
                "year": payload.get("year"),
                "director": payload.get("director"),
                "cast": payload.get("cast"),
                "themes": payload.get("themes"),
            }
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:dense_top_k]


def _retrieve_sparse_recall_dense_rerank(
    query: str,
    params: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    sparse_top_k = int(params.get("sparse_top_k", 50))
    dense_top_k = int(params.get("dense_top_k", 10))
    rerank_depth = int(params.get("rerank_depth", sparse_top_k))
    bm25_k1 = float(params.get("bm25_k1", 1.2))
    bm25_b = float(params.get("bm25_b", 0.75))
    rerank_mode = params.get("rerank_mode", "dense")  # "dense" or "fusion"
    fusion_alpha = float(params.get("fusion_alpha", 0.5))
    fusion_beta = float(params.get("fusion_beta", 0.5))
    score_norm = params.get("score_norm", "minmax")
    backend = params.get("bm25_backend", "qdrant")

    sparse_params = {
        "sparse_top_k": sparse_top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_backend": backend,
    }
    sparse_results = _retrieve_sparse_only(query, sparse_params, filters, collection_name)
    if not sparse_results:
        return []

    rerank_depth = min(rerank_depth, len(sparse_results))
    candidates = sparse_results[:rerank_depth]
    candidate_ids = [r["id"] for r in candidates]
    sparse_scores = {r["id"]: r["sparse_score"] for r in candidates}

    encoder = load_encoder(MODEL_NAME)
    query_vector = embed_text(encoder.model, query)

    id_filter = Filter(must=[HasIdCondition(has_id=candidate_ids)])

    client = get_client()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        query_filter=id_filter,
        limit=len(candidate_ids),
        with_payload=True,
        with_vectors=False,
    )

    dense_scores = {str(hit.id): hit.score for hit in response.points if hit.score is not None}
    dense_vals = [dense_scores.get(cid, 0.0) or 0.0 for cid in candidate_ids]
    sparse_vals = [sparse_scores.get(cid, 0.0) or 0.0 for cid in candidate_ids]

    if rerank_mode == "fusion":
        dense_norm = _normalize_scores(dense_vals, score_norm)
        sparse_norm = _normalize_scores(sparse_vals, score_norm)
        final_scores = [
            fusion_alpha * d + fusion_beta * s for d, s in zip(dense_norm, sparse_norm)
        ]
    else:
        final_scores = dense_vals

    results: List[Dict[str, Any]] = []
    for candidate, dense_score, sparse_score, final_score in zip(
        candidates, dense_vals, sparse_vals, final_scores
    ):
        results.append(
            {
                "id": candidate["id"],
                "name": candidate.get("name"),
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "final_score": final_score,
                "score": final_score,
                "match_explanation": _explain_sparse_then_dense(
                    query, candidate, filters, dense_score
                ),
                "year": candidate.get("year"),
                "director": candidate.get("director"),
                "cast": candidate.get("cast"),
                "themes": candidate.get("themes"),
            }
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:dense_top_k]


def _retrieve_hybrid_combined(
    query: str,
    params: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    dense_top_k = int(params.get("dense_top_k", 50))
    sparse_top_k = int(params.get("sparse_top_k", 50))
    bm25_k1 = float(params.get("bm25_k1", 1.2))
    bm25_b = float(params.get("bm25_b", 0.75))
    fusion_alpha = float(params.get("fusion_alpha", 0.5))
    fusion_beta = float(params.get("fusion_beta", 0.5))
    score_norm = params.get("score_norm", "minmax")
    backend = params.get("bm25_backend", "qdrant")

    dense_params = {"dense_top_k": dense_top_k}
    sparse_params = {
        "sparse_top_k": sparse_top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_backend": backend,
    }

    dense_results = _retrieve_dense_only(query, dense_params, filters, collection_name)
    sparse_results = _retrieve_sparse_only(query, sparse_params, filters, collection_name)

    candidates: Dict[str, Dict[str, Any]] = {}
    for result in dense_results:
        candidates[result["id"]] = result.copy()
    for result in sparse_results:
        if result["id"] in candidates:
            candidates[result["id"]].update(result)
        else:
            candidates[result["id"]] = result.copy()

    if not candidates:
        return []

    ordered_ids = list(candidates.keys())
    dense_vals = [candidates[cid].get("dense_score") or 0.0 for cid in ordered_ids]
    sparse_vals = [candidates[cid].get("sparse_score") or 0.0 for cid in ordered_ids]

    dense_norm = _normalize_scores(dense_vals, score_norm)
    sparse_norm = _normalize_scores(sparse_vals, score_norm)
    final_scores = [
        fusion_alpha * d + fusion_beta * s for d, s in zip(dense_norm, sparse_norm)
    ]

    results: List[Dict[str, Any]] = []
    for cid, dense_score, sparse_score, final_score in zip(
        ordered_ids, dense_vals, sparse_vals, final_scores
    ):
        payload = candidates[cid]
        results.append(
            {
                "id": cid,
                "name": payload.get("name"),
                "dense_score": dense_score if dense_score != 0.0 else payload.get("dense_score"),
                "sparse_score": sparse_score if sparse_score != 0.0 else payload.get("sparse_score"),
                "final_score": final_score,
                "score": final_score,
                "match_explanation": _explain_hybrid_combined(
                    query, payload, filters, dense_score, sparse_score
                ),
                "year": payload.get("year"),
                "director": payload.get("director"),
                "cast": payload.get("cast"),
                "themes": payload.get("themes"),
            }
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[: max(dense_top_k, sparse_top_k)]


def _build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filters:
        return None

    must: List[FieldCondition] = []
    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min is not None or year_max is not None:
        must.append(FieldCondition(key="year", range=Range(gte=year_min, lte=year_max)))

    director = _normalize_value(filters.get("director"))
    if director:
        must.append(FieldCondition(key="director_norm", match=MatchValue(value=director)))

    cast = _normalize_list(filters.get("cast_includes"))
    if cast:
        must.append(FieldCondition(key="cast_norm", match=MatchAny(any=cast)))

    themes = _normalize_list(filters.get("themes_includes"))
    if themes:
        must.append(FieldCondition(key="themes_norm", match=MatchAny(any=themes)))

    name = _normalize_value(filters.get("name"))
    if name:
        must.append(FieldCondition(key="name_norm", match=MatchValue(value=name)))

    if not must:
        return None
    return Filter(must=must)


def _normalize_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _normalize_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        return []
    normalized: List[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _explain_dense_only(filters: Optional[Dict[str, Any]]) -> str:
    if not filters:
        return "Semantic match to description."
    return "Semantic match to description; hard filters applied."


def _explain_sparse_only(query: str, payload: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> str:
    matches = _sparse_match_details(query, payload)
    if filters:
        matches.append("hard filters applied")
    if not matches:
        return "BM25 keyword hit."
    return "; ".join(matches)


def _sparse_match_details(query: str, payload: Dict[str, Any]) -> List[str]:
    query_norm = (query or "").strip().lower()
    matches: List[str] = []

    name_norm = (payload.get("name_norm") or "").lower()
    if name_norm and name_norm in query_norm:
        matches.append(f"matched title: {payload.get('name')}")

    director_norm = (payload.get("director_norm") or "").lower()
    if director_norm and director_norm in query_norm:
        matches.append(f"matched director: {payload.get('director')}")

    cast = payload.get("cast") or []
    cast_norm = payload.get("cast_norm") or []
    for orig, norm in zip(cast, cast_norm):
        if norm and norm in query_norm:
            matches.append(f"matched cast: {orig}")

    themes = payload.get("themes") or []
    themes_norm = payload.get("themes_norm") or []
    matched_themes = [orig for orig, norm in zip(themes, themes_norm) if norm and norm in query_norm]
    if matched_themes:
        matches.append(f"matched themes: {matched_themes}")

    return matches


def _explain_dense_then_sparse(
    query: str,
    payload: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    sparse_score: float,
) -> str:
    base = "Semantic match after dense recall"
    matches = _sparse_match_details(query, payload)
    if sparse_score > 0 and matches:
        base += f"; boosted by sparse hit ({', '.join(matches)})"
    elif sparse_score > 0:
        base += "; boosted by sparse hit"
    if filters:
        base += "; hard filters applied"
    return base


def _explain_sparse_prefilter_dense(
    query: str,
    payload: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    sparse_score: Optional[float],
) -> str:
    base = "Sparse prefilter then dense rank"
    matches = _sparse_match_details(query, payload)
    if sparse_score and matches:
        base += f"; prefilter matched ({', '.join(matches)})"
    elif sparse_score:
        base += "; prefilter matched"
    if filters:
        base += "; hard filters applied"
    return base


def _explain_sparse_then_dense(
    query: str,
    payload: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    dense_score: float,
) -> str:
    base = "Sparse recall then dense rerank"
    matches = _sparse_match_details(query, payload)
    if matches:
        base += f"; recall matched ({', '.join(matches)})"
    if dense_score > 0:
        base += "; reranked by semantic similarity"
    if filters:
        base += "; hard filters applied"
    return base


def _explain_hybrid_combined(
    query: str,
    payload: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    dense_score: float,
    sparse_score: float,
) -> str:
    base = "Hybrid fusion of dense and sparse"
    matches = _sparse_match_details(query, payload)
    if sparse_score > 0 and matches:
        base += f"; sparse matched ({', '.join(matches)})"
    elif sparse_score > 0:
        base += "; sparse matched"
    if dense_score > 0:
        base += "; dense similarity"
    if filters:
        base += "; hard filters applied"
    return base


def _format_sparse_results(
    query: str,
    points: Iterable[Any],
    filters: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for hit in points:
        payload = hit.payload or {}
        sparse_score = hit.score
        if sparse_score is None:
            continue
        results.append(
            {
                "id": str(hit.id),
                "name": payload.get("name"),
                "dense_score": None,
                "sparse_score": sparse_score,
                "final_score": sparse_score,
                "score": sparse_score,
                "match_explanation": _explain_sparse_only(query, payload, filters),
                "year": payload.get("year"),
                "director": payload.get("director"),
                "cast": payload.get("cast"),
                "themes": payload.get("themes"),
            }
        )
    return results


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _local_bm25_rank(
    query: str,
    points: Iterable[Any],
    limit: int,
    k1: float,
    b: float,
    filters: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    point_payloads: Dict[str, Dict[str, Any]] = {}
    docs: List[tuple[str, List[str]]] = []
    for point in points:
        payload = point.payload or {}
        point_id = str(point.id)
        point_payloads[point_id] = payload
        tokens = _tokenize(payload.get("sparse_text") or "")
        docs.append((point_id, tokens))

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = _bm25_scores(query_tokens, docs, k1, b)
    scored = []
    for point_id, score in scores.items():
        if score <= 0:
            continue
        payload = point_payloads[point_id]
        scored.append(
            {
                "id": point_id,
                "name": payload.get("name"),
                "dense_score": None,
                "sparse_score": score,
                "final_score": score,
                "score": score,
                "match_explanation": _explain_sparse_only(query, payload, filters),
                "year": payload.get("year"),
                "director": payload.get("director"),
                "cast": payload.get("cast"),
                "themes": payload.get("themes"),
            }
        )

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    return scored[:limit]


def _bm25_scores(
    query_tokens: List[str],
    docs: List[tuple[str, List[str]]],
    k1: float,
    b: float,
) -> Dict[str, float]:
    if not query_tokens:
        return {}

    doc_lens = [len(tokens) for _, tokens in docs]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0

    df: Dict[str, int] = {}
    for _, tokens in docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    scores: Dict[str, float] = {}
    N = len(docs)
    for (doc_id, tokens), dl in zip(docs, doc_lens):
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            n_qi = df.get(token, 0)
            idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1)
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * (dl / avgdl))
            score += idf * (freq * (k1 + 1) / denom)
        scores[doc_id] = score

    return scores


def _normalize_scores(values: List[float], method: str) -> List[float]:
    if not values:
        return []
    if method == "zscore":
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var) if var > 0 else 1.0
        return [(v - mean) / std for v in values]
    if method == "minmax":
        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]
    return values
