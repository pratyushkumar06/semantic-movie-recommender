"""Information retrieval metrics."""
from __future__ import annotations

from typing import Iterable, List, Sequence


def precision_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Precision@K = relevant in top K / K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Recall@K = relevant in top K / total relevant."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def mrr(relevant_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """MRR for a single query based on first relevant item."""
    relevant_set = set(relevant_ids)
    for idx, item in enumerate(retrieved_ids, 1):
        if item in relevant_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str], k: int) -> float:
    """nDCG@K with binary relevance."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)

    def dcg(items: Iterable[str]) -> float:
        score = 0.0
        for idx, item in enumerate(items, 1):
            if item in relevant_set:
                score += 1.0 / _log2(idx + 1)
        return score

    top_k = retrieved_ids[:k]
    ideal = list(relevant_set)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(top_k) / ideal_dcg


def _log2(value: int) -> float:
    from math import log

    return log(value, 2)
