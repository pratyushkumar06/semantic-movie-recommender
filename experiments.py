"""Experimental harness for comparing chunking strategies on movie search.

This module runs experiments comparing different text chunking strategies
(fixed tokens, sentence-aware, semantic) across various token limits and
overlap configurations. It benchmarks search quality and provides qualitative
expectations for trade-offs between speed and accuracy.
"""
from __future__ import annotations

from ingest import ingest_movies
from search import format_results, search


def run_experiment(token_limit: int, overlap: int) -> None:
    """Run a single experiment with given chunking parameters.
    
    Ingests movies, then performs searches with all chunking strategies,
    both ungrouped (top k chunks) and grouped (one chunk per movie).
    
    Args:
        token_limit: Maximum tokens per chunk.
        overlap: Token overlap between chunks.
    """
    print(f"=== Ingesting (token_limit={token_limit}, overlap={overlap}) ===")
    total_points, total_movies = ingest_movies(token_limit=token_limit, overlap=overlap)
    print(f"Ingested {total_movies} movies -> {total_points} vectors")
    print()

    query = "Alien invasion"
    for method in ["fixed_chunk", "sentence_chunk", "semantic_chunk"]:
        print(f"--- Query: '{query}' | method={method} ---")
        results = search(query, method, k=3)
        print(format_results(results))
        print()

    print("--- Grouped results (year >= 2000) ---")
    for method in ["fixed_chunk", "sentence_chunk", "semantic_chunk"]:
        print(f"method={method}")
        grouped = search(query, method, k=3, year_gte=2000, grouped=True)
        print(format_results(grouped))
        print()


if __name__ == "__main__":
    run_experiment(token_limit=256, overlap=40)
    run_experiment(token_limit=40, overlap=10)

    print("Experiment matrix (qualitative expectations):")
    print("Chunk Method | Token Size | Accuracy  | Speed")
    print("Fixed        | 256        | OK        | Fast")
    print("Sentence     | 256        | Better    | Medium")
    print("Semantic     | 256        | Best      | Slow")
    print("Fixed        | 40         | Poor      | Fast")
    print("Sentence     | 40         | OK        | Medium")
    print("Semantic     | 40         | Excellent | Slow")
