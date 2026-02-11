#!/usr/bin/env python3
# Usage: python3 scripts/interactive_cli.py
"""Interactive CLI for testing retrieval strategies."""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retrieval import retrieve

STRATEGY_ORDER = [
    "dense_only",
    "sparse_only",
    "sparse_prefilter_dense_rank",
    "dense_recall_sparse_rerank",
    "hybrid_combined",
    "hybrid_rrf",
]

DEFAULT_PARAMS: Dict[str, Dict[str, float | int | str]] = {
    "dense_only": {"dense_top_k": 20},
    "sparse_only": {"sparse_top_k": 20},
    "sparse_prefilter_dense_rank": {"sparse_top_k": 50, "dense_top_k": 20},
    "dense_recall_sparse_rerank": {"dense_top_k": 30, "rerank_depth": 15},
    "hybrid_combined": {"dense_top_k": 40, "sparse_top_k": 40},
    "hybrid_rrf": {"dense_top_k": 40, "sparse_top_k": 40},
}


def main() -> None:
    """Run the interactive CLI loop."""
    print("Semantic Movie Retrieval CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = _prompt_query()
        if query is None:
            print("Goodbye.")
            break

        strategy = _prompt_strategy()
        if strategy is None:
            continue

        compare_all = _prompt_compare_all()
        strategies = STRATEGY_ORDER if compare_all else [strategy]

        for strategy_name in strategies:
            _run_query(query, strategy_name)

        print("\n---\n")


def _prompt_query() -> Optional[str]:
    """Prompt user for a query string."""
    query = input("Enter your query: ").strip()
    if not query:
        print("Please enter a non-empty query.")
        return None
    if query.lower() in {"exit", "quit"}:
        return None
    return query


def _prompt_strategy() -> Optional[str]:
    """Prompt user to choose a retrieval strategy."""
    print("\nChoose a retrieval strategy:")
    for idx, name in enumerate(STRATEGY_ORDER, 1):
        print(f"{idx}. {name}")

    selection = input("Enter a number: ").strip()
    if not selection.isdigit():
        print("Invalid selection. Please enter a number from the list.")
        return None

    index = int(selection)
    if index < 1 or index > len(STRATEGY_ORDER):
        print("Invalid selection. Please choose a valid strategy number.")
        return None
    return STRATEGY_ORDER[index - 1]


def _prompt_compare_all() -> bool:
    """Ask whether to run all strategies."""
    choice = input("Compare all strategies? (y/N): ").strip().lower()
    return choice in {"y", "yes"}


def _run_query(query: str, strategy: str, top_k: int = 5) -> None:
    """Run a query with the selected strategy and print results."""
    params = DEFAULT_PARAMS.get(strategy, {})

    start = time.perf_counter()
    try:
        results = retrieve(query=query, strategy=strategy, params=params)
    except Exception as exc:
        print(f"\n[{strategy}] Retrieval error: {exc}")
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"\nStrategy: {strategy}")
    print(f"Latency: {elapsed_ms:.2f} ms")
    if not results:
        print("No results returned.")
        return

    print("\nTop results:")
    for rank, result in enumerate(results[:top_k], 1):
        title = result.get("name") or "Unknown"
        dense_score = _format_score(result.get("dense_score"))
        sparse_score = _format_score(result.get("sparse_score"))
        final_score = _format_score(result.get("final_score"))
        explanation = result.get("match_explanation") or "n/a"
        explanation = _truncate(explanation, 120)
        print(
            f"{rank}. {title} | final={final_score} | dense={dense_score} | "
            f"sparse={sparse_score}\n   {explanation}"
        )


def _format_score(value: Optional[float]) -> str:
    """Format score values for display."""
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _truncate(text: str, limit: int) -> str:
    """Truncate long strings for display."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    main()
