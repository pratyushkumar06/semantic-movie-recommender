"""Evaluation runner for retrieval strategies."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.ground_truth import GroundTruthQuery, get_ground_truth
from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from evaluation.report import build_report, write_csv, write_json
from retrieval import retrieve


def run_evaluation(
    strategies: List[str],
    params_by_strategy: Dict[str, Dict[str, Any]],
    output_json: str = "results.json",
    output_csv: str | None = None,
) -> Dict[str, Any]:
    """Run evaluation across all queries and strategies."""
    runs: List[Dict[str, Any]] = []
    ground_truth = get_ground_truth()

    for gt in ground_truth:
        for strategy in strategies:
            params = params_by_strategy.get(strategy, {})

            start = time.perf_counter()
            results = retrieve(query=gt.query, strategy=strategy, params=params)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            ranked_ids = [str(r["id"]) for r in results]
            metrics = {
                "precision_at_5": precision_at_k(gt.relevant_ids, ranked_ids, 5),
                "recall_at_5": recall_at_k(gt.relevant_ids, ranked_ids, 5),
                "precision_at_10": precision_at_k(gt.relevant_ids, ranked_ids, 10),
                "recall_at_10": recall_at_k(gt.relevant_ids, ranked_ids, 10),
                "mrr": mrr(gt.relevant_ids, ranked_ids),
                "ndcg_at_10": ndcg_at_k(gt.relevant_ids, ranked_ids, 10),
                "latency_ms": round(elapsed_ms, 3),
            }

            runs.append(
                {
                    "use_case": gt.use_case,
                    "query": gt.query,
                    "strategy": strategy,
                    "params": params,
                    "results": results,
                    "metrics": metrics,
                }
            )

    report = build_report(runs)
    write_json(report, output_json)
    if output_csv:
        write_csv(runs, output_csv)
    return report


if __name__ == "__main__":
    STRATEGIES = [
        "dense_only",
        "sparse_only",
        "dense_recall_sparse_rerank",
        "sparse_prefilter_dense_rank",
        "sparse_recall_dense_rerank",
        "hybrid_combined",
    ]

    PARAMS = {
        "dense_only": {"dense_top_k": 20},
        "sparse_only": {"sparse_top_k": 20},
        "dense_recall_sparse_rerank": {"dense_top_k": 30, "rerank_depth": 15},
        "sparse_prefilter_dense_rank": {"sparse_top_k": 50, "dense_top_k": 20},
        "sparse_recall_dense_rerank": {
            "sparse_top_k": 50,
            "dense_top_k": 20,
            "rerank_depth": 30,
        },
        "hybrid_combined": {
            "dense_top_k": 40,
            "sparse_top_k": 40,
            "fusion_alpha": 0.5,
            "fusion_beta": 0.5,
        },
    }

    run_evaluation(
        strategies=STRATEGIES,
        params_by_strategy=PARAMS,
        output_json="results.json",
        output_csv="results.csv",
    )
