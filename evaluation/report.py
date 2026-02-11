"""Reporting utilities for evaluation runs."""
from __future__ import annotations

import csv
import json
from typing import Any, Dict, List


def build_report(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-query runs into a report with strategy averages."""
    aggregates: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for run in runs:
        strategy = run["strategy"]
        metrics = run["metrics"]
        aggregates.setdefault(strategy, {})
        counts[strategy] = counts.get(strategy, 0) + 1
        for key, value in metrics.items():
            if isinstance(value, dict):
                continue
            aggregates[strategy][key] = aggregates[strategy].get(key, 0.0) + float(value)

    for strategy, totals in aggregates.items():
        count = counts.get(strategy, 1)
        for key in list(totals.keys()):
            totals[key] = totals[key] / count

    return {
        "runs": runs,
        "aggregates": aggregates,
    }


def write_json(report: Dict[str, Any], path: str) -> None:
    """Write report to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_csv(runs: List[Dict[str, Any]], path: str) -> None:
    """Write flat per-query results to CSV."""
    if not runs:
        return
    fieldnames = [
        "use_case",
        "query",
        "strategy",
        "params",
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "latency_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {
                "use_case": run["use_case"],
                "query": run["query"],
                "strategy": run["strategy"],
                "params": json.dumps(run["params"], sort_keys=True),
                "precision_at_5": run["metrics"].get("precision_at_5"),
                "recall_at_5": run["metrics"].get("recall_at_5"),
                "precision_at_10": run["metrics"].get("precision_at_10"),
                "recall_at_10": run["metrics"].get("recall_at_10"),
                "mrr": run["metrics"].get("mrr"),
                "ndcg_at_10": run["metrics"].get("ndcg_at_10"),
                "latency_ms": run["metrics"].get("latency_ms"),
            }
            writer.writerow(row)
