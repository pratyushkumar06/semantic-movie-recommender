from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import retrieve  # noqa: E402


def main() -> None:
    queries = [
        ("movies about memory and identity", {"dense_top_k": 20, "rerank_depth": 10}, None),
        ("movies starring Sigourney Weaver", {"dense_top_k": 20, "rerank_depth": 10}, None),
        (
            "post-apocalyptic movies with class conflict released after 2000",
            {"dense_top_k": 20, "rerank_depth": 10},
            {"year_min": 2000},
        ),
    ]

    for query, params, filters in queries:
        print("===", query, "===")
        results = retrieve(
            query=query,
            strategy="dense_recall_sparse_rerank",
            params=params,
            filters=filters,
        )
        for idx, r in enumerate(results, 1):
            print(
                f\"{idx}. {r['name']} | final={r['final_score']:.4f} | dense={r['dense_score']:.4f} | sparse={r['sparse_score']:.4f}\"
            )
        print()


if __name__ == "__main__":
    main()
