from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import retrieve  # noqa: E402


def main() -> None:
    queries = [
        ("movies about memory and identity", {"sparse_top_k": 5}, None),
        ("movies starring Sigourney Weaver", {"sparse_top_k": 5}, None),
        (
            "post-apocalyptic movies with class conflict released after 2000",
            {"sparse_top_k": 5},
            {"year_min": 2000},
        ),
    ]

    for query, params, filters in queries:
        print("===", query, "===")
        results = retrieve(query=query, strategy="sparse_only", params=params, filters=filters)
        for idx, r in enumerate(results, 1):
            print(
                f"{idx}. {r['name']} | score={r['final_score']:.4f} | year={r.get('year')} | director={r.get('director')}"
            )
        print()


if __name__ == "__main__":
    main()
