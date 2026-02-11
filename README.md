# Semantic Movie Recommender (Qdrant)

An experimental retrieval framework for movie recommendations using Qdrant. It supports dense, sparse (BM25), and hybrid retrieval strategies with explainable results and evaluation tooling.

## Overview

This project explores how different retrieval strategies perform across query types:

- **Dense embeddings** for semantic similarity
- **Sparse BM25** for keyword precision and explainability
- **Hybrid pipelines** that combine dense + sparse signals

Each movie is stored as a single Qdrant point with:
- `description` used only for dense embeddings
- `name`, `director`, `cast`, and `themes` used for sparse retrieval
- metadata fields for filtering

## Project Structure

```
semantic-movie-recommender/
├── data/
│   └── movies.py               # ~100 movies dataset
├── db/
│   └── qdrant_client.py         # Qdrant client + collection schema
├── embeddings/
│   └── encoder.py               # Sentence-transformers embedding utilities
├── evaluation/
│   ├── ground_truth.py          # Hand-labeled queries for evaluation
│   ├── metrics.py               # Precision/Recall/MRR/NDCG
│   ├── report.py                # JSON/CSV output
│   └── runner.py                # Evaluation runner
├── scripts/
│   ├── run_dense_queries.py     # Manual dense-only queries
│   ├── run_sparse_queries.py    # Manual sparse-only queries
│   └── interactive_cli.py       # Interactive CLI
├── ingest.py                    # Ingestion pipeline
├── retrieval.py                 # Retrieval strategies
├── requirements.txt
└── README.md
```

## Key Concepts

### 1. Qdrant Schema
- **Dense vector**: `dense` (384 dims)
- **Sparse text**: `sparse_text` (BM25 index)
- **Payload fields**:
  - `name`, `description`, `year`, `director`, `cast`, `themes`
  - `*_norm` fields for keyword filters

### 2. Retrieval Strategies
Implemented in `retrieval.py`:

- `dense_only`
- `sparse_only`
- `sparse_prefilter_dense_rank`
- `dense_recall_sparse_rerank`
- `sparse_recall_dense_rerank`
- `hybrid_combined`

Each result includes:
- `dense_score`, `sparse_score`, `final_score`
- `match_explanation`

### 3. Sparse Text Policy
Sparse indexing uses **only**:

```
name + director + cast + themes
```

`description` is excluded from sparse text to preserve explainability.

## Installation

### Prerequisites
- Python 3.10+
- Qdrant instance (local or remote)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Qdrant (local)

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

Configure a remote instance via:

```bash
export QDRANT_URL="https://your-qdrant-instance"
export QDRANT_API_KEY="your_api_key"
```

## Ingestion

```bash
python3 ingest.py
```

Outputs a summary like:
```
Ingested 100 movies -> 100 points
```

## Retrieval (Manual)

Dense-only:
```bash
python3 scripts/run_dense_queries.py
```

Sparse-only:
```bash
python3 scripts/run_sparse_queries.py
```

Interactive CLI:
```bash
python3 scripts/interactive_cli.py
```

## Evaluation

Run all strategies against ground truth queries:

```bash
python3 evaluation/runner.py
```

Outputs:
- `results.json` with raw runs + aggregates
- `results.csv` (optional)

## Dataset

`data/movies.py` contains ~100 sci-fi and related films with:
- name
- description
- year
- director
- cast
- themes

## Troubleshooting

**Qdrant connection refused**

```
httpx.ConnectError: [Errno 61] Connection refused
```

Make sure Qdrant is running and reachable:
```bash
curl http://localhost:6333/health
```

## Notes

- Dense embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- Sparse BM25 is used for explainable keyword matching
- Hybrid pipelines support weighted fusion and reranking

## License

This project is provided as-is for educational and experimental use.
