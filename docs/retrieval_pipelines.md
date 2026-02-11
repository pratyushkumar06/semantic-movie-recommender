# Retrieval Pipelines Spec (Hybrid Movie Retrieval)

This document defines retrieval contracts and explanation behavior.
No implementation details are included.

## Function Contract
`retrieve(query, strategy, params, filters) -> ranked_results`

## Retrieval Result Schema
Each item in `ranked_results` must include:
- `id` (string or int)
- `name` (string)
- `dense_score` (float or null)
- `sparse_score` (float or null)
- `final_score` (float)
- `match_explanation` (string or structured object)

Rules:
- `dense_score` and `sparse_score` are always present but may be `null`.
- `final_score` is the score used for ranking.
- `match_explanation` is computed at query time and always present.

## Final Score Meaning (Explicit)
- Dense‑only: `final_score` == `dense_score`
- Sparse‑only: `final_score` == `sparse_score`
- Hybrid pipelines: `final_score` is explicitly computed (fusion or rerank)

## Explanation Logic Per Pipeline

**Pipeline 1: Dense‑Only**
- Signals: dense only
- Explanation: “Semantic match to description.”

**Pipeline 2: Sparse‑Only (BM25)**
- Signals: sparse only
- Explanation includes matched fields and terms when available.
- Example: “Matched themes: ['identity', 'memory']; BM25 keyword hit.”

**Pipeline 3: Sparse Prefilter → Dense Rank**
- Signals: sparse gate, dense rank
- Explanation: “Passed sparse filter on cast/themes; ranked by semantic similarity.”

**Pipeline 4: Dense Recall → Sparse Rerank**
- Signals: dense recall, sparse rerank
- Explanation: “Semantic match after dense recall; boosted by sparse hit on 'AI'.”

**Pipeline 5: Hybrid Combined Scoring (Parallel)**
- Signals: dense + sparse
- Explanation: “Combined semantic similarity + keyword match on themes ['isolation', 'paranoia'].”

## Match Explanation Structure (Recommended)
Internal structured form (stringify for display if needed):
```
{
  "dense": "high semantic similarity",
  "sparse": ["matched theme: identity", "matched cast: Sigourney Weaver"],
  "filters": ["year >= 2000"]
}
```

## Sparse Text Construction Rules
`sparse_text` must be built from:
- `name`
- `director`
- `cast`
- `themes`

`description` must **not** be included in `sparse_text`.

## Notes / Assumptions
- English only. Tokenization, BM25, and embeddings assume English.
- Dense score is cosine (or dot) similarity.
- Sparse score is BM25 score.
- When combining dense and sparse scores, normalize per query before fusion.
- If candidate count after filters is less than `rerank_depth`, reduce `rerank_depth` safely.
