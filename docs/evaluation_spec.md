# Evaluation Framework Spec (Hybrid Movie Retrieval)

This document defines how we evaluate retrieval strategies **before** tuning or implementation.
It is the source of truth for metrics, queries, and experiment logging.

## Goals
- Compare dense, sparse, and hybrid strategies consistently
- Measure accuracy, explainability, and performance trade‑offs
- Enable reproducible experiments with clear ground truth

---

## 1. Use Case Categories
Each category represents a distinct retrieval intent.

1. **Semantic Discovery**
   - Conceptual similarity without exact keywords.
2. **Keyword Precision**
   - Exact matching on cast/themes/title.
3. **Director/Cast Queries**
   - Explicit metadata filters and explainability.
4. **Multi‑Constraint Queries**
   - Combination of semantic + keyword + metadata filters.
5. **Hybrid Edge Cases**
   - Queries where dense and sparse disagree or conflict.

---

## 2. Query Set (Initial)
Target: **5–10 queries per category**. Start small and expand.

### Semantic Discovery
- “existential sci‑fi about memory”
- “philosophical space station drama”
- “movies about artificial consciousness”
- “films exploring identity and reality”
- “slow‑burn sci‑fi about isolation”

### Keyword Precision
- “movies starring Sigourney Weaver”
- “films about time travel paradox”
- “movies with themes of surveillance”
- “AI rebellion movies”
- “cyberpunk films”

### Director/Cast Queries
- “Denis Villeneuve sci‑fi”
- “Ridley Scott space horror”
- “films starring Arnold Schwarzenegger”
- “Christopher Nolan science fiction”
- “movies with Tom Cruise in a time loop”

### Multi‑Constraint Queries
- “post‑apocalyptic movies with class conflict released after 2000”
- “space survival films after 2010”
- “AI movies directed by James Cameron”
- “time travel thrillers before 2000”
- “alien invasion movies with family themes after 2000”

### Hybrid Edge Cases
- “intelligent AI escaping confinement”
- “dream‑based sci‑fi heist”
- “genetic discrimination in near‑future society”
- “society collapse due to infertility”
- “philosophical sci‑fi about consciousness”

---

## 3. Ground Truth Strategy
We start with **manual gold labels** for high reliability.

Each query must include:
- `relevant_movies`: list of movie titles
- Optional: `relevance_tiers` (e.g., primary vs secondary matches)

Example:
```json
{
  "use_case": "semantic_discovery",
  "query": "movies about memory and identity",
  "relevant_movies": ["Blade Runner 2049", "Moon", "Solaris (1972)"]
}
```

If needed later, add **weak supervision** as a secondary signal:
- Theme overlap heuristic
- Cast overlap heuristic
- Director match heuristic

---

## 4. Metrics
Compute for each query and strategy:
- **Precision@K**
- **Recall@K**
- **MRR**
- **Latency** (split into embedding, retrieval, rerank)
- **Explainability notes** (human‑readable short note)
- **Failure reason** (if wrong)

---

## 5. Experiment Logging Format
Each run should emit a structured record:
```json
{
  "use_case": "semantic_discovery",
  "query": "movies about memory and identity",
  "strategy": "dense_only",
  "params": {
    "dense_top_k": 100
  },
  "results": [
    {
      "name": "Blade Runner 2049",
      "dense_score": 0.82,
      "sparse_score": null,
      "final_score": 0.82,
      "match_explanation": "Semantic match to description."
    }
  ],
  "metrics": {
    "precision_at_5": 0.6,
    "recall_at_10": 0.8,
    "mrr": 1.0,
    "latency_ms": {
      "embedding": 12,
      "retrieval": 35,
      "rerank": 4
    }
  },
  "qualitative_notes": "Good semantic recall, weak keyword precision",
  "failure_reason": "semantic drift"
}
```

---

## 6. Notes / Assumptions
- Ground truth is **manual first**, expanded later.
- Query intent is selected by experiment (no auto‑classification).
- All text is English.
- Missing scores are always `null`, not omitted.
