# Qdrant Schema (Hybrid Movie Retrieval)

This schema is the source of truth for the hybrid retrieval experiments.
It **does not** include the legacy `author` field — use `director` only.

## Collection
- Name: `my_movies` (configurable via `COLLECTION_NAME`)
- Granularity: **one movie per point**

## Vectors
- `dense` (float vector)
  - Source: embedding of full `description` (optionally include `name`)
  - Distance: cosine (or dot if embeddings are normalized)
  - Purpose: semantic similarity over plot/description

## Sparse Signal (BM25)
**Default**: Qdrant text index over a single field.
- `sparse_text` (string)
  - Concatenation of: `name`, `director`, `cast`, `themes`
  - Used by BM25 for sparse retrieval and reranking
  - **IMPORTANT**: `description` is excluded by policy
  - Purpose: keyword precision and explainability

If we later switch to precomputed sparse vectors, add:
- `sparse` (sparse vector)
  - Source: BM25 vectorization outside Qdrant
  - Purpose: keyword scoring without Qdrant text index

## Payload Fields (What Each Field Does)
| Field | Type | Purpose |
|---|---|---|
| `name` | string | Display title and keyword matching |
| `description` | string | Long semantic text for dense embeddings |
| `year` | int | Hard filter for time‑based queries |
| `director` | string | Director name for filtering and display (replaces `author`) |
| `cast` | array of strings | Actor list for keyword filtering and explainability |
| `themes` | array of strings | Curated keywords for thematic matching |
| `name_norm` | string | Lowercased `name` for case‑insensitive filters |
| `director_norm` | string | Lowercased `director` for case‑insensitive filters |
| `cast_norm` | array of strings | Lowercased `cast` for case‑insensitive filters |
| `themes_norm` | array of strings | Lowercased `themes` for case‑insensitive filters |
| `sparse_text` | string | Concatenated text used by BM25 (name + director + cast + themes only) |

## Indexes (Why We Index)
| Index | Type | Purpose |
|---|---|---|
| `year` | range | Fast numeric filtering (e.g., year >= 2000) |
| `director_norm` | keyword | Exact match on director |
| `cast_norm` | keyword array | Match on any cast member |
| `themes_norm` | keyword array | Match on any theme keyword |
| `name_norm` | keyword | Exact match on title |
| `sparse_text` | text | BM25 scoring over keyword text |

## Notes
- The legacy `author` field is **removed** and should not be used in any
  payload or filter logic.
- `match_explanation` is computed at query time and is **not** stored in payload.
- If you need chunking experiments, keep a **separate collection** to avoid
  mixing movie‑level retrieval with chunk‑level vectors.
