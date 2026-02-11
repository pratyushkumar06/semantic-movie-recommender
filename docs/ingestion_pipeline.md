# Ingestion Pipeline Spec (Movie‑Level Points)

This document defines how raw movie data becomes Qdrant points.
No code here — only deterministic rules.

## Goals
- One movie == one Qdrant point
- Deterministic, repeatable ingestion
- Supports dense + sparse hybrid retrieval
- Keeps sparse retrieval explainable and precise

## Inputs
Source: `data/movies.py`  
Each movie dict contains:
- `name` (string)
- `description` (string)
- `year` (int)
- `director` (string)
- `cast` (list of strings)
- `themes` (list of strings)

## Outputs (Per Movie Point)
- `id` (UUID string, deterministic)
- `vector`: `dense`
- `payload`: fields defined in `docs/qdrant_schema.md`

## Preprocessing & Normalization
For each movie:
1. **Trim** whitespace on all string fields.
2. **Ensure lists** (`cast`, `themes`) are lists of strings.
3. **Deduplicate** `cast` and `themes` while preserving order.
4. **Normalize** for filters:
   - `name_norm` = `name` lowercased
   - `director_norm` = `director` lowercased
   - `cast_norm` = each cast member lowercased
   - `themes_norm` = each theme lowercased

No stemming, lemmatization, or language detection (English only).

## Sparse Text Construction (BM25)
`sparse_text` is built **only** from:
- `name`
- `director`
- `cast`
- `themes`

Rule:
```
sparse_text = name + " " + director + " " + " ".join(cast) + " " + " ".join(themes)
```

`description` is **never** included in `sparse_text`.
Use original casing for `sparse_text` (normalized fields are for filtering only).

## Dense Embedding
`dense_text` is **exactly**:
- `description`

Embed `dense_text` using the chosen English embedding model and store as the
`dense` vector.

## ID Strategy (Deterministic)
Use a stable UUID derived from movie identity:
- `id_source = f"{name}|{year}"`
- `id = uuid5(NAMESPACE_DNS, id_source)`

Rationale:
- Idempotent re‑ingestion
- Avoids collisions across similarly named movies from different years

## Upsert / Re‑ingestion Behavior
Default behavior:
1. Recreate collection for clean experiments
2. Ingest all movies from scratch

Alternative behavior (allowed):
- Upsert by deterministic `id`
- If `id` exists, payload and vectors are overwritten

## Validation & Safety
- If a required field is missing, skip the movie and log an error.
- If duplicate `id` appears in a single run, keep the first and log a warning.
- Do not insert points with empty `name` or `description`.

## Logging (For Observability)
At minimum log:
- total_movies_seen
- total_points_upserted
- movies_skipped (with reason)
- embedding model name + version/hash (if available) + vector dimension
