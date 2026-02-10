# Qdrant Movie Recommendations

A comprehensive experimental framework for evaluating multiple text chunking strategies in semantic search systems. This project compares fixed-token, sentence-aware, and semantic chunking methods on movie recommendation tasks using Qdrant as the vector database backend.

## Overview

This project investigates the trade-offs between different text chunking strategies for embedding-based semantic search. The key question: **How do different ways of breaking documents into chunks affect search quality, speed, and coherence?**

### Chunking Strategies Evaluated

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Fixed Token** | Splits at fixed token counts with overlap | Fast, predictable size | May split sentences awkwardly |
| **Sentence** | Respects sentence boundaries | More coherent semantically | Variable chunk sizes |
| **Semantic** | Groups by embedding similarity | Most coherent chunks | Computationally expensive |

## Project Structure

```
Qdrant-Movie-Recs/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── experiments.py            # Experimental harness for benchmarking
├── ingest.py                 # Data ingestion pipeline
├── search.py                 # Search and query interface
│
├── chunking/
│   └── chunkers.py          # Implementation of three chunking strategies
│
├── data/
│   └── movies.py            # Movie dataset (16 sci-fi films with descriptions)
│
├── db/
│   └── qdrant_client.py     # Qdrant database client & configuration
│
└── embeddings/
    └── encoder.py           # Text encoding & token management utilities
```

## Key Components

### 1. Embeddings Module (`embeddings/encoder.py`)

Handles all text encoding operations using Hugging Face models:

- **`EncoderBundle`**: Container for model + tokenizer instances
- **`load_encoder()`**: Lazy-loads the sentence transformer model
- **`embed_text()` / `embed_texts()`**: Generates normalized embeddings
- **`count_tokens()`**: Analyzes token usage per text
- **`token_overflow()`**: Detects when text exceeds token limits
- **`inspect_text_tokens()`**: Generates debugging reports

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding size: 384 dimensions
- Default token limit: 256 tokens per chunk

### 2. Chunking Module (`chunking/chunkers.py`)

Implements three distinct text splitting algorithms:

#### Fixed Token Chunking
```python
chunkers.fixed_token_chunks(text)
```
- Splits by exact token count (e.g., 256 tokens)
- Configurable overlap (e.g., 40 tokens)
- **Use case**: When predictable chunk sizes are critical

#### Sentence-Based Chunking
```python
chunkers.sentence_chunks(text)
```
- Leverages LlamaIndex `SentenceSplitter`
- Respects sentence boundaries
- **Use case**: When coherence matters more than uniformity

#### Semantic Chunking
```python
chunkers.semantic_chunks(text)
```
- Uses embedding-based semantic similarity (LlamaIndex `SemanticSplitterNodeParser`)
- Groups semantically related content
- **Use case**: When maximum chunk coherence is the goal

### 3. Database Module (`db/qdrant_client.py`)

Manages Qdrant vector database operations:

- **`get_client()`**: Singleton client with lazy initialization
- **`recreate_collection()`**: Creates multi-vector collections
  - Each collection has 3 vector fields (one per chunking strategy)
  - Uses cosine distance metric
- **Configuration**: Via environment variables
  - `QDRANT_URL`: Database endpoint (default: `http://localhost:6333`)
  - `QDRANT_API_KEY`: Optional authentication

### 4. Ingestion Pipeline (`ingest.py`)

End-to-end data loading:

1. Loads movie descriptions from `data/movies.py`
2. Initializes encoder and chunkers
3. Applies all three chunking strategies to each movie
4. Generates embeddings for every chunk
5. Uploads to Qdrant with full metadata

```python
total_vectors, total_movies = ingest_movies(
    token_limit=256,
    overlap=40,
    show_token_report=True
)
```

**Output**: Each movie chunk becomes a separate Qdrant point with:
- Exactly one named vector (the chunking strategy used)
- Payload metadata (movie name, year, author, chunk text, method)

### 5. Search Interface (`search.py`)

Query and retrieval functions:

```python
# Search with filtering
results = search(
    query="Alien invasion",
    chunk_method="semantic_chunk",
    k=3,
    year_gte=2000,
    grouped=False
)

# Format for display
print(format_results(results))
```

**Features**:
- Multiple chunking strategy support
- Year-based filtering via metadata
- Grouped mode: One best chunk per movie
- Ungrouped mode: Top k chunks (may be from same movie)

### 6. Experiment Runner (`experiments.py`)

Comparative benchmarking framework:

```python
run_experiment(token_limit=256, overlap=40)
run_experiment(token_limit=40, overlap=10)
```

Tests combinations of:
- Token limits (256 vs 40)
- Overlap amounts
- All chunking strategies
- Both grouped/ungrouped search modes

## Installation & Setup

### Prerequisites
- Python 3.10+
- Qdrant instance (Docker or cloud)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `sentence-transformers`: NLP embeddings
- `transformers`: Tokenizer & model utilities
- `qdrant-client`: Vector DB interface
- `llama-index`: Text chunking & splitting
- `llama-index-embeddings-huggingface`: Embedding integration

### 2. Configure Qdrant

**Option A: Local Docker**
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

**Option B: Environment Variables** (for remote instance)
```bash
export QDRANT_URL="https://your-qdrant-instance.com"
export QDRANT_API_KEY="your_api_key"
```

## Usage Examples

### Example 1: Ingest Data
```python
from ingest import ingest_movies

# Load movies with token analysis
total_points, total_movies = ingest_movies(
    token_limit=256,
    overlap=40,
    show_token_report=True
)
print(f"Ingested {total_movies} movies → {total_points} vectors")
```

### Example 2: Semantic Search
```python
from search import search, format_results

# Find movies about alien invasions
results = search(
    query="Alien invasion",
    chunk_method="semantic_chunk",
    k=3
)
print(format_results(results))
```

### Example 3: Filtered Search
```python
# Find recent movies (2000+) matching query
results = search(
    query="Time travel and alternate dimensions",
    chunk_method="sentence_chunk",
    k=5,
    year_gte=2000,
    grouped=True  # One best result per movie
)
```

### Example 4: Run Full Benchmark
```python
from experiments import run_experiment

# Compare chunking strategies with different parameters
run_experiment(token_limit=256, overlap=40)
run_experiment(token_limit=40, overlap=10)
```

## Experimental Results Matrix

The `experiments.py` module generates qualitative expectations:

| Chunk Method | Token Limit | Accuracy | Speed | Notes |
|--------------|------------|----------|-------|-------|
| Fixed | 256 | OK | Fast | Predictable but may split mid-thought |
| Sentence | 256 | Better | Medium | Respects language structure |
| Semantic | 256 | Best | Slow | Highest coherence, expensive |
| Fixed | 40 | Poor | Fast | Too fine-grained |
| Sentence | 40 | OK | Medium | May exceed limit on long sentences |
| Semantic | 40 | Excellent | Slow | Ideal for short, focused chunks |

## Dataset

The project includes 16 curated sci-fi films in `data/movies.py`:

1. **Annihilation** (2018) - Alex Garland
2. **Arrival** (2016) - Denis Villeneuve
3. **Blade Runner 2049** (2017) - Denis Villeneuve
4. **Ex Machina** (2014) - Alex Garland
5. **Dune** (2021) - Denis Villeneuve
6. **The Matrix** (1999) - The Wachowskis
7. **Interstellar** (2014) - Christopher Nolan
8. **The Thing** (1982) - John Carpenter
9. **Alien** (1979) - Ridley Scott
10. **Contact** (1997) - Robert Zemeckis
11. **Moon** (2009) - Duncan Jones
12. **Solaris** (1972) - Andrei Tarkovsky
13. **Predator** (1987) - John McTiernan
14. **Gattaca** (1997) - Andrew Niccol
15. **Snowpiercer** (2013) - Bong Joon-ho
16. **Edge of Tomorrow** (2014) - Doug Liman

Each movie includes:
- Detailed plot/theme description (typically under 256 tokens at the current length)
- Release year
- Director name

## Key Insights

### When to Use Each Strategy

**Fixed Token Chunking**:
- Simple, predictable infrastructure
- Good for systems with strict memory constraints
- Fast processing, no extra embedding overhead

**Sentence Chunking**:
- Balanced trade-off between speed and coherence
- Natural language boundaries improve relevance
- Recommended for most production systems

**Semantic Chunking**:
- Maximum search quality when budget allows
- Identify natural topic boundaries
- Best for high-accuracy, low-latency scenarios
- Requires extra compute during ingestion

### Performance Considerations

| Metric | Fixed | Sentence | Semantic |
|--------|-------|----------|----------|
| Ingestion Speed | ✓✓✓ | ✓✓ | ✓ |
| Query Latency | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| Result Quality | ✓ | ✓✓ | ✓✓✓ |
| Memory Overhead | ✓✓ | ✓✓ | ✓ |

## Code Documentation

The codebase uses type hints and clear function names for readability. If you want full docstrings and autogenerated docs, we can add them as a follow-up.

## Troubleshooting

### Issue: Connection to Qdrant Failed
```
QdrantException: Connection refused
```
**Solution**: Ensure Qdrant is running and accessible
```bash
# Check local instance
curl http://localhost:6333/health

# Or set remote URL
export QDRANT_URL="http://your-server:6333"
```

### Issue: Out of Memory During Semantic Chunking
**Solution**: Reduce token limit or batch size
```python
chunkers = Chunkers(token_limit=128, overlap=20, ...)
```

### Issue: Token Count Exceeds Limit
**Solution**: Check `show_token_report=True` in ingestion
```python
ingest_movies(token_limit=256, show_token_report=True)
```

## Architecture Decisions

1. **Modular Design**: Each responsibility isolated (encoding, chunking, DB, search)
2. **Multi-Strategy Support**: Store vectors from all methods for comparison
3. **Metadata Preservation**: Full movie info available with chunks
4. **Configuration Flexibility**: Environment variables + function parameters
5. **Type Hints**: Full Python type annotations for clarity

## Future Enhancements

- [ ] Support for additional embedding models
- [ ] Batch ingestion for large datasets
- [ ] Query result caching
- [ ] Performance profiling dashboard
- [ ] Additional movie metadata (genre, rating, etc.)
- [ ] Hybrid search combining vector + keyword matching
- [ ] A/B testing framework for strategy comparison

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [LlamaIndex Chunking](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## License

This project is provided as-is for educational and experimental purposes.

## Author

Created as a comprehensive experiment in semantic search architecture and text chunking strategies.
