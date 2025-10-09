# schema-search

Natural language search over database schemas using graph-aware semantic retrieval.

## The Problem

You have a 200-table database. Someone asks "where are refunds stored?"

Standard RAG approaches fail because:
- They match isolated chunks without understanding relationships
- Missing a foreign key means missing half the context
- No way to traverse `payments` → `transactions` → `refunds` automatically

**schema-search** solves this by combining semantic search with graph traversal over foreign key relationships.

## Quick Start

```bash
# Basic install
pip install -e .

# With PostgreSQL support
pip install -e ".[postgres]"

# With MySQL support
pip install -e ".[mysql]"
```

```python
from sqlalchemy import create_engine
from schema_search import SchemaSearch

engine = create_engine("postgresql://user:pass@localhost/mydb")
search = SchemaSearch(engine)
search.index()

results = search.search("where are refunds stored?")
for result in results:
    print(result['table'], result['score'])
    print(result['schema'])
    print(result['related_tables'])
```

## How It Works

### Why Graph-Aware Search?

**Naive RAG**: Match "refunds" → return `refund_requests` table → miss that actual refund data is in `payment_transactions` linked via foreign key.

**schema-search**:
1. Semantic + BM25 hybrid search on chunked schema markdown
2. Extract top matching tables
3. **Graph expansion**: traverse foreign keys N-hops (bidirectional)
4. Re-rank expanded set with all related context
5. Return full schemas for top-K tables + their relationships

### Pipeline

```
Query: "where are refunds stored?"
  ↓
Chunk matching (embedding + BM25)
  ↓
Initial candidates: [refund_requests, payment_logs]
  ↓
Graph expansion (1-hop FK traversal)
  ↓
Expanded set: [refund_requests, payments, transactions, payment_logs, orders]
  ↓
Re-rank with full context
  ↓
Top 3: [transactions, payments, refund_requests]
```

### Configuration

```yaml
embedding:
  model: "all-MiniLM-L6-v2"
  cache_dir: ".schema_search_cache"

chunking:
  max_tokens: 512
  overlap_tokens: 50

search:
  embedding_weight: 0.6
  bm25_weight: 0.4
  initial_top_k: 20
  rerank_top_k: 5
  graph_expand_hops: 1
```

Custom config: `SchemaSearch(engine, config_path="config.yml")`

## Architecture

```
schema_search/
├── metadata_extractor.py   # SQLAlchemy schema extraction
├── chunker.py              # Markdown chunking with overlap
├── embedding_manager.py    # Embedding generation & caching
├── graph_builder.py        # FK relationship graph
├── ranker.py               # Hybrid scoring (BM25 + embeddings)
└── schema_search.py        # Main orchestrator
```

Dependencies: `sqlalchemy`, `sentence-transformers`, `networkx`, `rank-bm25`

## Cache

All artifacts cached in `.schema_search_cache/`:
- `metadata.json` - Extracted schema
- `embeddings.npz` - Chunk embeddings
- `graph.pkl` - FK graph

Delete to force re-index.

## Testing

Run integration tests:

```bash
# Install with test dependencies and database driver (use quotes for zsh)
pip install -e ".[test,postgres]"  # or [test,mysql]

# Setup test database connection
cp tests/env.template tests/.env
# Edit tests/.env with your DATABASE_URL

# Run tests
pytest tests/
```

See `tests/README.md` for more details.

## License

MIT
