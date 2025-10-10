# schema-search

Ask questions about your database in natural language. Get back the exact tables you need, with all their relationships mapped out.

## Why

Suppose you have 200 tables in a database. Someone asks "where are user refunds stored?" 

You could:
- Grep through SQL files for 20 minutes
- Ask an LLM, which will struggle to sift through 200 table schemas

Or just ask the database directly.

## Install

```bash
pip install -e .
```

## Use

```python
from sqlalchemy import create_engine
from schema_search import SchemaSearch

engine = create_engine("postgresql://user:pass@localhost/db")
search = SchemaSearch(engine)

search.index()
results = search.search("where are user refunds stored?")

for result in results['results']:
    print(result['table'])           # "refund_transactions"
    print(result['schema'])           # Full column info, types, constraints
    print(result['related_tables'])   # ["users", "payments", "transactions"]
```

## Configuration

Edit `config.yml`:

```yaml
embedding:
  location: "memory"  # vectordb coming soon
  model: "multi-qa-MiniLM-L6-cos-v1"
  metric: "cosine"

chunking:
  strategy: "raw"  # or "llm"

reranker:
  strategy: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### LLM Chunking

Use LLM to generate semantic summaries instead of raw schema text:

1. Set `strategy: "llm"` in `config.yml`
2. Pass API credentials:

```python
search = SchemaSearch(
    engine,
    llm_api_key="sk-ant-...",
    llm_base_url="https://api.anthropic.com"  # optional
)
```

## How It Works

1. Semantic search on schema chunks (in-memory embeddings)
2. Expand results via foreign key graph (N-hops)
3. Re-rank with CrossEncoder
4. Return top tables with relationships

Cache stored in `.schema_search_cache/`.

## License

MIT
