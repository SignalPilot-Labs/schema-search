# Cache filenames
CACHE_FILE_METADATA = "metadata.json"
CACHE_FILE_CHUNK_METADATA = "chunk_metadata.json"
CACHE_FILE_EMBEDDINGS = "embeddings.npz"
CACHE_FILE_CACHE_CONFIG = "cache_config.json"
CACHE_FILE_GRAPH = "graph.pkl"

# Numeric constants
EPSILON = 1e-8
TOKEN_ESTIMATION_DIVISOR = 4
LLM_MAX_TOKENS = 500
DEBUG_TRUNCATION_LENGTH = 100
FUZZY_SCORE_CUTOFF = 0
FUZZY_NORMALIZATION = 100.0
BM25_MIN_SUFFIX_LENGTH = 2

# Search/strategy strings
STRATEGY_SEMANTIC = "semantic"
STRATEGY_BM25 = "bm25"
STRATEGY_HYBRID = "hybrid"
STRATEGY_FUZZY = "fuzzy"

# Cache type string
CACHE_TYPE_MEMORY = "memory"

# DB dialect strings
DIALECT_DATABRICKS = "databricks"
DIALECT_SNOWFLAKE = "snowflake"

# Chunker strategy strings
CHUNKER_RAW = "raw"
CHUNKER_LLM = "llm"

# Output format strings
OUTPUT_FORMAT_JSON = "json"
OUTPUT_FORMAT_MARKDOWN = "markdown"

# Optional dependency module names
MODULE_SENTENCE_TRANSFORMERS = "sentence_transformers"
MODULE_OPENAI = "openai"
MODULE_BM25S = "bm25s"

# App identity
APP_NAME = "schema-search"

# Database magic strings
DB_NAME_DEFAULT = "default"
NULLABLE_YES = "YES"
SKIP_CATALOG_SYSTEM = "system"

# BM25 token normalization synonym maps
BM25_PK_SYNONYMS: set[str] = {"pk", "pkey", "key"}
BM25_TIMESTAMP_SYNONYMS: set[str] = {"ts", "time", "timestamp"}
BM25_INDEX_SYNONYMS: set[str] = {"ix", "index", "idx"}
BM25_PK_NORMAL = "id"
BM25_TIMESTAMP_NORMAL = "timestamp"
BM25_INDEX_NORMAL = "index"
