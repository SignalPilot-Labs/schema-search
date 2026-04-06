# LLM chunker token limits
LLM_SUMMARY_MAX_PROMPT_TOKENS: int = 250
LLM_SUMMARY_MAX_RESPONSE_TOKENS: int = 500

# Cache filenames
SCHEMA_CACHE_FILENAME: str = "metadata.json"
CHUNK_CACHE_FILENAME: str = "chunk_metadata.json"
EMBEDDINGS_CACHE_FILENAME: str = "embeddings.npz"
EMBEDDING_CONFIG_CACHE_FILENAME: str = "cache_config.json"
GRAPH_CACHE_FILENAME: str = "graph.pkl"

# Databricks user agent
DATABRICKS_USER_AGENT: str = "schema-search"

# MCP server name
MCP_SERVER_NAME: str = "schema-search"

# Default config path (relative to package root, resolved at runtime)
DEFAULT_CONFIG_FILENAME: str = "config.yml"

# Schema skip lists
SKIP_SCHEMAS: frozenset[str] = frozenset({
    "information_schema",
    "pg_catalog",
    "pg_toast",
    "pg_temp_1",
    "pg_toast_temp_1",
    "mysql",
    "performance_schema",
    "sys",
    "timescaledb_information",
    "timescaledb_experimental",
    "_timescaledb_catalog",
    "_timescaledb_config",
    "_timescaledb_cache",
    "_timescaledb_internal",
    "snowflake",
    "default",
})

SKIP_CATALOGS: frozenset[str] = frozenset({
    "system",
})
