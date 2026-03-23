"""Constants for Spider 2.0-Snow evaluation."""

from pathlib import Path

from anthropic.types import ToolParam

# Paths
_EVAL_DIR = Path(__file__).parent
SPIDER2_SNOW_DIR = _EVAL_DIR / "Spider2" / "spider2-snow"
JSONL_PATH = SPIDER2_SNOW_DIR / "spider2-snow.jsonl"
DOCUMENTS_DIR = SPIDER2_SNOW_DIR / "resource" / "documents"
CREDENTIAL_PATH = _EVAL_DIR / "snowflake_credential.json"
PROMPT_PATH = _EVAL_DIR / "prompt.md"
ENV_PATH = _EVAL_DIR.parent / ".env"
DEFAULT_OUTPUT_DIR = _EVAL_DIR / "results"

# Spider2 Snowflake shared account
SNOWFLAKE_ACCOUNT = "RSRSBDK-YDB67606"
DEFAULT_WAREHOUSE = "COMPUTE_WH_PARTICIPANT"
DEFAULT_ROLE = "PARTICIPANT"

# Model
MODEL_NAME = "claude-opus-4-6"
MAX_TOOL_TURNS = 20
MAX_TOKENS = 8192
TEMPERATURE = 0
SEARCH_LIMIT = 10
SEARCH_HOPS = 1

# run_sql limits
RUN_SQL_TIMEOUT = 30
RUN_SQL_MAX_ROWS = 50

DEFAULT_WORKERS = 4

SQL_CODE_BLOCK_PATTERN = r"```sql\s*(.*?)\s*```"

TOOL_SCHEMA_SEARCH: ToolParam = {
    "name": "schema_search",
    "description": (
        "Search database schema using natural language. "
        "Finds relevant database tables and their relationships by searching "
        "through schema metadata using semantic similarity. "
        "Expands results by traversing foreign key relationships."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question about database schema (e.g., 'tables related to payments')",
            },
            "schemas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of schema names to filter results (e.g., ['public', 'sales'])",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of table schemas to return in results",
            },
        },
        "required": ["query"],
    },
}

TOOL_GET_SCHEMA: ToolParam = {
    "name": "get_schema",
    "description": (
        "Get the full database schema structure. "
        "Returns the complete schema metadata including all tables, columns, "
        "foreign keys, and indices."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "schemas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of schema names to filter (e.g., ['public', 'sales'])",
            },
        },
        "required": [],
    },
}

TOOL_RUN_SQL: ToolParam = {
    "name": "run_sql",
    "description": (
        "Execute a SQL query against the Snowflake database and return results. "
        "Use this to verify your query works before submitting the final answer. "
        "Returns up to 50 rows. If the query errors, returns the error message."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL query to execute",
            },
        },
        "required": ["sql"],
    },
}

# Vanilla: get_schema + run_sql. MCP: get_schema + schema_search + run_sql.
TOOLS_VANILLA = [TOOL_GET_SCHEMA, TOOL_RUN_SQL]
TOOLS_MCP = [TOOL_GET_SCHEMA, TOOL_SCHEMA_SEARCH, TOOL_RUN_SQL]

MODE_TOOLS = {
    "vanilla": TOOLS_VANILLA,
    "mcp": TOOLS_MCP,
}
