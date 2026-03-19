"""Constants for Spider 2.0-Snow evaluation."""

MODEL_NAME = "claude-opus-4-6"
MAX_TOOL_TURNS = 10
MAX_TOKENS = 4096
TEMPERATURE = 0
SEARCH_LIMIT = 10
SEARCH_HOPS = 1

SQL_CODE_BLOCK_PATTERN = r"```sql\s*(.*?)\s*```"

TOOL_SCHEMA_SEARCH = {
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

TOOL_GET_SCHEMA = {
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

# Vanilla: only get_schema. MCP: get_schema + schema_search.
TOOLS_VANILLA = [TOOL_GET_SCHEMA]
TOOLS_MCP = [TOOL_GET_SCHEMA, TOOL_SCHEMA_SEARCH]
