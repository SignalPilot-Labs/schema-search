#!/usr/bin/env python3
import logging
from typing import List, Optional

from fastmcp import FastMCP
from sqlalchemy import create_engine

from schema_search import SchemaSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("schema-search")


@mcp.tool()
def schema_search(
    query: str,
    schemas: Optional[List[str]] = None,
    catalogs: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> str:
    """Search database schema using natural language.

    Finds relevant database tables and their relationships by searching through schema metadata
    using semantic similarity. Expands results by traversing foreign key relationships.

    Args:
        query: Natural language question about database schema (e.g., 'tables related to payments')
        schemas: Optional list of schema names to filter results (e.g., ['public', 'sales']).
        catalogs: Optional list of catalog names to filter results (Databricks only).
        limit: Maximum number of table schemas to return in results. Uses config default if not specified.

    Returns:
        Formatted search results (markdown or json based on config)
    """
    if limit is None:
        limit = int(mcp.search_engine.config["output"]["limit"])  # type: ignore

    search_result = mcp.search_engine.search(  # type: ignore
        query, catalogs=catalogs, schemas=schemas, limit=limit
    )
    return str(search_result)


@mcp.tool()
def get_schema(
    schemas: Optional[List[str]] = None,
    catalogs: Optional[List[str]] = None,
) -> dict:
    """Get the full database schema structure.

    Returns the complete schema metadata including all tables, columns, foreign keys,
    and indices. Useful for understanding the database structure without searching.

    Args:
        schemas: Optional list of schema names to filter (e.g., ['public', 'sales']).
        catalogs: Optional list of catalog names to filter (Databricks only).

    Returns:
        Schema structure as nested dict: {schema_key: {table_name: TableSchema}}.
    """
    return mcp.search_engine.get_schema(catalogs=catalogs, schemas=schemas)  # type: ignore


def run_server(
    database_url: str,
    config_path: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
):
    engine = create_engine(database_url)

    mcp.search_engine = SchemaSearch(  # type: ignore
        engine,
        config_path=config_path,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )

    logger.info("Indexing database schema...")
    mcp.search_engine.index()  # type: ignore
    logger.info("Index ready")

    mcp.run()


def main():
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: schema-search <database_url> [config_path] [llm_api_key] [llm_base_url]"
        )
        sys.exit(1)

    database_url = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    llm_api_key = sys.argv[3] if len(sys.argv) > 3 else None
    llm_base_url = sys.argv[4] if len(sys.argv) > 4 else None

    run_server(database_url, config_path, llm_api_key, llm_base_url)


if __name__ == "__main__":
    main()
