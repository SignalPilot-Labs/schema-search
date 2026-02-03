import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import Engine, text

from schema_search import SchemaSearch
from schema_search.utils.utils import create_engine_from_url


@pytest.fixture(scope="module")
def snowflake_url() -> str:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    url = os.getenv("DATABASE_SNOWFLAKE_URL")
    if not url:
        pytest.skip("DATABASE_SNOWFLAKE_URL not set in tests/.env")

    return url


@pytest.fixture(scope="module")
def snowflake_engine(snowflake_url: str) -> Engine:
    return create_engine_from_url(snowflake_url)


@pytest.mark.timeout(60)
def test_snowflake_basic_query(snowflake_engine: Engine) -> None:
    """Test basic Snowflake connectivity."""
    print("\nTesting basic Snowflake query...")
    print(f"Engine URL: {snowflake_engine.url}")
    print("Attempting to connect...")

    try:
        with snowflake_engine.connect() as conn:
            print("Connection established, executing query...")
            result = conn.execute(text("SELECT 1 as test"))
            print("Query executed, fetching result...")
            row = result.fetchone()
            assert row is not None and row[0] == 1
        print("✓ Basic query works")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_snowflake_list_tables(snowflake_engine: Engine) -> None:
    """Test listing tables from Snowflake."""
    print("\nListing tables from Snowflake...")

    query = text("""
        SELECT table_catalog, table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
        LIMIT 5
    """)

    with snowflake_engine.connect() as conn:
        result = conn.execute(query)
        rows = list(result)

    print(f"✓ Found {len(rows)} tables:")
    for row in rows:
        print(f"  - {row[0]}.{row[1]}.{row[2]}")

    assert len(rows) > 0, "No tables found"


def test_snowflake_connection(snowflake_engine: Engine) -> None:
    """Test full SchemaSearch indexing and search."""
    print("\nTesting SchemaSearch with Snowflake...")
    search = SchemaSearch(snowflake_engine)

    print("Indexing...")
    search.index(force=True)
    print(f"✓ Indexed {len(search.schemas)} tables")

    print("Searching...")
    results = search.search("user")
    print(f"✓ Search complete: found {len(results.results)} results")

    assert len(results.results) > 0
