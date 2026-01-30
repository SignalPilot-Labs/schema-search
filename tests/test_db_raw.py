"""Raw database connection test for Databricks."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


@pytest.fixture(scope="module")
def databricks_engine():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    url = os.getenv("DATABASE_DATABRICKS_URL")
    if not url:
        pytest.skip("DATABASE_DATABRICKS_URL not set in tests/.env")

    engine = create_engine(
        url,
        connect_args={
            "user_agent_entry": "schema-search",
            "_retry_stop_after_attempts_count": 1,
            "_socket_timeout": 10,
        },
    )
    return engine


def test_databricks_raw_connection(databricks_engine):
    """Test raw Databricks connection."""
    print(f"\nEngine URL: {databricks_engine.url}")

    with databricks_engine.connect() as conn:
        result = conn.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row[0] == 1
        print("Raw connection test passed")
