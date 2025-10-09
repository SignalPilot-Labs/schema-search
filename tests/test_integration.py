import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine

from schema_search import SchemaSearch


@pytest.fixture(scope="module")
def database_url():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set in tests/.env file")

    return url


@pytest.fixture(scope="module")
def search_engine(database_url):
    engine = create_engine(database_url)
    search = SchemaSearch(engine)
    return search


def test_index_creation(search_engine):
    """Test that the index can be built successfully."""
    stats = search_engine.index(force=True)

    assert len(search_engine.metadata_dict) > 0, "No tables found in database"
    assert len(search_engine.chunks) > 0, "No chunks generated"
    assert (
        search_engine.embedding_manager.embeddings is not None
    ), "Embeddings not generated"

    print(f"\nIndexing: {stats}")


def test_search_user_information(search_engine):
    """Test searching for user-related information in the schema."""
    search_engine.index(force=False)

    query = "which table has user email address?"
    response = search_engine.search(query)

    results = response["results"]

    for result in results:
        print(f"Result: {result['table']} (score: {result['score']:.3f})")
        # print(f"Related tables: {result['related_tables']}")
        # print("-" * 100)

    assert len(results) > 0, "No search results returned"

    top_result = results[0]
    assert "table" in top_result, "Result missing 'table' field"
    assert "score" in top_result, "Result missing 'score' field"
    assert "schema" in top_result, "Result missing 'schema' field"
    assert "matched_chunks" in top_result, "Result missing 'matched_chunks' field"
    assert "related_tables" in top_result, "Result missing 'related_tables' field"

    assert top_result["score"] > 0, "Top result has invalid score"

    print(f"\nTop result: {top_result['table']} (score: {top_result['score']:.3f})")
    print(f"Related tables: {top_result['related_tables']}")
    print(f"Search latency: {response['latency_sec']}s")
