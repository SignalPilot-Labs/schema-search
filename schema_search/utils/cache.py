"""Cache I/O utilities for schema and chunk persistence."""

import json
import logging
from pathlib import Path
from typing import List, Optional

from schema_search.types import Chunk, DBSchema

logger = logging.getLogger(__name__)


def load_schema(cache_dir: Path) -> Optional[DBSchema]:
    """Load cached schema from disk.

    Args:
        cache_dir: Directory containing cache files.

    Returns:
        Cached schema or None if not found.
    """
    schema_cache = cache_dir / "metadata.json"

    if not schema_cache.exists():
        logger.debug("Schema cache missing")
        return None

    with open(schema_cache) as f:
        return json.load(f)


def save_schema(cache_dir: Path, schema: DBSchema) -> None:
    """Save schema to cache.

    Args:
        cache_dir: Directory for cache files.
        schema: Schema to save.
    """
    schema_cache = cache_dir / "metadata.json"
    with open(schema_cache, "w") as f:
        json.dump(schema, f, indent=2)


def schema_changed(cached: Optional[DBSchema], current: DBSchema) -> bool:
    """Check if schema has changed.

    Args:
        cached: Previously cached schema (or None).
        current: Current schema from database.

    Returns:
        True if schema changed or no cache exists.
    """
    if cached is None:
        return True
    if cached != current:
        logger.debug("Cached schema differs from current schema")
        return True
    logger.debug("Schema matches cached version")
    return False


def load_chunks(cache_dir: Path) -> Optional[List[Chunk]]:
    """Load cached chunks from disk.

    Args:
        cache_dir: Directory containing cache files.

    Returns:
        List of chunks or None if not found or incompatible.
    """
    chunks_cache = cache_dir / "chunk_metadata.json"

    if not chunks_cache.exists():
        return None

    logger.info(f"Loading chunks from cache: {chunks_cache}")
    try:
        with open(chunks_cache) as f:
            chunk_data = json.load(f)
            return [
                Chunk(
                    catalog=c.get("catalog"),
                    schema_name=c["schema_name"],
                    table_name=c["table_name"],
                    content=c["content"],
                    chunk_id=c["chunk_id"],
                    token_count=c["token_count"],
                )
                for c in chunk_data
            ]
    except Exception as e:
        logger.warning(f"Failed to load chunks cache: {e}")
        return None


def save_chunks(cache_dir: Path, chunks: List[Chunk]) -> None:
    """Save chunks to cache.

    Args:
        cache_dir: Directory for cache files.
        chunks: Chunks to save.
    """
    chunks_cache = cache_dir / "chunk_metadata.json"
    with open(chunks_cache, "w") as f:
        chunk_data = [
            {
                "catalog": c.catalog,
                "schema_name": c.schema_name,
                "table_name": c.table_name,
                "content": c.content,
                "chunk_id": c.chunk_id,
                "token_count": c.token_count,
            }
            for c in chunks
        ]
        json.dump(chunk_data, f, indent=2)
