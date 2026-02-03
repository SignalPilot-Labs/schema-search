"""Utility functions for schema-search."""

import logging
import os
import time
from functools import wraps
from importlib import import_module
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse, urlunparse

from sqlalchemy import Engine, create_engine

from schema_search.types import SearchResult

logger = logging.getLogger(__name__)


def time_it(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        if isinstance(result, dict):
            result["latency_sec"] = round(elapsed, 3)
        elif isinstance(result, SearchResult):
            result.latency_sec = round(elapsed, 3)

        return result

    return wrapper


def lazy_import_check(module_name: str, extra_name: str, feature: str) -> Any:
    """Lazily import a module and provide helpful error if missing.

    Args:
        module_name: Python module to import (e.g., "sentence_transformers")
        extra_name: pip extra name (e.g., "semantic")
        feature: User-facing feature description (e.g., "semantic search")

    Returns:
        Imported module

    Raises:
        ImportError: With installation instructions if module not found
    """
    try:
        return import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"'{module_name}' is required for {feature}. "
            f"Install with: pip install schema-search[{extra_name}]"
        ) from e


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on config settings.

    Args:
        config: Configuration dictionary with logging.level key.
    """
    level = getattr(logging, config["logging"]["level"])
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.setLevel(level)


def _load_snowflake_private_key(key_path: str) -> bytes:
    """Load private key from file for Snowflake key-pair authentication.

    Args:
        key_path: Path to the private key file (supports ~ expansion).

    Returns:
        Private key bytes in DER format for Snowflake connector.
    """
    serialization = lazy_import_check(
        "cryptography.hazmat.primitives.serialization",
        "snowflake",
        "Snowflake key-pair authentication",
    )
    default_backend = lazy_import_check(
        "cryptography.hazmat.backends",
        "snowflake",
        "Snowflake key-pair authentication",
    ).default_backend

    with open(os.path.expanduser(key_path), "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend(),
        )
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _parse_snowflake_url(url: str) -> tuple[str, str | None]:
    """Parse Snowflake URL and extract private_key_path parameter.

    Args:
        url: Snowflake connection URL with optional private_key_path param.

    Returns:
        Tuple of (clean_url without private_key_path, key_path or None).
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    key_path = params.pop("private_key_path", [None])[0]

    new_query = "&".join(f"{k}={v[0]}" for k, v in params.items())
    clean_url = urlunparse(parsed._replace(query=new_query))

    return clean_url, key_path


def _create_snowflake_engine(url: str) -> Engine:
    """Create SQLAlchemy engine for Snowflake with key-pair auth support.

    Args:
        url: Snowflake URL, optionally with private_key_path parameter.

    Returns:
        SQLAlchemy Engine configured for Snowflake.
    """
    clean_url, key_path = _parse_snowflake_url(url)

    if key_path:
        private_key = _load_snowflake_private_key(key_path)
        return create_engine(clean_url, connect_args={"private_key": private_key})

    return create_engine(clean_url)


def _create_databricks_engine(url: str) -> Engine:
    """Create SQLAlchemy engine for Databricks with required connect_args.

    Args:
        url: Databricks connection URL.

    Returns:
        SQLAlchemy Engine configured for Databricks.
    """
    return create_engine(url, connect_args={"user_agent_entry": "schema-search"})


def create_engine_from_url(url: str) -> Engine:
    """Create SQLAlchemy engine from URL with DB-specific handling.

    Handles special cases:
    - Snowflake: Extracts private_key_path param and loads key for auth.
    - Databricks: Adds required user_agent_entry to connect_args.
    - Others: Standard create_engine call.

    Args:
        url: Database connection URL.

    Returns:
        SQLAlchemy Engine for the specified database.
    """
    parsed = urlparse(url)
    dialect = parsed.scheme.split("+")[0]

    if dialect == "snowflake":
        return _create_snowflake_engine(url)
    elif dialect == "databricks":
        return _create_databricks_engine(url)
    else:
        return create_engine(url)
