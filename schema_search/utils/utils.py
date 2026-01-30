"""Utility functions for schema-search."""

import logging
import time
from functools import wraps
from importlib import import_module
from typing import Any, Dict

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
