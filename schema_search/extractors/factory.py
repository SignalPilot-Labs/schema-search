"""Factory function for creating schema extractors."""

from typing import Dict, Any

from sqlalchemy.engine import Engine

from schema_search.extractors.base import BaseExtractor
from schema_search.extractors.sqlalchemy import SQLAlchemyExtractor
from schema_search.extractors.databricks import DatabricksExtractor


def create_extractor(engine: Engine, config: Dict[str, Any]) -> BaseExtractor:
    """Create appropriate extractor based on database dialect."""
    if engine.dialect.name == "databricks":
        return DatabricksExtractor(engine, config)
    return SQLAlchemyExtractor(engine, config)
