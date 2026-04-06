"""Base class for schema extractors."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from sqlalchemy.engine import Engine

from schema_search.constants import SKIP_SCHEMAS
from schema_search.types import DBSchema


class BaseExtractor(ABC):
    """Base class for database schema extraction."""

    def __init__(self, engine: Engine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config

    @abstractmethod
    def extract(self) -> DBSchema:
        """Extract all schemas and tables from the database.

        Returns:
            Nested dict: {schema_name: {table_name: TableSchema}}
        """
        raise NotImplementedError

    def _should_skip_schema(self, schema_name: str) -> bool:
        return schema_name.lower() in SKIP_SCHEMAS

    def _include_columns(self) -> bool:
        return self.config["schema"]["include_columns"]

    def _include_foreign_keys(self) -> bool:
        return self.config["schema"]["include_foreign_keys"]

    def _include_indices(self) -> bool:
        return self.config["schema"]["include_indices"]

    def _include_constraints(self) -> bool:
        return self.config["schema"]["include_constraints"]
