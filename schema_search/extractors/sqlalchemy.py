"""SQLAlchemy-based schema extractor for PostgreSQL, MySQL, Snowflake, BigQuery."""

from typing import Dict, List, Any
from sqlalchemy import inspect

from schema_search.extractors.base import BaseExtractor
from schema_search.types import (
    DBSchema,
    TableSchema,
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    ConstraintInfo,
    CheckConstraintInfo,
)


class SQLAlchemyExtractor(BaseExtractor):
    """Extracts schema using SQLAlchemy's inspector API."""

    def extract(self) -> DBSchema:
        inspector = inspect(self.engine)
        result: DBSchema = {}

        for schema_name in inspector.get_schema_names():
            if self._should_skip_schema(schema_name):
                continue

            result[schema_name] = {}
            for table_name in inspector.get_table_names(schema=schema_name):
                result[schema_name][table_name] = self._extract_table(
                    inspector, table_name, schema_name
                )

        return result

    def _extract_table(
        self, inspector, table_name: str, schema_name: str
    ) -> TableSchema:
        pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name)

        return {
            "name": table_name,
            "schema": schema_name,
            "primary_keys": pk_constraint["constrained_columns"],
            "columns": (
                self._extract_columns(
                    inspector.get_columns(table_name, schema=schema_name)
                )
                if self._include_columns()
                else None
            ),
            "foreign_keys": (
                self._extract_foreign_keys(
                    inspector.get_foreign_keys(table_name, schema=schema_name),
                    schema_name,
                )
                if self._include_foreign_keys()
                else None
            ),
            "indices": (
                self._extract_indices(
                    inspector.get_indexes(table_name, schema=schema_name)
                )
                if self._include_indices()
                else None
            ),
            "unique_constraints": (
                self._extract_constraints(
                    inspector.get_unique_constraints(table_name, schema=schema_name)
                )
                if self._include_constraints()
                else None
            ),
            "check_constraints": (
                self._extract_check_constraints(
                    inspector.get_check_constraints(table_name, schema=schema_name)
                )
                if self._include_constraints()
                else None
            ),
        }

    def _extract_columns(self, columns: List[Dict[str, Any]]) -> List[ColumnInfo]:
        return [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"],
                "default": str(col["default"]) if col["default"] else None,
            }
            for col in columns
        ]

    def _extract_foreign_keys(
        self, foreign_keys: List[Dict[str, Any]], default_schema: str
    ) -> List[ForeignKeyInfo]:
        return [
            {
                "constrained_columns": fk["constrained_columns"],
                "referred_schema": fk.get("referred_schema") or default_schema,
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"],
            }
            for fk in foreign_keys
        ]

    def _extract_indices(self, indices: List[Dict[str, Any]]) -> List[IndexInfo]:
        return [
            {
                "name": idx["name"] or f"idx_{i}",
                "columns": idx["column_names"],
                "unique": idx["unique"],
            }
            for i, idx in enumerate(indices)
        ]

    def _extract_constraints(
        self, constraints: List[Dict[str, Any]]
    ) -> List[ConstraintInfo]:
        return [
            {
                "name": constraint["name"],
                "columns": constraint["column_names"],
            }
            for constraint in constraints
        ]

    def _extract_check_constraints(
        self, constraints: List[Dict[str, Any]]
    ) -> List[CheckConstraintInfo]:
        return [
            {
                "name": constraint["name"],
                "sqltext": constraint["sqltext"],
            }
            for constraint in constraints
        ]
