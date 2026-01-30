"""Databricks-specific schema extractor using information_schema queries."""

import logging
from typing import Dict, List, Tuple

from sqlalchemy import text

from schema_search.extractors.base import BaseExtractor
from schema_search.types import DBSchema, ColumnInfo, ForeignKeyInfo

logger = logging.getLogger(__name__)

TableKey = Tuple[str, str, str]  # (catalog, schema, table)

SKIP_CATALOGS = {"system", "samples", "hive_metastore"}


class DatabricksExtractor(BaseExtractor):
    """Extracts schema from Databricks using information_schema queries."""

    def extract(self) -> DBSchema:
        catalogs = self._get_catalogs()
        logger.info(f"Extracting from Databricks catalogs: {catalogs}")

        tables = self._get_tables(catalogs)
        logger.info(f"Found {len(tables)} tables")

        all_columns = self._get_all_columns(catalogs) if self._include_columns() else {}
        all_primary_keys = self._get_all_primary_keys(catalogs)
        all_foreign_keys = self._get_all_foreign_keys(catalogs) if self._include_foreign_keys() else {}

        result: DBSchema = {}
        for catalog, schema, table_name in tables:
            schema_key = f"{catalog}.{schema}"
            if schema_key not in result:
                result[schema_key] = {}

            table_key: TableKey = (catalog, schema, table_name)
            result[schema_key][table_name] = {
                "name": table_name,
                "schema": schema_key,
                "primary_keys": all_primary_keys.get(table_key, []),
                "columns": all_columns.get(table_key),
                "foreign_keys": all_foreign_keys.get(table_key),
                "indices": None,
                "unique_constraints": None,
                "check_constraints": None,
            }

        return result

    def _get_catalogs(self) -> List[str]:
        """Get all catalogs to extract."""
        query = text("""
            SELECT catalog_name
            FROM system.information_schema.catalogs
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result if row[0] not in SKIP_CATALOGS]

    def _get_tables(self, catalogs: List[str]) -> List[Tuple[str, str, str]]:
        """Get all tables from specified catalogs."""
        all_tables: List[Tuple[str, str, str]] = []

        with self.engine.connect() as conn:
            for catalog in catalogs:
                query = text("""
                    SELECT table_catalog, table_schema, table_name
                    FROM system.information_schema.tables
                    WHERE table_catalog = :catalog
                """)
                result = conn.execute(query, {"catalog": catalog})
                for row in result:
                    if not self._should_skip_schema(row[1]):
                        all_tables.append((row[0], row[1], row[2]))

                logger.debug(f"Found tables in catalog {catalog}")

        return all_tables

    def _get_all_columns(self, catalogs: List[str]) -> Dict[TableKey, List[ColumnInfo]]:
        columns_by_table: Dict[TableKey, List[ColumnInfo]] = {}

        with self.engine.connect() as conn:
            for catalog in catalogs:
                query = text(f"""
                    SELECT
                        table_schema,
                        table_name,
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM {catalog}.information_schema.columns
                    ORDER BY table_schema, table_name, ordinal_position
                """)
                result = conn.execute(query)
                for row in result:
                    if self._should_skip_schema(row[0]):
                        continue

                    table_key: TableKey = (catalog, row[0], row[1])
                    columns_by_table.setdefault(table_key, []).append({
                        "name": row[2],
                        "type": row[3],
                        "nullable": row[4] == "YES",
                        "default": row[5],
                    })

        return columns_by_table

    def _get_all_primary_keys(self, catalogs: List[str]) -> Dict[TableKey, List[str]]:
        pks_by_table: Dict[TableKey, List[str]] = {}

        with self.engine.connect() as conn:
            for catalog in catalogs:
                query = text(f"""
                    SELECT
                        tc.table_schema,
                        tc.table_name,
                        kcu.column_name
                    FROM {catalog}.information_schema.table_constraints tc
                    JOIN {catalog}.information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                        AND tc.table_name = kcu.table_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                    ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
                """)
                result = conn.execute(query)
                for row in result:
                    if self._should_skip_schema(row[0]):
                        continue

                    table_key: TableKey = (catalog, row[0], row[1])
                    pks_by_table.setdefault(table_key, []).append(row[2])

        return pks_by_table

    def _get_all_foreign_keys(self, catalogs: List[str]) -> Dict[TableKey, List[ForeignKeyInfo]]:
        fks_by_table: Dict[TableKey, Dict[str, ForeignKeyInfo]] = {}

        with self.engine.connect() as conn:
            for catalog in catalogs:
                query = text(f"""
                    SELECT
                        tc.table_schema,
                        tc.table_name,
                        kcu.column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM {catalog}.information_schema.table_constraints tc
                    JOIN {catalog}.information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN {catalog}.information_schema.referential_constraints rc
                        ON tc.constraint_name = rc.constraint_name
                        AND tc.table_schema = rc.constraint_schema
                    JOIN {catalog}.information_schema.constraint_column_usage ccu
                        ON rc.unique_constraint_name = ccu.constraint_name
                        AND rc.unique_constraint_schema = ccu.constraint_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                """)
                result = conn.execute(query)
                for row in result:
                    if self._should_skip_schema(row[0]):
                        continue

                    table_key: TableKey = (catalog, row[0], row[1])
                    ref_schema = f"{catalog}.{row[3]}"
                    ref_table = row[4]
                    ref_key = f"{ref_schema}.{ref_table}"

                    if table_key not in fks_by_table:
                        fks_by_table[table_key] = {}

                    fks_by_table[table_key].setdefault(ref_key, {
                        "constrained_columns": [],
                        "referred_schema": ref_schema,
                        "referred_table": ref_table,
                        "referred_columns": [],
                    })
                    fks_by_table[table_key][ref_key]["constrained_columns"].append(row[2])
                    fks_by_table[table_key][ref_key]["referred_columns"].append(row[5])

        return {k: list(v.values()) for k, v in fks_by_table.items()}
