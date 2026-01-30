"""Databricks-specific schema extractor using information_schema queries."""

import logging
from typing import Dict, List, Any, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from schema_search.extractors.base import BaseExtractor
from schema_search.types import DBSchema, ColumnInfo, ForeignKeyInfo

logger = logging.getLogger(__name__)

TableKey = Tuple[str, str]  # (schema_name, table_name)


class DatabricksExtractor(BaseExtractor):
    """Extracts schema from Databricks using information_schema queries."""

    def __init__(self, engine: Engine, config: Dict[str, Any]):
        super().__init__(engine, config)
        self.catalog = engine.url.query["catalog"]

    def extract(self) -> DBSchema:
        logger.info(f"Extracting from Databricks catalog: {self.catalog}")
        tables = self._get_tables()
        logger.info(f"Found {len(tables)} tables")

        all_columns = self._get_all_columns() if self._include_columns() else {}
        all_primary_keys = self._get_all_primary_keys()
        all_foreign_keys = self._get_all_foreign_keys() if self._include_foreign_keys() else {}

        result: DBSchema = {}
        for table_name, schema_name in tables:
            if schema_name not in result:
                result[schema_name] = {}

            table_key: TableKey = (schema_name, table_name)
            result[schema_name][table_name] = {
                "name": table_name,
                "schema": schema_name,
                "primary_keys": all_primary_keys.get(table_key, []),
                "columns": all_columns.get(table_key),
                "foreign_keys": all_foreign_keys.get(table_key),
                "indices": None,
                "unique_constraints": None,
                "check_constraints": None,
            }

        return result

    def _get_tables(self) -> List[Tuple[str, str]]:
        query = text("""
            SELECT table_name, table_schema
            FROM system.information_schema.tables
            WHERE table_catalog = :catalog
            AND table_schema NOT IN ('information_schema', 'sys')
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"catalog": self.catalog})
            rows = [(row[0], row[1]) for row in result]
            logger.debug(f"Found {len(rows)} tables in catalog {self.catalog}")
            return rows

    def _get_all_columns(self) -> Dict[TableKey, List[ColumnInfo]]:
        query = text(f"""
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default,
                ordinal_position
            FROM {self.catalog}.information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'sys')
            ORDER BY table_schema, table_name, ordinal_position
        """)

        columns_by_table: Dict[TableKey, List[ColumnInfo]] = {}
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                table_key: TableKey = (row[0], row[1])
                if table_key not in columns_by_table:
                    columns_by_table[table_key] = []

                columns_by_table[table_key].append({
                    "name": row[2],
                    "type": row[3],
                    "nullable": row[4] == "YES",
                    "default": row[5],
                })

        return columns_by_table

    def _get_all_primary_keys(self) -> Dict[TableKey, List[str]]:
        query = text(f"""
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name,
                kcu.ordinal_position
            FROM {self.catalog}.information_schema.table_constraints tc
            JOIN {self.catalog}.information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                AND tc.table_name = kcu.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
            ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
        """)

        pks_by_table: Dict[TableKey, List[str]] = {}
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                table_key: TableKey = (row[0], row[1])
                if table_key not in pks_by_table:
                    pks_by_table[table_key] = []
                pks_by_table[table_key].append(row[2])

        return pks_by_table

    def _get_all_foreign_keys(self) -> Dict[TableKey, List[ForeignKeyInfo]]:
        query = text(f"""
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name,
                ccu.table_schema AS foreign_table_schema,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM {self.catalog}.information_schema.table_constraints tc
            JOIN {self.catalog}.information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN {self.catalog}.information_schema.referential_constraints rc
                ON tc.constraint_name = rc.constraint_name
                AND tc.table_schema = rc.constraint_schema
            JOIN {self.catalog}.information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name = ccu.constraint_name
                AND rc.unique_constraint_schema = ccu.constraint_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """)

        fks_by_table: Dict[TableKey, Dict[TableKey, ForeignKeyInfo]] = {}
        with self.engine.connect() as conn:
            result = conn.execute(query)

            for row in result:
                table_key: TableKey = (row[0], row[1])
                col_name = row[2]
                ref_schema = row[3]
                ref_table = row[4]
                ref_col = row[5]
                ref_key: TableKey = (ref_schema, ref_table)

                if table_key not in fks_by_table:
                    fks_by_table[table_key] = {}

                if ref_key not in fks_by_table[table_key]:
                    fks_by_table[table_key][ref_key] = {
                        "constrained_columns": [],
                        "referred_schema": ref_schema,
                        "referred_table": ref_table,
                        "referred_columns": [],
                    }

                fks_by_table[table_key][ref_key]["constrained_columns"].append(col_name)
                fks_by_table[table_key][ref_key]["referred_columns"].append(ref_col)

        return {k: list(v.values()) for k, v in fks_by_table.items()}
