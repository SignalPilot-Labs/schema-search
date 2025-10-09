from typing import Dict, List, Any
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


class MetadataExtractor:
    def __init__(self, engine: Engine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config

    def extract(self) -> Dict[str, Dict[str, Any]]:
        inspector = inspect(self.engine)
        metadata_dict = {}

        for table_name in inspector.get_table_names():
            table_info = self._extract_table(inspector, table_name)
            metadata_dict[table_name] = table_info

        return metadata_dict

    def _extract_table(self, inspector, table_name: str) -> Dict[str, Any]:
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        indices = inspector.get_indexes(table_name)
        pk_constraint = inspector.get_pk_constraint(table_name)
        unique_constraints = inspector.get_unique_constraints(table_name)
        check_constraints = inspector.get_check_constraints(table_name)

        table_info = {
            "name": table_name,
            "columns": self._extract_columns(columns),
            "primary_keys": pk_constraint.get("constrained_columns", []),
            "foreign_keys": [],
            "indices": [],
            "unique_constraints": [],
            "check_constraints": [],
        }

        if self.config["metadata"]["include_foreign_keys"]:
            table_info["foreign_keys"] = self._extract_foreign_keys(foreign_keys)

        if self.config["metadata"]["include_indices"]:
            table_info["indices"] = self._extract_indices(indices)
            table_info["unique_constraints"] = self._extract_constraints(
                unique_constraints
            )
            table_info["check_constraints"] = self._extract_constraints(
                check_constraints
            )

        return table_info

    def _extract_columns(self, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted = []
        for col in columns:
            extracted.append(
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "default": str(col.get("default")) if col.get("default") else None,
                }
            )
        return extracted

    def _extract_foreign_keys(
        self, foreign_keys: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        extracted = []
        for fk in foreign_keys:
            extracted.append(
                {
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                }
            )
        return extracted

    def _extract_indices(self, indices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted = []
        for idx in indices:
            extracted.append(
                {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx["unique"],
                }
            )
        return extracted

    def _extract_constraints(
        self, constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        extracted = []
        for constraint in constraints:
            extracted.append(
                {
                    "name": constraint.get("name"),
                    "columns": constraint.get("column_names", []),
                }
            )
        return extracted
