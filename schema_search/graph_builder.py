import logging
import pickle
from pathlib import Path
from typing import Set

import networkx as nx

from schema_search.types import DBSchema

logger = logging.getLogger(__name__)


def make_table_key(schema_name: str, table_name: str) -> str:
    return f"{schema_name}.{table_name}"


class GraphBuilder:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.graph: nx.DiGraph

    def build(self, schemas: DBSchema, force: bool) -> None:
        cache_file = self.cache_dir / "graph.pkl"

        if not force and cache_file.exists():
            self._load_from_cache(cache_file)
        else:
            self._build_and_cache(schemas, cache_file)

    def _load_from_cache(self, cache_file: Path) -> None:
        logger.debug(f"Loading graph from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            self.graph = pickle.load(f)

    def _build_and_cache(self, schemas: DBSchema, cache_file: Path) -> None:
        logger.info("Building foreign key relationship graph")
        self.graph = nx.DiGraph()

        for schema_name, tables in schemas.items():
            for table_name, table_schema in tables.items():
                node_name = make_table_key(schema_name, table_name)
                self.graph.add_node(node_name)

        for schema_name, tables in schemas.items():
            for table_name, table_schema in tables.items():
                foreign_keys = table_schema.get("foreign_keys")
                if not foreign_keys:
                    continue

                source = make_table_key(schema_name, table_name)
                for fk in foreign_keys:
                    ref_schema = fk["referred_schema"]
                    ref_table = fk["referred_table"]
                    target = make_table_key(ref_schema, ref_table)

                    if target in self.graph:
                        self.graph.add_edge(source, target)

        with open(cache_file, "wb") as f:
            pickle.dump(self.graph, f)

    def get_neighbors(self, qualified_table: str, hops: int) -> Set[str]:
        if qualified_table not in self.graph:
            return set()

        neighbors: Set[str] = set()

        forward = nx.single_source_shortest_path_length(
            self.graph, qualified_table, cutoff=hops
        )
        neighbors.update(forward.keys())

        backward = nx.single_source_shortest_path_length(
            self.graph.reverse(), qualified_table, cutoff=hops
        )
        neighbors.update(backward.keys())

        neighbors.discard(qualified_table)

        return neighbors
