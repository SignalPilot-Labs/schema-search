import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Set

import networkx as nx

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config["embedding"]["cache_dir"])
        self.cache_dir.mkdir(exist_ok=True)
        self.graph = None

    def build(self, metadata_dict: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
        cache_file = self.cache_dir / "graph.pkl"

        if cache_file.exists():
            return self._load_from_cache(cache_file)

        return self._build_and_cache(metadata_dict, cache_file)

    def _load_from_cache(self, cache_file: Path) -> nx.DiGraph:
        logger.debug(f"Loading graph from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            self.graph = pickle.load(f)
        return self.graph

    def _build_and_cache(
        self, metadata_dict: Dict[str, Dict[str, Any]], cache_file: Path
    ) -> nx.DiGraph:
        logger.debug("Building foreign key relationship graph")
        self.graph = nx.DiGraph()

        for table_name, table_info in metadata_dict.items():
            self.graph.add_node(table_name, **table_info)

        for table_name, table_info in metadata_dict.items():
            for fk in table_info.get("foreign_keys", []):
                referred_table = fk["referred_table"]
                if referred_table in self.graph:
                    self.graph.add_edge(table_name, referred_table, **fk)

        with open(cache_file, "wb") as f:
            pickle.dump(self.graph, f)

        return self.graph

    def get_neighbors(self, table_name: str, hops: int) -> Set[str]:
        if self.graph is None:
            raise RuntimeError("Graph not built")

        if table_name not in self.graph:
            return set()

        neighbors = set()

        forward = nx.single_source_shortest_path_length(
            self.graph, table_name, cutoff=hops
        )
        neighbors.update(forward.keys())

        backward = nx.single_source_shortest_path_length(
            self.graph.reverse(), table_name, cutoff=hops
        )
        neighbors.update(backward.keys())

        neighbors.discard(table_name)

        return neighbors
