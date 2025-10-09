import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from sqlalchemy.engine import Engine

from schema_search.metadata_extractor import MetadataExtractor
from schema_search.chunker import Chunker, Chunk
from schema_search.embedding_manager import EmbeddingManager
from schema_search.graph_builder import GraphBuilder
from schema_search.ranker import Ranker

logger = logging.getLogger(__name__)


class SchemaSearch:
    def __init__(self, engine: Engine, config_path: Optional[str] = None):
        self.engine = engine
        self.config = self._load_config(config_path)
        self._setup_logging()

        self.metadata_dict: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Chunk] = []

        self.metadata_extractor = MetadataExtractor(engine, self.config)
        self.chunker = Chunker(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.graph_builder = GraphBuilder(self.config)
        self.ranker = Ranker(self.config)

    def _setup_logging(self):
        level = getattr(logging, self.config.get("logging", {}).get("level", "INFO"))
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yml")

        with open(config_path) as f:
            return yaml.safe_load(f)

    def index(self):
        logger.info("Starting schema indexing")
        self.metadata_dict = self._load_or_extract_metadata()
        self.graph_builder.build(self.metadata_dict)
        self.chunks = self.chunker.chunk_metadata(self.metadata_dict)
        self.embedding_manager.load_or_generate(self.chunks)
        self.ranker.build_bm25(self.chunks)
        logger.info(
            f"Indexing complete: {len(self.metadata_dict)} tables, {len(self.chunks)} chunks"
        )

    def _load_or_extract_metadata(self) -> Dict[str, Dict[str, Any]]:
        cache_dir = Path(self.config["embedding"]["cache_dir"])
        cache_dir.mkdir(exist_ok=True)
        metadata_cache = cache_dir / "metadata.json"

        if metadata_cache.exists():
            logger.debug(f"Loading metadata from cache: {metadata_cache}")
            with open(metadata_cache) as f:
                return json.load(f)

        logger.info("Extracting schema metadata from database")
        metadata_dict = self.metadata_extractor.extract()

        with open(metadata_cache, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        return metadata_dict

    def search(self, query: str) -> List[Dict[str, Any]]:
        if self.embedding_manager.embeddings is None:
            raise RuntimeError("Embeddings not generated. Call index() before search()")

        logger.info(f"Searching: {query}")
        query_embedding = self.embedding_manager.encode_query(query)
        ranked_chunks = self.ranker.rank(
            query, query_embedding, self.embedding_manager.embeddings
        )

        initial_top_k = self.config["search"]["initial_top_k"]
        top_chunk_indices = [idx for idx, _, _, _ in ranked_chunks[:initial_top_k]]

        initial_tables = set()
        for idx in top_chunk_indices:
            chunk = self.chunks[idx]
            initial_tables.add(chunk.table_name)

        expanded_tables = set(initial_tables)
        hops = self.config["search"]["graph_expand_hops"]
        for table in initial_tables:
            neighbors = self.graph_builder.get_neighbors(table, hops)
            expanded_tables.update(neighbors)

        table_chunks = {}
        for idx, chunk in enumerate(self.chunks):
            if chunk.table_name in expanded_tables:
                if chunk.table_name not in table_chunks:
                    table_chunks[chunk.table_name] = []
                table_chunks[chunk.table_name].append(idx)

        rerank_chunk_indices = []
        for table_name in expanded_tables:
            rerank_chunk_indices.extend(table_chunks.get(table_name, []))

        if self.embedding_manager.embeddings is None:
            raise RuntimeError("Embeddings not available for reranking")
        rerank_embeddings = self.embedding_manager.embeddings[rerank_chunk_indices]
        rerank_chunks = [self.chunks[idx] for idx in rerank_chunk_indices]

        temp_ranker = Ranker(self.config)
        temp_ranker.chunks = rerank_chunks
        temp_ranker.build_bm25(rerank_chunks)

        reranked = temp_ranker.rank(query, query_embedding, rerank_embeddings)

        rerank_top_k = self.config["search"]["rerank_top_k"]
        final_table_chunk_map = temp_ranker.get_top_tables_from_chunks(
            reranked, rerank_top_k
        )

        results = []
        for table_name, chunk_indices in final_table_chunk_map.items():
            max_score = max(reranked[idx][1] for idx in chunk_indices)

            results.append(
                {
                    "table": table_name,
                    "score": float(max_score),
                    "schema": self.metadata_dict[table_name],
                    "matched_chunks": [
                        rerank_chunks[idx].content for idx in chunk_indices
                    ],
                    "related_tables": list(
                        self.graph_builder.get_neighbors(table_name, hops)
                    ),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Found {len(results)} results")

        return results
